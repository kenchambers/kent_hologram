"""
Unit tests for FaissAdapter and persistence layer.

Tests validate:
1. Correct index behavior (IndexFlatIP, normalization)
2. ID management and persistence
3. Metadata consistency
4. Error handling
5. Equivalence with InMemoryDatabase
"""

import json
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from hologram.persistence.faiss_adapter import FaissAdapter
from hologram.persistence.in_memory_db import InMemoryDatabase


class TestFaissAdapterBasics:
    """Test basic FaissAdapter operations."""

    def test_init_creates_empty_adapter(self):
        """Adapter should initialize with empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            assert adapter.vector_count == 0
            assert adapter._id_counter == 0
            assert len(adapter.metadata) == 0

    def test_init_invalid_dimensions(self):
        """Should reject invalid dimensions."""
        with pytest.raises(ValueError):
            FaissAdapter(dimensions=0, persist_path="/tmp")

        with pytest.raises(ValueError):
            FaissAdapter(dimensions=-1, persist_path="/tmp")

    def test_store_single_vector(self):
        """Should store vector and return ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            meta = {"source": "test"}

            vec_id = adapter.store(vec, meta)

            assert vec_id == 0
            assert adapter.vector_count == 1
            assert adapter._id_counter == 1
            assert 0 in adapter.metadata
            assert adapter.metadata[0] == meta

    def test_store_multiple_vectors(self):
        """IDs should increment sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vecs = [torch.randn(10000) for _ in range(5)]

            ids = [adapter.store(v, {"idx": i}) for i, v in enumerate(vecs)]

            assert ids == [0, 1, 2, 3, 4]
            assert adapter.vector_count == 5
            assert adapter._id_counter == 5

    def test_store_rejects_wrong_dimensions(self):
        """Should validate vector dimensionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            with pytest.raises(ValueError):
                vec = torch.randn(5000)  # Wrong size
                adapter.store(vec, {})

            with pytest.raises(ValueError):
                vec = torch.randn(10001)  # Wrong size
                adapter.store(vec, {})

    def test_store_rejects_zero_vector(self):
        """Should reject all-zero vectors (can't normalize)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.zeros(10000)

            with pytest.raises(ValueError):
                adapter.store(vec, {})

    def test_store_rejects_non_dict_metadata(self):
        """Should require dict metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)

            with pytest.raises(TypeError):
                adapter.store(vec, "not a dict")

            with pytest.raises(TypeError):
                adapter.store(vec, 123)

    def test_store_handles_1d_and_2d_vectors(self):
        """Should accept both (d,) and (1, d) shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            # 1D vector
            vec_1d = torch.randn(10000)
            id_1d = adapter.store(vec_1d, {"shape": "1d"})
            assert id_1d == 0

            # 2D vector with batch size 1
            vec_2d = torch.randn(1, 10000)
            id_2d = adapter.store(vec_2d, {"shape": "2d"})
            assert id_2d == 1

            assert adapter.vector_count == 2


class TestFaissAdapterQuery:
    """Test query operations."""

    def test_query_single_vector(self):
        """Should find stored vector in query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            vec_id = adapter.store(vec, {"data": "a"})

            # Query with same vector (should have highest similarity)
            results = adapter.query(vec, k=1)

            assert len(results) == 1
            result_id, similarity, meta = results[0]
            assert result_id == vec_id
            assert similarity > 0.99  # Should be very close to 1.0
            assert meta == {"data": "a"}

    def test_query_multiple_vectors(self):
        """Should return k most similar vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            # Store 5 random vectors
            vecs = [torch.randn(10000) for _ in range(5)]
            ids = [adapter.store(v, {"idx": i}) for i, v in enumerate(vecs)]

            # Query with first vector
            results = adapter.query(vecs[0], k=5)

            assert len(results) == 5
            # First result should be the query vector itself
            assert results[0][0] == ids[0]
            assert results[0][1] > 0.99

    def test_query_rejects_invalid_k(self):
        """Should validate k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {})

            with pytest.raises(ValueError):
                adapter.query(vec, k=0)

            with pytest.raises(ValueError):
                adapter.query(vec, k=-1)

            with pytest.raises(ValueError):
                adapter.query(vec, k=2)  # More than stored vectors

    def test_query_rejects_wrong_dimensions(self):
        """Should validate query vector dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {})

            with pytest.raises(ValueError):
                bad_vec = torch.randn(5000)
                adapter.query(bad_vec, k=1)

    def test_query_rejects_zero_vector(self):
        """Should reject all-zero query vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {})

            with pytest.raises(ValueError):
                zero_vec = torch.zeros(10000)
                adapter.query(zero_vec, k=1)

    def test_query_results_sorted_by_similarity(self):
        """Results should be sorted by similarity (highest first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            # Store orthogonal vectors
            base = torch.zeros(10000)
            base[0] = 1.0
            adapter.store(base, {"idx": 0})

            # Vector close to base
            close = base.clone()
            close[1:100] += torch.randn(99) * 0.01
            close = close / torch.norm(close)
            adapter.store(close, {"idx": 1})

            # Vector far from base
            far = torch.randn(10000)
            adapter.store(far, {"idx": 2})

            # Query with base
            results = adapter.query(base, k=3)

            # Results should be ordered: base, close, far
            similarities = [r[1] for r in results]
            assert similarities == sorted(similarities, reverse=True)


class TestFaissAdapterPersistence:
    """Test save/load persistence."""

    def test_save_creates_files(self):
        """Save should create index and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {"data": "a"})

            adapter.save()

            index_file = Path(tmpdir) / "index.faiss"
            metadata_file = Path(tmpdir) / "metadata.json"

            assert index_file.exists()
            assert metadata_file.exists()

    def test_metadata_json_contains_counter(self):
        """CRITICAL: Metadata must contain _id_counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {"data": "a"})

            adapter.save()

            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, 'r') as f:
                saved_data = json.load(f)

            # CRITICAL: This key must be saved
            assert '_id_counter' in saved_data
            assert saved_data['_id_counter'] == 1

    def test_load_restores_counter(self):
        """CRITICAL: Loading must restore _id_counter to prevent ID collisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Store 3 vectors
            adapter1 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            for i in range(3):
                vec = torch.randn(10000)
                adapter1.store(vec, {"idx": i})
            adapter1.save()

            # Session 2: Load and add new vector
            adapter2 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            adapter2.load()

            # CRITICAL: Next ID should be 3, not 0 (no collision)
            vec_new = torch.randn(10000)
            new_id = adapter2.store(vec_new, {"idx": 3})

            assert new_id == 3  # Would be 0 if counter not restored!
            assert adapter2._id_counter == 4

    def test_load_restores_metadata(self):
        """Loading should restore all metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Store vectors with metadata
            adapter1 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vecs = [torch.randn(10000) for _ in range(3)]
            expected_meta = [
                {"source": "a", "value": 1},
                {"source": "b", "value": 2},
                {"source": "c", "value": 3},
            ]
            for vec, meta in zip(vecs, expected_meta):
                adapter1.store(vec, meta)
            adapter1.save()

            # Session 2: Load and verify metadata
            adapter2 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            adapter2.load()

            assert len(adapter2.metadata) == 3
            for i in range(3):
                assert adapter2.metadata[i] == expected_meta[i]

    def test_load_nonexistent_raises_error(self):
        """Loading from nonexistent path should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            with pytest.raises(FileNotFoundError):
                adapter.load()

    def test_round_trip_persistence(self):
        """Full round-trip: store → save → load → query should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Create and save
            adapter1 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec1 = torch.randn(10000)
            vec2 = torch.randn(10000)
            id1 = adapter1.store(vec1, {"name": "first"})
            id2 = adapter1.store(vec2, {"name": "second"})
            adapter1.save()

            # Session 2: Load and query
            adapter2 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            adapter2.load()

            # Query with first vector
            results = adapter2.query(vec1, k=2)

            # First result should be the matching vector
            assert results[0][0] == id1
            assert results[0][2]["name"] == "first"


class TestFaissAdapterConsistency:
    """Test consistency validation."""

    def test_consistency_check_on_save(self):
        """Save should validate consistency before writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter.store(vec, {})

            # Corrupt state (should be caught)
            adapter._id_counter = 999  # Doesn't match vector count

            with pytest.raises(ValueError):
                adapter.save()

    def test_consistency_check_on_load(self):
        """Load should validate consistency and raise on mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid save first
            adapter1 = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            vec = torch.randn(10000)
            adapter1.store(vec, {})
            adapter1.save()

            # Corrupt metadata file
            metadata_file = Path(tmpdir) / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({"_id_counter": 99}, f)  # Wrong count

            # Load should detect corruption
            adapter2 = FaissAdapter(dimensions=10000, persist_path=tmpdir)

            with pytest.raises(ValueError):
                adapter2.load()


class TestComparisonWithInMemoryDatabase:
    """Verify FaissAdapter and InMemoryDatabase are equivalent."""

    def test_same_store_retrieve_behavior(self):
        """Both should store and retrieve identically."""
        vec = torch.randn(10000)
        meta = {"test": "data"}

        # FaissAdapter
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_db = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            faiss_id = faiss_db.store(vec, meta)
            faiss_results = faiss_db.query(vec, k=1)

        # InMemoryDatabase
        mem_db = InMemoryDatabase(dimensions=10000)
        mem_id = mem_db.store(vec, meta)
        mem_results = mem_db.query(vec, k=1)

        # Compare
        assert faiss_id == mem_id == 0
        assert len(faiss_results) == len(mem_results) == 1

        # Similarities should be very close (both normalize, compute cosine)
        faiss_sim = faiss_results[0][1]
        mem_sim = mem_results[0][1]
        assert abs(faiss_sim - mem_sim) < 0.001  # Small numerical error OK

    def test_query_consistency_across_implementations(self):
        """Query results should be consistent between implementations."""
        vecs = [torch.randn(10000) for _ in range(10)]

        # FaissAdapter
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_db = FaissAdapter(dimensions=10000, persist_path=tmpdir)
            for i, vec in enumerate(vecs):
                faiss_db.store(vec, {"idx": i})
            faiss_results = faiss_db.query(vecs[0], k=5)

        # InMemoryDatabase
        mem_db = InMemoryDatabase(dimensions=10000)
        for i, vec in enumerate(vecs):
            mem_db.store(vec, {"idx": i})
        mem_results = mem_db.query(vecs[0], k=5)

        # Compare results
        for (f_id, f_sim, f_meta), (m_id, m_sim, m_meta) in zip(
            faiss_results, mem_results
        ):
            assert f_id == m_id
            assert abs(f_sim - m_sim) < 0.01  # Allow small numerical error
            assert f_meta == m_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

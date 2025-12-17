"""
Unit tests for Codebook and text chunking functionality.

Tests validate:
1. Deterministic encoding
2. Text chunking with overlap
3. Chunk encoding for document storage
"""

import pytest
import torch

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace


@pytest.fixture
def codebook():
    """Create a codebook with default vector space."""
    space = VectorSpace(dimensions=10000)
    return Codebook(space)


class TestCodebookChunking:
    """Test text chunking functionality."""

    def test_chunk_text_basic(self, codebook):
        """Basic text chunking should work."""
        text = "a" * 1000  # 1000 character text
        chunks = codebook.chunk_text(text, chunk_size=500, overlap=100)

        # Should produce 3 chunks with 500 char size and 100 overlap
        # Chunk 1: 0-500, Chunk 2: 400-900, Chunk 3: 800-1000
        assert len(chunks) == 3
        assert len(chunks[0]) == 500
        assert len(chunks[1]) == 500
        assert len(chunks[2]) == 200  # Last chunk is smaller

    def test_chunk_text_empty(self, codebook):
        """Empty text should return empty list."""
        chunks = codebook.chunk_text("", chunk_size=500, overlap=100)
        assert chunks == []

    def test_chunk_text_smaller_than_chunk_size(self, codebook):
        """Text smaller than chunk_size should return single chunk."""
        text = "Hello world"
        chunks = codebook.chunk_text(text, chunk_size=500, overlap=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_chunk_size(self, codebook):
        """Text exactly chunk_size produces 2 chunks due to overlap stepping."""
        text = "a" * 500
        chunks = codebook.chunk_text(text, chunk_size=500, overlap=100)

        # Algorithm steps: [0:500] then [400:500] due to overlap stepping
        assert len(chunks) == 2
        assert len(chunks[0]) == 500  # Full text
        assert len(chunks[1]) == 100  # Overlap portion

    def test_chunk_text_overlap_preserved(self, codebook):
        """Chunks should have proper overlap."""
        text = "0123456789" * 100  # 1000 chars
        chunks = codebook.chunk_text(text, chunk_size=500, overlap=100)

        # Verify overlap: end of chunk N should overlap with start of chunk N+1
        for i in range(len(chunks) - 1):
            chunk_end = chunks[i][-100:]  # Last 100 chars
            next_start = chunks[i + 1][:100]  # First 100 chars
            assert chunk_end == next_start, f"Overlap mismatch at chunk {i}"

    def test_chunk_text_no_overlap(self, codebook):
        """Chunks with zero overlap should be disjoint."""
        text = "0123456789" * 100  # 1000 chars
        chunks = codebook.chunk_text(text, chunk_size=500, overlap=0)

        assert len(chunks) == 2
        assert chunks[0] + chunks[1] == text

    def test_encode_chunks_returns_vectors_and_text(self, codebook):
        """encode_chunks should return parallel lists of vectors and text."""
        text = "This is a test document with some content for chunking."
        vectors, texts = codebook.encode_chunks(text, chunk_size=30, overlap=10)

        # Should have same number of vectors and text chunks
        assert len(vectors) == len(texts)
        assert len(vectors) > 0

        # Each vector should be correct dimensions
        for vec in vectors:
            assert vec.shape == (10000,)

        # Each text should be non-empty
        for t in texts:
            assert len(t) > 0

    def test_encode_chunks_empty_text(self, codebook):
        """encode_chunks on empty text should return empty lists."""
        vectors, texts = codebook.encode_chunks("", chunk_size=500, overlap=100)

        assert vectors == []
        assert texts == []

    def test_encode_chunks_vectors_are_different(self, codebook):
        """Different chunks should produce different vectors."""
        text = "Apple banana cherry date elderberry fig grape honeydew"
        vectors, texts = codebook.encode_chunks(text, chunk_size=15, overlap=5)

        # Ensure we have multiple chunks
        assert len(vectors) >= 2

        # Vectors for different chunks should be different
        for i in range(len(vectors) - 1):
            similarity = torch.cosine_similarity(
                vectors[i], vectors[i + 1], dim=0
            ).item()
            # Different text chunks should have low similarity
            # (unless they happen to overlap significantly)
            assert similarity < 0.99

    def test_chunk_text_preserves_content(self, codebook):
        """Reassembling chunks (with overlap removed) should give original."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunk_size = 100
        overlap = 20
        chunks = codebook.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Reconstruct by taking non-overlapping portions
        reconstructed = chunks[0]
        for i in range(1, len(chunks)):
            # Skip the overlapping part
            reconstructed += chunks[i][overlap:]

        # Should reconstruct the original text
        assert reconstructed == text


class TestCodebookEncoding:
    """Test basic codebook encoding (existing functionality)."""

    def test_deterministic_encoding(self, codebook):
        """Same concept should always produce same vector."""
        v1 = codebook.encode("test")
        v2 = codebook.encode("test")

        assert torch.allclose(v1, v2)

    def test_different_concepts_different_vectors(self, codebook):
        """Different concepts should produce different vectors."""
        v1 = codebook.encode("apple")
        v2 = codebook.encode("orange")

        # Should not be the same
        assert not torch.allclose(v1, v2)

        # Should be approximately orthogonal (low similarity)
        similarity = torch.cosine_similarity(v1, v2, dim=0).item()
        assert abs(similarity) < 0.3

    def test_encode_batch(self, codebook):
        """Batch encoding should work correctly."""
        concepts = ["apple", "banana", "cherry"]
        batch = codebook.encode_batch(concepts)

        assert batch.shape == (3, 10000)

        # Each row should match individual encoding
        for i, concept in enumerate(concepts):
            individual = codebook.encode(concept)
            assert torch.allclose(batch[i], individual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

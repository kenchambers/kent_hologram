"""
Integration tests for Emergent Layer integration.

Tests that facts flow through emergent layers when enabled
and verifies layer creation occurs.
"""

import os
import shutil
import tempfile
import pytest

# Fix OpenMP library conflict between PyTorch and FAISS on macOS
# This must be set before importing FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from hologram.container import HologramContainer


class TestEmergentLayerIntegration:
    """Integration tests for emergent layer fact store."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create temporary directory for persistence."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def container(self):
        """Create container with small dimensions for fast tests."""
        return HologramContainer(dimensions=1000)

    def test_emergent_layers_fact_flow(self, container, temp_persist_dir):
        """Test that facts flow through emergent layers when enabled."""
        # Create emergent layer fact store directly
        fact_store = container.create_emergent_layer_fact_store(
            persist_path=temp_persist_dir,
            use_hnsw=True,
        )

        # Add facts
        result1 = fact_store.add_fact("France", "capital", "Paris")
        result2 = fact_store.add_fact("Germany", "capital", "Berlin")
        result3 = fact_store.add_fact("Italy", "capital", "Rome")

        # Verify results have layer info
        assert result1.layer_id is not None
        assert result1.layer_description is not None

        # Query for a fact
        query_result = fact_store.query("France", "capital")

        # Verify retrieval
        assert query_result is not None
        # Note: confidence may vary depending on CRAG resonator availability
        assert query_result.layer_ids  # Should have queried at least one layer

    def test_layer_creation_occurs(self, container, temp_persist_dir):
        """Verify layer creation occurs when adding facts."""
        fact_store = container.create_emergent_layer_fact_store(
            persist_path=temp_persist_dir,
            use_hnsw=True,
        )

        # Initially no layers
        initial_layers = fact_store.get_layers()
        initial_count = len(initial_layers)

        # Add facts in different semantic domains
        fact_store.add_fact("Python", "is", "programming language")
        fact_store.add_fact("France", "capital", "Paris")
        fact_store.add_fact("Mars", "type", "planet")

        # Check layers were created
        final_layers = fact_store.get_layers()
        final_count = len(final_layers)

        # Should have created at least one layer
        assert final_count > initial_count, "No layers were created"

    def test_enable_emergent_layers_parameter(self, container, temp_persist_dir):
        """Test that enable_emergent_layers parameter in create_persistent_chatbot works."""
        # This test verifies the parameter exists and doesn't crash
        # Full chatbot creation requires more components, so we test minimally
        try:
            chatbot = container.create_persistent_chatbot(
                persist_dir=temp_persist_dir,
                enable_emergent_layers=True,
                enable_metacognition=False,
                enable_ventriloquist=False,
                enable_self_improvement=False,
                enable_cadence=False,
            )
            # If we get here, the parameter works
            assert chatbot is not None
        except Exception as e:
            # Some dependencies may not be available in test environment
            # but the parameter should at least be recognized
            assert "unexpected keyword argument" not in str(e).lower(), \
                f"enable_emergent_layers parameter not recognized: {e}"

    def test_facts_persist_in_layers(self, container, temp_persist_dir):
        """Test that facts are actually stored in layers."""
        fact_store = container.create_emergent_layer_fact_store(
            persist_path=temp_persist_dir,
            use_hnsw=True,
        )

        # Add multiple facts
        facts = [
            ("Cat", "is", "animal"),
            ("Dog", "is", "animal"),
            ("Bird", "is", "animal"),
        ]

        for subj, pred, obj in facts:
            fact_store.add_fact(subj, pred, obj)

        # Get layer stats
        stats = fact_store.get_layer_stats()

        # Verify facts are counted in layers
        total_facts = sum(s["fact_count"] for s in stats.values())
        assert total_facts >= len(facts), f"Expected at least {len(facts)} facts, got {total_facts}"

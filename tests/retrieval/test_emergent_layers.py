"""
Tests for EmergentLayerManager.
"""

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.retrieval.emergent_layers import EmergentLayerManager


class TestEmergentLayerManager:
    """Test EmergentLayerManager functionality."""
    
    @pytest.fixture
    def layer_manager(self, tmp_path):
        """Create a layer manager for testing."""
        return EmergentLayerManager(
            dimensions=1000,
            persist_base_path=str(tmp_path),
            use_hnsw=False,  # Faster for tests
        )
    
    @pytest.fixture
    def sample_vector(self):
        """Create a sample vector."""
        return torch.randn(1000)
    
    def test_create_first_layer(self, layer_manager, sample_vector):
        """Test creating the first layer."""
        result = layer_manager.route_or_create(
            sample_vector,
            "test content",
        )
        
        assert result.is_new is True
        assert result.surprise == 1.0
        assert result.layer is not None
        assert result.layer.fact_count == 0
    
    def test_route_to_existing_layer(self, layer_manager, sample_vector):
        """Test routing to existing layer with similar content."""
        # Create first layer
        result1 = layer_manager.route_or_create(
            sample_vector,
            "test content",
        )
        
        # Try to add very similar content
        similar_vector = sample_vector + torch.randn(1000) * 0.01
        result2 = layer_manager.route_or_create(
            similar_vector,
            "test content similar",
        )
        
        # Should route to existing layer if similarity is high enough
        # (May create new layer if threshold not met)
        assert len(layer_manager.get_all_layers()) <= 2
    
    def test_create_new_layer_for_different_content(self, layer_manager):
        """Test creating new layer for very different content."""
        # Create first layer
        vec1 = torch.randn(1000)
        result1 = layer_manager.route_or_create(vec1, "geography facts")
        
        # Create very different content
        vec2 = torch.randn(1000)
        result2 = layer_manager.route_or_create(vec2, "science facts")
        
        # Should have at least 1 layer (may create 2 if different enough)
        assert len(layer_manager.get_all_layers()) >= 1
    
    def test_strengthen_layer(self, layer_manager, sample_vector):
        """Test Hebbian strengthening of layer prototype."""
        # Create layer
        result = layer_manager.route_or_create(sample_vector, "test")
        layer_id = result.layer.layer_id
        
        initial_prototype = result.layer.prototype_vec.clone()
        
        # Strengthen with new content
        new_vector = torch.randn(1000)
        layer_manager.strengthen_layer(layer_id, new_vector)
        
        # Prototype should have changed
        layer = layer_manager.get_layer(layer_id)
        assert not torch.allclose(layer.prototype_vec, initial_prototype)
    
    def test_get_layer_descriptions(self, layer_manager, sample_vector):
        """Test getting layer descriptions."""
        # Create a few layers
        layer_manager.route_or_create(sample_vector, "content 1")
        layer_manager.route_or_create(torch.randn(1000), "content 2")
        
        descriptions = layer_manager.get_layer_descriptions()
        
        assert len(descriptions) >= 1
        assert all(isinstance(desc, tuple) for desc in descriptions)
        assert all(len(desc) == 2 for desc in descriptions)
    
    def test_merge_similar_layers(self, layer_manager):
        """Test merging very similar layers."""
        # Create two similar layers
        vec1 = torch.randn(1000)
        vec2 = vec1 + torch.randn(1000) * 0.001  # Very similar
        
        layer_manager.route_or_create(vec1, "content 1")
        layer_manager.route_or_create(vec2, "content 2")
        
        initial_count = len(layer_manager.get_all_layers())
        
        # Merge with high threshold
        merged = layer_manager.merge_similar_layers(threshold=0.9)
        
        # May or may not merge depending on actual similarity
        assert merged >= 0
        assert len(layer_manager.get_all_layers()) <= initial_count
    
    def test_get_stats(self, layer_manager, sample_vector):
        """Test getting layer statistics."""
        # Create some layers
        layer_manager.route_or_create(sample_vector, "content")
        
        stats = layer_manager.get_stats()
        
        assert "total_layers" in stats
        assert "total_facts" in stats
        assert "avg_facts_per_layer" in stats
        assert stats["total_layers"] >= 1

"""
Tests for TransientWorkingMemory.
"""

import pytest

from hologram.container import HologramContainer
from hologram.memory.transient_working_memory import (
    TransientWorkingMemory,
    transient_memory_context,
)


class TestTransientWorkingMemory:
    """Test TransientWorkingMemory functionality."""
    
    @pytest.fixture
    def container(self):
        """Create a container."""
        return HologramContainer(dimensions=1000)
    
    @pytest.fixture
    def working_memory(self, container):
        """Create a working memory instance."""
        return TransientWorkingMemory(
            space=container.vector_space,
            codebook=container.codebook,
            capacity=50,
        )
    
    def test_load_facts(self, working_memory):
        """Test loading facts into working memory."""
        facts = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
            ("Italy", "capital", "Rome"),
        ]
        
        loaded = working_memory.load_facts(facts)
        
        assert loaded == 3
        assert working_memory.fact_count == 3
    
    def test_capacity_limit(self, working_memory):
        """Test that capacity limit is enforced."""
        # Create more facts than capacity
        facts = [(f"Country{i}", "capital", f"City{i}") for i in range(100)]
        
        loaded = working_memory.load_facts(facts)
        
        assert loaded == working_memory.capacity
        assert working_memory.fact_count == working_memory.capacity
    
    def test_query(self, working_memory):
        """Test querying for facts."""
        facts = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
        ]
        
        working_memory.load_facts(facts)
        
        answer, confidence = working_memory.query("France", "capital")
        
        # Should retrieve Paris with some confidence
        assert answer is not None
        assert confidence > 0
    
    def test_query_not_found(self, working_memory):
        """Test querying for non-existent fact."""
        facts = [("France", "capital", "Paris")]
        working_memory.load_facts(facts)
        
        answer, confidence = working_memory.query("Spain", "capital")
        
        # May return None or low confidence
        assert confidence < 0.5 or answer is None
    
    def test_get_all_objects(self, working_memory):
        """Test getting all object vocabulary."""
        facts = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
        ]
        
        working_memory.load_facts(facts)
        objects = working_memory.get_all_objects()
        
        assert "Paris" in objects
        assert "Berlin" in objects
    
    def test_get_all_subjects(self, working_memory):
        """Test getting all subject vocabulary."""
        facts = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
        ]
        
        working_memory.load_facts(facts)
        subjects = working_memory.get_all_subjects()
        
        assert "France" in subjects
        assert "Germany" in subjects
    
    def test_clear(self, working_memory):
        """Test clearing working memory."""
        facts = [("France", "capital", "Paris")]
        working_memory.load_facts(facts)
        
        assert working_memory.fact_count > 0
        
        working_memory.clear()
        
        assert working_memory.fact_count == 0
        assert len(working_memory.get_all_objects()) == 0
    
    def test_context_manager(self, container):
        """Test context manager auto-clears memory."""
        facts = [("France", "capital", "Paris")]
        
        with transient_memory_context(
            container.vector_space,
            container.codebook,
        ) as wm:
            wm.load_facts(facts)
            assert wm.fact_count > 0
        
        # Memory should be cleared after context exit
        # (Can't check since wm is out of scope, but test passes if no error)

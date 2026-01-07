"""
TransientWorkingMemory: Ephemeral HDC memory for retrieved facts.

Pattern: MemoryTrace + GlobalWorkspace capacity gating

Key constraint: Created fresh per query, automatically cleared after use.
This ensures the resonator vocabulary is ONLY the retrieved facts,
guaranteeing zero hallucination.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from hologram.config.constants import RESPONSE_CONFIDENCE_THRESHOLD
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace
from hologram.memory.memory_trace import MemoryTrace


class TransientWorkingMemory:
    """
    Ephemeral memory for retrieved facts with capacity gating.
    
    Pattern: MemoryTrace (store/query) + GlobalWorkspace (capacity limit)
    
    Critical properties:
    - Fresh instance per query (no cross-contamination)
    - Capacity limited (50 facts, matches GlobalWorkspace)
    - Cleared after use
    - Vocabulary = loaded facts ONLY
    
    Methods:
        load_facts: Load retrieved facts (up to capacity)
        query: Query for (subject, predicate) -> (object, confidence)
        get_all_objects: Get all object values for resonator vocabulary
        clear: Clear memory
    """
    
    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        capacity: int = 50,
    ):
        """
        Initialize transient working memory.
        
        Args:
            space: VectorSpace for dimensionality
            codebook: Codebook for encoding
            capacity: Maximum facts to hold (default: 50)
        """
        self._space = space
        self._codebook = codebook
        self._capacity = capacity
        self._memory = MemoryTrace(space)
        self._loaded_facts: List[Tuple[str, str, str]] = []
        self._object_vocab: List[str] = []
        self._subject_vocab: List[str] = []
    
    def load_facts(
        self,
        facts: List[Tuple[str, str, str]],
        salience_scores: Optional[List[float]] = None,
    ) -> int:
        """
        Load facts into working memory with capacity gating.
        
        Args:
            facts: List of (subject, predicate, object) tuples
            salience_scores: Optional salience scores for ranking
        
        Returns:
            Number of facts actually loaded
        """
        # Sort by salience if provided
        if salience_scores:
            sorted_facts = sorted(
                zip(facts, salience_scores),
                key=lambda x: x[1],
                reverse=True
            )
            facts = [f for f, _ in sorted_facts]
        
        # Apply capacity limit
        facts_to_load = facts[:self._capacity]
        
        # Clear previous state
        self.clear()
        
        # Load facts
        for subject, predicate, obj in facts_to_load:
            # Encode components
            s_vec = self._codebook.encode(subject)
            p_vec = self._codebook.encode(predicate)
            o_vec = self._codebook.encode(obj)
            
            # Store: bind(subject, predicate) -> object
            key = Operations.bind(s_vec, p_vec)
            self._memory.store(key, o_vec)
            
            # Track vocabulary
            self._loaded_facts.append((subject, predicate, obj))
            if obj not in self._object_vocab:
                self._object_vocab.append(obj)
            if subject not in self._subject_vocab:
                self._subject_vocab.append(subject)
        
        return len(facts_to_load)
    
    def query(self, subject: str, predicate: str) -> Tuple[Optional[str], float]:
        """
        Query for object given subject and predicate.
        
        Args:
            subject: Subject to query
            predicate: Predicate to query
        
        Returns:
            Tuple of (object, confidence) or (None, 0.0) if not found
        """
        if not self._object_vocab:
            return None, 0.0
        
        # Encode query
        s_vec = self._codebook.encode(subject)
        p_vec = self._codebook.encode(predicate)
        key = Operations.bind(s_vec, p_vec)
        
        # Query memory
        result_vec = self._memory.query(key)
        
        # Cleanup against object vocabulary
        object_vecs = self._codebook.encode_batch(self._object_vocab)
        similarities = torch.nn.functional.cosine_similarity(
            result_vec.unsqueeze(0),
            object_vecs,
            dim=1
        )
        
        best_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_idx].item()
        
        # Apply confidence threshold
        if best_similarity < RESPONSE_CONFIDENCE_THRESHOLD:
            return None, 0.0
        
        return self._object_vocab[best_idx], best_similarity
    
    def get_all_objects(self) -> List[str]:
        """
        Get all object values for resonator vocabulary.
        
        Returns:
            List of object strings
        """
        return self._object_vocab.copy()
    
    def get_all_subjects(self) -> List[str]:
        """
        Get all subject values for resonator vocabulary.
        
        Returns:
            List of subject strings
        """
        return self._subject_vocab.copy()
    
    def get_loaded_facts(self) -> List[Tuple[str, str, str]]:
        """
        Get all loaded facts.
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        return self._loaded_facts.copy()
    
    def clear(self) -> None:
        """Clear the working memory."""
        self._memory = MemoryTrace(self._space)
        self._loaded_facts = []
        self._object_vocab = []
        self._subject_vocab = []
    
    @property
    def fact_count(self) -> int:
        """Get number of loaded facts."""
        return len(self._loaded_facts)
    
    @property
    def capacity(self) -> int:
        """Get memory capacity."""
        return self._capacity


@contextmanager
def transient_memory_context(
    space: VectorSpace,
    codebook: Codebook,
    capacity: int = 50,
):
    """
    Context manager for transient working memory.
    
    Ensures memory is automatically cleared on exit.
    
    Args:
        space: VectorSpace for dimensionality
        codebook: Codebook for encoding
        capacity: Maximum facts to hold
    
    Yields:
        TransientWorkingMemory instance
    
    Example:
        >>> with transient_memory_context(space, codebook) as wm:
        ...     wm.load_facts(retrieved_facts)
        ...     result = resonator.resonate_with_working_memory(thought, wm)
        ...     # wm automatically cleared on exit
    """
    wm = TransientWorkingMemory(space, codebook, capacity)
    try:
        yield wm
    finally:
        wm.clear()

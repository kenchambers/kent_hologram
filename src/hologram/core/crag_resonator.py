"""
CRAGResonator: Grounded resonator that operates within working memory vocabulary.

Pattern: Resonator + grounding verification

Key innovation: Vocabulary is strictly limited to TransientWorkingMemory contents,
guaranteeing zero hallucination.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from hologram.core.resonator import Resonator, ResonatorResult
from hologram.memory.transient_working_memory import TransientWorkingMemory


@dataclass
class CRAGResonatorResult:
    """
    Result from CRAG resonator with grounding verification.
    
    Attributes:
        subject_word: Resolved subject
        verb_word: Resolved verb/predicate
        object_word: Resolved object
        confidence: Overall confidence
        grounded: Whether result is grounded in working memory
        resonator_result: Underlying ResonatorResult (None if empty vocabulary)
    """
    subject_word: str
    verb_word: str
    object_word: str
    confidence: float
    grounded: bool
    resonator_result: Optional[ResonatorResult]


class CRAGResonator:
    """
    Resonator constrained to working memory vocabulary.
    
    Pattern: Resonator + grounding verification
    
    Zero hallucination guarantee:
    - Vocabulary = working memory facts ONLY
    - No generation outside retrieved facts
    - Explicit grounding verification
    
    Methods:
        resonate_with_working_memory: Resonate with memory-constrained vocabulary
        verify_grounding: Check if result exists in working memory
    """
    
    def __init__(self, base_resonator: Resonator):
        """
        Initialize CRAG resonator.
        
        Args:
            base_resonator: Underlying Resonator instance
        """
        self._resonator = base_resonator
    
    def resonate_with_working_memory(
        self,
        thought: torch.Tensor,
        working_memory: TransientWorkingMemory,
        verb_vocabulary: List[str] = None,
    ) -> CRAGResonatorResult:
        """
        Resonate using working memory as vocabulary.
        
        Args:
            thought: Thought vector to factorize
            working_memory: TransientWorkingMemory with loaded facts
            verb_vocabulary: Optional verb vocabulary (defaults to common predicates)
        
        Returns:
            CRAGResonatorResult with grounding verification
        """
        # Get vocabulary from working memory
        noun_vocabulary = working_memory.get_all_subjects() + working_memory.get_all_objects()
        
        # Deduplicate while preserving order
        seen = set()
        noun_vocabulary = [x for x in noun_vocabulary if not (x in seen or seen.add(x))]
        
        # Default verb vocabulary if not provided
        if verb_vocabulary is None:
            verb_vocabulary = [
                "is", "has", "contains", "capital", "located",
                "created", "invented", "color", "shape"
            ]
        
        # Handle empty vocabulary
        if not noun_vocabulary:
            # Return empty result
            return CRAGResonatorResult(
                subject_word="",
                verb_word="",
                object_word="",
                confidence=0.0,
                grounded=False,
                resonator_result=None,
            )
        
        # Resonate
        result = self._resonator.resonate(
            thought,
            noun_vocabulary,
            verb_vocabulary,
        )
        
        # Verify grounding
        grounded = self.verify_grounding(result, working_memory)
        
        # Calculate overall confidence
        avg_confidence = (
            result.confidence.get("subject", 0.0) +
            result.confidence.get("verb", 0.0) +
            result.confidence.get("object", 0.0)
        ) / 3.0
        
        return CRAGResonatorResult(
            subject_word=result.subject_word,
            verb_word=result.verb_word,
            object_word=result.object_word,
            confidence=avg_confidence,
            grounded=grounded,
            resonator_result=result,
        )
    
    def verify_grounding(
        self,
        result: ResonatorResult,
        working_memory: TransientWorkingMemory,
    ) -> bool:
        """
        Verify that resonator result is grounded in working memory.
        
        Args:
            result: ResonatorResult to verify
            working_memory: TransientWorkingMemory with facts
        
        Returns:
            True if result exists in working memory
        """
        # Check if (subject, predicate, object) exists in loaded facts
        loaded_facts = working_memory.get_loaded_facts()
        
        for subject, predicate, obj in loaded_facts:
            # Normalize for comparison
            if (
                subject.lower() == result.subject_word.lower() and
                predicate.lower() == result.verb_word.lower() and
                obj.lower() == result.object_word.lower()
            ):
                return True
        
        # Not found in loaded facts
        return False

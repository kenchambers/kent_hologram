"""
CadenceExtractor: Extract sentence structure patterns from text.

Converts "Paris is the capital of France" into:
- Content slots: [ENTITY1, "is the capital of", ENTITY2]
- Structure pattern: "[ENTITY] is the [RELATION] of [ENTITY]"
- Position-encoded template vector
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from hologram.memory.sequence_encoder import SequenceEncoder
from hologram.core.codebook import Codebook


class TransitionType(Enum):
    """Transition types between sentences."""
    CONTINUATION = "continuation"  # also, additionally, moreover
    CONTRAST = "contrast"  # however, but, although
    ELABORATION = "elaboration"  # specifically, for example, namely
    CAUSATION = "causation"  # because, therefore, thus
    NEUTRAL = "neutral"  # no explicit transition


@dataclass
class CadencePattern:
    """A single sentence cadence pattern."""
    template: str  # Template with slot markers (e.g., "__SLOT_ENTITY__ is the capital of __SLOT_ENTITY__")
    structure_vector: torch.Tensor  # Position-encoded structure vector
    slot_positions: List[Tuple[int, str, str]]  # [(position, slot_type, original_text), ...]
    original_text: str  # Original sentence text


@dataclass
class MultiSentenceCadence:
    """Multi-sentence cadence with transitions."""
    patterns: List[CadencePattern]  # One pattern per sentence
    transitions: List[TransitionType]  # Transition types between sentences
    discourse_vector: torch.Tensor  # Encoded discourse structure


class CadenceExtractor:
    """
    Extract cadence patterns (sentence structure) from text.

    Uses POS-like slot detection to separate content from structure.
    """

    def __init__(self, codebook: Codebook):
        """
        Initialize cadence extractor.

        Args:
            codebook: Codebook for encoding
        """
        self._codebook = codebook
        self._seq_encoder = SequenceEncoder(codebook)

        # Structure slot markers
        self._slots = {
            "ENTITY": "__SLOT_ENTITY__",
            "RELATION": "__SLOT_RELATION__",
            "VALUE": "__SLOT_VALUE__",
            "TRANSITION": "__SLOT_TRANSITION__",
        }

    def extract_cadence(self, text: str, entities: List[str]) -> CadencePattern:
        """
        Extract cadence pattern from text.

        Args:
            text: Full response text (e.g., "Paris is the capital of France")
            entities: Known entities in text (e.g., ["Paris", "France"])

        Returns:
            CadencePattern with structure vector and slot positions
        """
        # Replace entities with slot markers
        template = text
        slot_positions = []
        
        # Sort entities by length (longest first) to avoid partial matches
        sorted_entities = sorted(entities, key=len, reverse=True)
        
        for entity in sorted_entities:
            if not entity or len(entity) < 2:
                continue
                
            # Case-insensitive search and replace
            text_lower = template.lower()
            entity_lower = entity.lower()
            
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(entity_lower, start)
                if pos == -1:
                    break
                
                # Check word boundaries to avoid partial matches
                if (pos == 0 or not template[pos-1].isalnum()) and \
                   (pos + len(entity) >= len(template) or not template[pos + len(entity)].isalnum()):
                    slot_positions.append((pos, "ENTITY", entity))
                    # Replace with slot marker
                    template = template[:pos] + self._slots["ENTITY"] + template[pos+len(entity):]
                    text_lower = template.lower()
                    start = pos + len(self._slots["ENTITY"])
                else:
                    start = pos + 1

        # Encode template as position-bound sequence
        tokens = template.split()
        structure_vector = self._seq_encoder.encode(tokens)

        return CadencePattern(
            template=template,
            structure_vector=structure_vector,
            slot_positions=sorted(slot_positions, key=lambda x: x[0]),
            original_text=text,
        )

    def extract_multi_sentence_cadence(
        self, text: str, entities: List[str]
    ) -> MultiSentenceCadence:
        """
        Extract cadence from multi-sentence response.

        Identifies transition patterns between sentences.

        Args:
            text: Multi-sentence text
            entities: Known entities in text

        Returns:
            MultiSentenceCadence with patterns and transitions
        """
        sentences = self._split_sentences(text)

        if len(sentences) == 1:
            pattern = self.extract_cadence(sentences[0], entities)
            return MultiSentenceCadence(
                patterns=[pattern],
                transitions=[],
                discourse_vector=pattern.structure_vector,
            )

        patterns = []
        transitions = []

        for i, sentence in enumerate(sentences):
            pattern = self.extract_cadence(sentence, entities)
            patterns.append(pattern)

            # Detect transition at start of sentence (after first)
            if i > 0:
                transition = self._detect_transition(sentence)
                transitions.append(transition)

        # Encode discourse structure
        discourse_vector = self._encode_discourse(patterns, transitions)

        return MultiSentenceCadence(
            patterns=patterns,
            transitions=transitions,
            discourse_vector=discourse_vector,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Simple sentence splitting on punctuation.
        """
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _detect_transition(self, sentence: str) -> TransitionType:
        """Detect transition type from sentence start."""
        lower = sentence.lower().strip()

        # Learned transitions (not just rule-based)
        CONTINUATION = ["also", "additionally", "moreover", "furthermore", "and"]
        CONTRAST = ["however", "but", "although", "yet", "nevertheless"]
        ELABORATION = ["specifically", "in particular", "for example", "namely"]
        CAUSATION = ["because", "therefore", "thus", "so", "hence"]

        words = lower.split()
        if not words:
            return TransitionType.NEUTRAL

        first_word = words[0]

        if first_word in CONTINUATION:
            return TransitionType.CONTINUATION
        elif first_word in CONTRAST:
            return TransitionType.CONTRAST
        elif first_word in ELABORATION:
            return TransitionType.ELABORATION
        elif first_word in CAUSATION:
            return TransitionType.CAUSATION
        else:
            return TransitionType.NEUTRAL

    def _encode_discourse(
        self,
        patterns: List[CadencePattern],
        transitions: List[TransitionType],
    ) -> torch.Tensor:
        """
        Encode discourse structure as a vector.

        Combines pattern vectors with transition markers.
        """
        from hologram.core.operations import Operations

        # Bundle all pattern structure vectors
        pattern_vecs = [p.structure_vector for p in patterns]
        if not pattern_vecs:
            # Fallback: encode empty discourse
            return self._codebook.encode("__DISCOURSE_EMPTY__")

        discourse_vec = Operations.bundle(*pattern_vecs)

        # Add transition markers
        for transition in transitions:
            transition_vec = self._codebook.encode(f"__TRANSITION_{transition.value}__")
            discourse_vec = Operations.bundle(discourse_vec, transition_vec)

        return discourse_vec


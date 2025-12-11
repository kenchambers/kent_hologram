"""
Response pattern storage and retrieval using HDC.

Stores conversation patterns as holographic vectors and supports
Hebbian learning to strengthen successful patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib

import torch

from hologram.config.constants import (
    LEARNING_STRENGTHEN_FACTOR,
    LEARNING_WEAKEN_FACTOR,
    PATTERN_MATCH_THRESHOLD,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.core.vector_space import VectorSpace
from hologram.conversation.intent import IntentType
from hologram.modulation.sesame import StyleType


@dataclass
class ResponsePattern:
    """A stored response pattern."""

    pattern_id: str
    intent: IntentType
    entity_pattern: List[str]  # Required entities in input
    response_template: str  # Template with {entity}, {answer} slots
    style: StyleType
    strength: float = 1.0  # Learning weight
    vector: Optional[torch.Tensor] = None  # Computed pattern vector

    def __repr__(self) -> str:
        return f"Pattern({self.pattern_id}, {self.intent.value}, strength={self.strength:.2f})"


class ResponsePatternStore:
    """
    Store and retrieve conversation patterns via HDC.

    Patterns are stored as holographic vectors combining intent,
    entity requirements, and response templates. Supports Hebbian
    learning to strengthen successful patterns.

    Attributes:
        _space: VectorSpace configuration
        _codebook: Codebook for encoding
        _patterns: Dictionary of patterns by ID
        _threshold: Minimum similarity for pattern matching

    Example:
        >>> store = ResponsePatternStore(space, codebook)
        >>> matches = store.match(IntentType.GREETING, [], context_vec)
        >>> # Returns matching patterns with scores
    """

    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        threshold: float = PATTERN_MATCH_THRESHOLD,
    ):
        """
        Initialize pattern store.

        Args:
            space: VectorSpace configuration
            codebook: Shared Codebook instance
            threshold: Minimum similarity for pattern matching
        """
        self._space = space
        self._codebook = codebook
        self._threshold = threshold
        self._patterns: Dict[str, ResponsePattern] = {}
        self._init_seed_patterns()

    def _init_seed_patterns(self) -> None:
        """Initialize with basic conversation patterns."""
        seed_patterns = [
            # Greetings
            ResponsePattern(
                pattern_id="greeting_simple",
                intent=IntentType.GREETING,
                entity_pattern=[],
                response_template="Hello! How can I help you today?",
                style=StyleType.NEUTRAL,
            ),
            ResponsePattern(
                pattern_id="greeting_casual",
                intent=IntentType.GREETING,
                entity_pattern=[],
                response_template="Hey! What's on your mind?",
                style=StyleType.CASUAL,
            ),
            ResponsePattern(
                pattern_id="greeting_formal",
                intent=IntentType.GREETING,
                entity_pattern=[],
                response_template="Good day. How may I assist you?",
                style=StyleType.FORMAL,
            ),
            # Questions - fact lookup
            ResponsePattern(
                pattern_id="what_is_query",
                intent=IntentType.QUESTION,
                entity_pattern=["what"],
                response_template="{answer}",
                style=StyleType.NEUTRAL,
            ),
            ResponsePattern(
                pattern_id="capital_query",
                intent=IntentType.QUESTION,
                entity_pattern=["capital"],
                response_template="The capital of {entity} is {answer}.",
                style=StyleType.FORMAL,
            ),
            ResponsePattern(
                pattern_id="who_is_query",
                intent=IntentType.QUESTION,
                entity_pattern=["who"],
                response_template="{entity} is {answer}.",
                style=StyleType.NEUTRAL,
            ),
            ResponsePattern(
                pattern_id="where_is_query",
                intent=IntentType.QUESTION,
                entity_pattern=["where"],
                response_template="{entity} is located in {answer}.",
                style=StyleType.NEUTRAL,
            ),
            # Unknown / clarification
            ResponsePattern(
                pattern_id="unknown_rephrase",
                intent=IntentType.UNKNOWN,
                entity_pattern=[],
                response_template="I'm not sure I understand. Could you rephrase that?",
                style=StyleType.NEUTRAL,
                strength=1.2, # Bump strength
            ),
            ResponsePattern(
                pattern_id="unknown_simple",
                intent=IntentType.UNKNOWN,
                entity_pattern=[],
                response_template="Could you explain that simply?",
                style=StyleType.NEUTRAL,
                strength=1.1,
            ),
            ResponsePattern(
                pattern_id="unknown_learning",
                intent=IntentType.UNKNOWN,
                entity_pattern=[],
                response_template="I am still learning. Can you tell me more about that?",
                style=StyleType.NEUTRAL,
                strength=1.15,
            ),
            ResponsePattern(
                pattern_id="no_information",
                intent=IntentType.QUESTION,
                entity_pattern=[],
                response_template="I don't have information about that yet.",
                style=StyleType.NEUTRAL,
                strength=0.6,  # Lower strength - prefer conversational responses
            ),
            ResponsePattern(
                pattern_id="low_confidence",
                intent=IntentType.QUESTION,
                entity_pattern=[],
                response_template="I'm not confident about that. Could you tell me more?",
                style=StyleType.NEUTRAL,
            ),
            # Farewells
            ResponsePattern(
                pattern_id="farewell_simple",
                intent=IntentType.FAREWELL,
                entity_pattern=[],
                response_template="Goodbye! Have a great day!",
                style=StyleType.NEUTRAL,
            ),
            ResponsePattern(
                pattern_id="thanks_response",
                intent=IntentType.FAREWELL,
                entity_pattern=["thanks", "thank"],
                response_template="You're welcome! Is there anything else?",
                style=StyleType.NEUTRAL,
            ),
            ResponsePattern(
                pattern_id="farewell_casual",
                intent=IntentType.FAREWELL,
                entity_pattern=[],
                response_template="See ya! Take care!",
                style=StyleType.CASUAL,
            ),
            # Statement / Conversational Responses
            # These will strengthen through Hebbian learning during conversations
            # Note: "Got it! I'll remember that." is ONLY returned by teach_fact()
            # when a fact is successfully learned. This pattern is for general statements.
            # IMPORTANT: Varied strengths ensure rotation via repetition penalty
            ResponsePattern(
                pattern_id="statement_noted",
                intent=IntentType.STATEMENT,
                entity_pattern=["is"],
                response_template="Noted. That's interesting information.",
                style=StyleType.NEUTRAL,
                strength=0.8,
            ),
            ResponsePattern(
                pattern_id="statement_acknowledge",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="I see. That's interesting.",
                style=StyleType.NEUTRAL,
                strength=1.0,
            ),
            ResponsePattern(
                pattern_id="statement_curious",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Interesting! Tell me more about that.",
                style=StyleType.NEUTRAL,
                strength=1.1,  # Slightly higher to ensure variety
            ),
            ResponsePattern(
                pattern_id="statement_agree",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="That makes sense. What else?",
                style=StyleType.NEUTRAL,
                strength=1.05,
            ),
            ResponsePattern(
                pattern_id="statement_casual",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Yeah, I hear you.",
                style=StyleType.CASUAL,
                strength=1.0,
            ),
            # Additional statement patterns for variety
            ResponsePattern(
                pattern_id="statement_thoughtful",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="That's a good point. I hadn't thought of it that way.",
                style=StyleType.NEUTRAL,
                strength=1.15,
            ),
            ResponsePattern(
                pattern_id="statement_engaged",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Oh, that's fascinating! Go on.",
                style=StyleType.CASUAL,
                strength=1.08,
            ),
            ResponsePattern(
                pattern_id="statement_reflective",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Hmm, that gives me something to think about.",
                style=StyleType.NEUTRAL,
                strength=1.12,
            ),
            ResponsePattern(
                pattern_id="statement_followup",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="That's cool. What made you think of that?",
                style=StyleType.CASUAL,
                strength=1.07,
            ),
            # Active listening patterns - for when Hologram is being taught
            ResponsePattern(
                pattern_id="listening_continue",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Go on, I'm listening.",
                style=StyleType.NEUTRAL,
                strength=1.06,
            ),
            ResponsePattern(
                pattern_id="listening_encourage",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Tell me more!",
                style=StyleType.CASUAL,
                strength=1.04,
            ),
            ResponsePattern(
                pattern_id="listening_understand",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="I think I understand. What else?",
                style=StyleType.NEUTRAL,
                strength=1.03,
            ),
            ResponsePattern(
                pattern_id="listening_absorb",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="Okay, I'm taking that in.",
                style=StyleType.CASUAL,
                strength=1.02,
            ),
            ResponsePattern(
                pattern_id="listening_process",
                intent=IntentType.STATEMENT,
                entity_pattern=[],
                response_template="That's new to me. Keep going.",
                style=StyleType.NEUTRAL,
                strength=1.01,
            ),
            # Generic conversational question (catches questions that aren't fact lookups)
            ResponsePattern(
                pattern_id="question_conversational",
                intent=IntentType.QUESTION,
                entity_pattern=[],
                response_template="That's a good question. What do you think?",
                style=StyleType.CASUAL,
                strength=1.8,  # Much higher to beat no_information after penalties
            ),
            ResponsePattern(
                pattern_id="question_curious",
                intent=IntentType.QUESTION,
                entity_pattern=[],
                response_template="Hmm, interesting question. I'd have to think about that.",
                style=StyleType.NEUTRAL,
                strength=1.6,
            ),
            ResponsePattern(
                pattern_id="question_reflective",
                intent=IntentType.QUESTION,
                entity_pattern=[],
                response_template="I'm curious about that too. What are your thoughts?",
                style=StyleType.NEUTRAL,
                strength=1.5,
            ),
            # UNKNOWN - conversational fallback (will learn better responses over time)
            ResponsePattern(
                pattern_id="unknown_conversational",
                intent=IntentType.UNKNOWN,
                entity_pattern=[],
                response_template="Interesting point. Go on.",
                style=StyleType.NEUTRAL,
                strength=1.2,  # Prefer conversational over "rephrase"
            ),
        ]

        for pattern in seed_patterns:
            self.add_pattern(pattern)

    def _compute_pattern_vector(self, pattern: ResponsePattern) -> torch.Tensor:
        """Compute holographic vector for a pattern.

        IMPORTANT: Pattern encoding must match query encoding in match() method.
        We only use the PRIMARY entity (first in list) to avoid bundling interference.
        Multiple entity_pattern entries are used for keyword FILTERING, not HDC encoding.
        """
        # Encode intent
        intent_vec = self._codebook.encode(f"__INTENT_{pattern.intent.value}__")

        # Encode entity requirements - use only PRIMARY entity (matches query logic)
        # Don't bundle multiple entities - it creates vector interference
        if pattern.entity_pattern:
            # Use only the first entity for HDC encoding
            # Additional entities in entity_pattern are used for filtering in match()
            primary_entity = pattern.entity_pattern[0]
            entity_vec = self._codebook.encode(primary_entity)
        else:
            entity_vec = self._space.empty_vector()

        # Encode style
        style_vec = self._codebook.encode(f"__STYLE_{pattern.style.value}__")

        # Combine: intent bound with entity, bundled with style
        combined = Operations.bind(intent_vec, entity_vec)
        pattern_vec = Operations.bundle(combined, style_vec)

        return pattern_vec

    def add_pattern(self, pattern: ResponsePattern) -> None:
        """
        Add a new pattern to the store.

        Args:
            pattern: The response pattern to add
        """
        # Compute vector if not provided
        if pattern.vector is None:
            pattern.vector = self._compute_pattern_vector(pattern)

        self._patterns[pattern.pattern_id] = pattern

    def match(
        self,
        intent: IntentType,
        entities: List[str],
        context_vec: Optional[torch.Tensor] = None,
        style: Optional[StyleType] = None,
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Find matching patterns via resonance.

        Args:
            intent: The detected intent
            entities: Extracted entity canonical forms
            context_vec: Optional conversation context
            style: Optional preferred style

        Returns:
            List of (pattern, score) tuples, sorted by score descending
        """
        # Build query vector
        intent_vec = self._codebook.encode(f"__INTENT_{intent.value}__")

        # ARCHITECTURAL FIX: Don't bundle entities - it creates vector interference
        # Instead, use only the PRIMARY entity (first content word) for HDC matching
        # and rely on entity_pattern matching (lines 451-459) for filtering
        if entities:
            # Filter out question words that pollute the query
            question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
            content_entities = [e for e in entities if e.lower() not in question_words]

            if content_entities:
                # Use only the first content entity (usually the subject/topic)
                # This preserves semantic clarity without interference
                primary_entity = content_entities[0]
                entity_vec = self._codebook.encode(primary_entity)
            else:
                entity_vec = self._space.empty_vector()
        else:
            entity_vec = self._space.empty_vector()

        query_vec = Operations.bind(intent_vec, entity_vec)

        # Add context influence if provided
        if context_vec is not None:
            query_vec = Operations.bundle(query_vec, context_vec)

        # Match against all patterns
        matches: List[Tuple[ResponsePattern, float]] = []

        for pattern in self._patterns.values():
            if pattern.vector is None:
                continue

            # Compute base similarity
            sim = float(Similarity.cosine(query_vec, pattern.vector))

            # Boost by pattern strength (learned weight)
            sim *= pattern.strength

            # Boost if intent matches exactly
            if pattern.intent == intent:
                sim *= 1.2

            # Boost if style matches preference
            if style and pattern.style == style:
                sim *= 1.1

            # Check entity pattern requirements
            if pattern.entity_pattern:
                entity_match = any(
                    e.lower() in [x.lower() for x in entities]
                    for e in pattern.entity_pattern
                )
                if entity_match:
                    sim *= 1.3
                else:
                    sim *= 0.5  # Penalty for missing required entities

            if sim >= self._threshold:
                matches.append((pattern, sim))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def strengthen_pattern(
        self, pattern_id: str, factor: float = LEARNING_STRENGTHEN_FACTOR
    ) -> None:
        """
        Strengthen a successful pattern (Hebbian learning).

        Args:
            pattern_id: ID of pattern to strengthen
            factor: Multiplication factor for strength
        """
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.strength = min(pattern.strength * factor, 5.0)  # Cap at 5x

    def weaken_pattern(
        self, pattern_id: str, factor: float = LEARNING_WEAKEN_FACTOR
    ) -> None:
        """
        Weaken an unsuccessful pattern.

        Args:
            pattern_id: ID of pattern to weaken
            factor: Multiplication factor for strength
        """
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.strength = max(pattern.strength * factor, 0.2)  # Floor at 0.2x

    def get_pattern(self, pattern_id: str) -> Optional[ResponsePattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_patterns_for_intent(self, intent: IntentType) -> List[ResponsePattern]:
        """Get all patterns for a specific intent."""
        return [p for p in self._patterns.values() if p.intent == intent]

    def create_pattern_from_interaction(
        self,
        intent: IntentType,
        entities: List[str],
        response: str,
        style: StyleType = StyleType.NEUTRAL,
    ) -> ResponsePattern:
        """
        Create a new pattern from a successful interaction.

        Args:
            intent: The intent of the interaction
            entities: Entities involved
            response: The successful response
            style: Style of the response

        Returns:
            Newly created ResponsePattern
        """
        # Generate unique ID
        content = f"{intent.value}_{','.join(entities)}_{response[:20]}"
        pattern_id = f"learned_{hashlib.md5(content.encode()).hexdigest()[:8]}"

        pattern = ResponsePattern(
            pattern_id=pattern_id,
            intent=intent,
            entity_pattern=entities,
            response_template=response,
            style=style,
            strength=0.8,  # Start slightly weaker than seed patterns
        )

        self.add_pattern(pattern)
        return pattern

    @property
    def pattern_count(self) -> int:
        """Number of patterns in store."""
        return len(self._patterns)

    def __repr__(self) -> str:
        return f"ResponsePatternStore(patterns={self.pattern_count})"

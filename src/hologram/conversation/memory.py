"""
Conversation memory using holographic traces.

Maintains session-level context through HDC operations,
enabling context-aware follow-up understanding.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import torch

from hologram.config.constants import CONTEXT_LOOKBACK, MAX_CONVERSATION_TURNS
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.core.vector_space import VectorSpace
from hologram.conversation.intent import IntentType
from hologram.conversation.entity import Entity
from hologram.memory.memory_trace import MemoryTrace


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    user_input: str
    user_vector: torch.Tensor
    intent: IntentType
    entities: List[Entity]
    response: str
    response_vector: torch.Tensor
    timestamp: datetime = field(default_factory=datetime.now)
    pattern_id: Optional[str] = None  # Which pattern was used

    def __repr__(self) -> str:
        return f"Turn(user='{self.user_input[:20]}...', intent={self.intent.value})"


class ConversationMemory:
    """
    Session-level conversation context using holographic traces.

    Stores conversation turns and provides context vectors for
    follow-up understanding and response selection.

    Attributes:
        _space: VectorSpace for dimension configuration
        _codebook: Codebook for encoding
        _turns: List of conversation turns
        _context_trace: Holographic trace of all turns
        _max_turns: Maximum turns to retain

    Example:
        >>> memory = ConversationMemory(space, codebook)
        >>> memory.add_turn(turn)
        >>> context = memory.get_context_vector()
        >>> # Use context for follow-up question handling
    """

    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        max_turns: int = MAX_CONVERSATION_TURNS,
    ):
        """
        Initialize conversation memory.

        Args:
            space: VectorSpace configuration
            codebook: Shared Codebook instance
            max_turns: Maximum turns to keep in memory
        """
        self._space = space
        self._codebook = codebook
        self._max_turns = max_turns
        self._turns: List[ConversationTurn] = []
        self._context_trace = MemoryTrace(space)

    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Record a conversation turn.

        Args:
            turn: The conversation turn to record
        """
        # Add to list
        self._turns.append(turn)

        # Create turn vector: bind(user, response) with position
        position_vec = self._codebook.get_positional(len(self._turns) - 1)
        turn_vec = Operations.bind(turn.user_vector, turn.response_vector)
        positioned_turn = Operations.bind(turn_vec, position_vec)

        # Store in holographic trace
        # Key: position, Value: turn content
        self._context_trace.store(position_vec, turn_vec)

        # Trim if exceeds max
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_context_vector(self, lookback: int = CONTEXT_LOOKBACK) -> torch.Tensor:
        """
        Get holographic context from recent turns.

        Args:
            lookback: Number of recent turns to include

        Returns:
            Bundled vector representing recent context
        """
        if not self._turns:
            # Return a deterministic "null context" vector instead of zeros
            # This avoids "norm is zero" warnings in torchhd operations
            # Use seed 0 for "start of conversation" state
            return self._space.random_vector(0)

        # Get recent turns
        recent_turns = self._turns[-lookback:]

        # Bundle their vectors with recency weighting
        # More recent = stronger signal (bundled multiple times)
        context_vecs = []
        for i, turn in enumerate(recent_turns):
            # Weight by recency: more recent = added more times
            weight = i + 1  # 1, 2, 3 for last 3 turns
            turn_vec = Operations.bind(turn.user_vector, turn.response_vector)
            for _ in range(weight):
                context_vecs.append(turn_vec)

        if not context_vecs:
            # Return a deterministic "null context" vector instead of zeros
            return self._space.random_vector(0)

        result = context_vecs[0]
        for vec in context_vecs[1:]:
            result = Operations.bundle(result, vec)

        return result

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent turn."""
        return self._turns[-1] if self._turns else None

    def get_last_entities(self) -> List[Entity]:
        """Get entities from the last turn."""
        last = self.get_last_turn()
        return last.entities if last else []

    def get_last_intent(self) -> Optional[IntentType]:
        """Get intent from the last turn."""
        last = self.get_last_turn()
        return last.intent if last else None

    def query_context(self, query_vec: torch.Tensor, top_k: int = 3) -> List[ConversationTurn]:
        """
        Find relevant past turns via resonance.

        Args:
            query_vec: Query vector to match against
            top_k: Number of turns to return

        Returns:
            List of most relevant past turns
        """
        if not self._turns:
            return []

        # Compute similarity to each turn
        similarities = []
        for turn in self._turns:
            turn_vec = Operations.bind(turn.user_vector, turn.response_vector)
            sim = Similarity.cosine(query_vec, turn_vec)
            similarities.append((turn, float(sim)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [turn for turn, _ in similarities[:top_k]]

    def get_recent_patterns(self, lookback: int = 5) -> List[str]:
        """
        Get IDs of recently used patterns.
        
        Args:
            lookback: Number of recent turns to check
            
        Returns:
            List of pattern IDs (most recent first)
        """
        recent_turns = self._turns[-lookback:]
        # Reverse to have most recent first
        return [t.pattern_id for t in reversed(recent_turns) if t.pattern_id]

    def get_recent_messages(self, lookback: int = 5) -> List[str]:
        """
        Get text of recently used responses AND user inputs.
        
        Args:
            lookback: Number of recent turns to check
            
        Returns:
            List of message strings (most recent first)
        """
        recent_turns = self._turns[-lookback:]
        messages = []
        for t in reversed(recent_turns):
            if t.response:
                messages.append(t.response)
            if t.user_input:
                messages.append(t.user_input)
        return messages

    def get_topic_entities(self) -> List[str]:
        """
        Get entities that have appeared multiple times (topic indicators).

        Returns:
            List of entity canonical forms that appear in multiple turns
        """
        entity_counts: dict = {}
        for turn in self._turns:
            for entity in turn.entities:
                key = entity.canonical_form
                entity_counts[key] = entity_counts.get(key, 0) + 1

        # Return entities appearing more than once
        return [e for e, count in entity_counts.items() if count > 1]

    def clear(self) -> None:
        """Clear session memory."""
        self._turns = []
        self._context_trace = MemoryTrace(self._space)

    @property
    def turn_count(self) -> int:
        """Number of turns in memory."""
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={self.turn_count}, max={self._max_turns})"

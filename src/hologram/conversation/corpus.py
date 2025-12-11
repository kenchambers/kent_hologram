"""
Response Corpus: Store and retrieve learned conversational responses.

Stores successful conversational responses from training with their context
vectors, enabling HDC-based retrieval for natural conversation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.conversation.intent import IntentType
from hologram.modulation.sesame import StyleType


@dataclass
class CorpusEntry:
    """A stored response entry in the corpus."""

    response: str
    context_vector: torch.Tensor
    intent: IntentType
    style: StyleType
    source: str  # "claude", "gemini", "learned"
    usage_count: int = 0  # Track how often this response is used


class ResponseCorpus:
    """
    Store and retrieve conversational responses via HDC.

    Stores responses with their context vectors, allowing retrieval
    of contextually appropriate responses through holographic resonance.

    Attributes:
        _codebook: Codebook for encoding
        _entries: List of stored corpus entries
        _threshold: Minimum similarity for retrieval

    Example:
        >>> corpus = ResponseCorpus(codebook)
        >>> corpus.add_response(context_vec, "That's interesting!", IntentType.STATEMENT)
        >>> matches = corpus.retrieve(query_vec, IntentType.STATEMENT)
    """

    def __init__(
        self,
        codebook: Codebook,
        threshold: float = 0.3,
    ):
        """
        Initialize response corpus.

        Args:
            codebook: Codebook for encoding
            threshold: Minimum similarity for retrieval
        """
        self._codebook = codebook
        self._threshold = threshold
        self._entries: List[CorpusEntry] = []

    def add_response(
        self,
        context_vector: torch.Tensor,
        response: str,
        intent: IntentType,
        style: StyleType = StyleType.NEUTRAL,
        source: str = "learned",
    ) -> None:
        """
        Add a response to the corpus.

        Args:
            context_vector: HDC vector representing the context
            response: The response text to store
            intent: Intent type for this response
            style: Style type
            source: Source identifier ("claude", "gemini", "learned")
        """
        entry = CorpusEntry(
            response=response,
            context_vector=context_vector,
            intent=intent,
            style=style,
            source=source,
        )
        self._entries.append(entry)

    def retrieve(
        self,
        query_vector: torch.Tensor,
        intent: Optional[IntentType] = None,
        style: Optional[StyleType] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar responses via HDC resonance.

        Args:
            query_vector: Query vector to match against
            intent: Optional intent filter
            style: Optional style filter
            top_k: Number of results to return

        Returns:
            List of (response, similarity) tuples, sorted by similarity descending
        """
        if not self._entries:
            return []

        matches: List[Tuple[str, float]] = []

        for entry in self._entries:
            # Filter by intent if specified
            if intent and entry.intent != intent:
                continue

            # Filter by style if specified
            if style and entry.style != style:
                continue

            # Compute similarity
            similarity = float(Similarity.cosine(query_vector, entry.context_vector))

            # Boost by usage count (popular responses are slightly preferred)
            # But don't let this dominate - similarity is primary
            usage_boost = 1.0 + (entry.usage_count * 0.01)
            similarity *= usage_boost

            if similarity >= self._threshold:
                matches.append((entry.response, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)

        # Track usage for top match
        if matches:
            best_response = matches[0][0]
            for entry in self._entries:
                if entry.response == best_response:
                    entry.usage_count += 1
                    break

        return matches[:top_k]

    def get_entry_count(self) -> int:
        """Get number of entries in corpus."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []

    def get_stats(self) -> dict:
        """Get corpus statistics."""
        by_source = {}
        by_intent = {}
        by_style = {}

        for entry in self._entries:
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
            by_intent[entry.intent.value] = by_intent.get(entry.intent.value, 0) + 1
            by_style[entry.style.value] = by_style.get(entry.style.value, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_source": by_source,
            "by_intent": by_intent,
            "by_style": by_style,
        }

    def __repr__(self) -> str:
        return f"ResponseCorpus(entries={len(self._entries)}, threshold={self._threshold})"

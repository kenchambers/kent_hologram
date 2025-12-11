"""
Entity extraction via vocabulary resonance.

Identifies known concepts in user input by comparing against
the vocabulary stored in the Codebook and FactStore.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

import torch

from hologram.config.constants import ENTITY_MATCH_THRESHOLD
from hologram.core.codebook import Codebook
from hologram.core.similarity import Similarity
from hologram.memory.fact_store import FactStore


@dataclass
class Entity:
    """A recognized entity from user input."""

    surface_form: str  # Word as it appeared in input
    canonical_form: str  # Matched vocabulary word
    similarity: float  # How close the match was
    entity_type: str  # "known_concept" | "fact_subject" | "fact_object"

    def __repr__(self) -> str:
        return f"Entity({self.surface_form}->{self.canonical_form}, {self.similarity:.2f})"


class EntityExtractor:
    """
    Extract known entities from user input via vocabulary resonance.

    Compares each token against:
    1. Fact store subjects and objects
    2. Codebook cached concepts
    3. Custom learned vocabulary

    Attributes:
        _codebook: Shared Codebook for encoding
        _fact_store: FactStore for fact-based vocabulary
        _custom_vocabulary: Entities learned from conversation
        _threshold: Minimum similarity for entity match

    Example:
        >>> extractor = EntityExtractor(codebook, fact_store)
        >>> entities = extractor.extract("What is the capital of France?")
        >>> # [Entity(capital->capital, 1.0), Entity(France->France, 1.0)]
    """

    def __init__(
        self,
        codebook: Codebook,
        fact_store: Optional[FactStore] = None,
        threshold: float = ENTITY_MATCH_THRESHOLD,
    ):
        """
        Initialize entity extractor.

        Args:
            codebook: Shared Codebook instance
            fact_store: Optional FactStore for vocabulary
            threshold: Minimum similarity for entity matching
        """
        self._codebook = codebook
        self._fact_store = fact_store
        self._threshold = threshold
        self._custom_vocabulary: Set[str] = set()
        self._stopwords = self._init_stopwords()

    def _init_stopwords(self) -> Set[str]:
        """Initialize common stopwords to ignore."""
        return {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "you",
            "your",
            "yours",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "it",
            "its",
            "they",
            "them",
            "their",
            # "what", "which", "who", "whom", "this", "that", "these", "those",
            # "where", "when", "why", "how" - removed to allow entity matching
            "which",
            "whom",
            "this",
            "that",
            "these",
            "those",
        }

    def extract(self, text: str) -> List[Entity]:
        """
        Find entities in text that match known vocabulary.

        Args:
            text: User input text

        Returns:
            List of recognized entities with similarity scores
        """
        tokens = self._tokenize(text)
        entities: List[Entity] = []

        # Build vocabulary to check against
        vocabulary = self._build_vocabulary()

        for token in tokens:
            # Skip stopwords
            if token in self._stopwords:
                continue

            # Skip very short tokens
            if len(token) < 2:
                continue

            # Check for exact match first (fast path)
            if token in vocabulary:
                entities.append(
                    Entity(
                        surface_form=token,
                        canonical_form=token,
                        similarity=1.0,
                        entity_type=self._get_entity_type(token),
                    )
                )
                continue

            # Check for similar matches via HDC
            token_vec = self._codebook.encode(token)
            best_match: Optional[Entity] = None
            best_sim = self._threshold

            for vocab_word in vocabulary:
                vocab_vec = self._codebook.encode(vocab_word)
                sim = Similarity.cosine(token_vec, vocab_vec)

                if sim > best_sim:
                    best_sim = float(sim)
                    best_match = Entity(
                        surface_form=token,
                        canonical_form=vocab_word,
                        similarity=best_sim,
                        entity_type=self._get_entity_type(vocab_word),
                    )

            if best_match:
                entities.append(best_match)

        return entities

    def _build_vocabulary(self) -> Set[str]:
        """Build vocabulary set from all sources."""
        vocab: Set[str] = set()

        # Add from fact store
        if self._fact_store:
            # Get subjects and objects from stored facts
            facts = self._fact_store.get_all_facts() if hasattr(self._fact_store, 'get_all_facts') else []
            for fact in facts:
                # Handle both Fact objects and dict-like objects
                subject = getattr(fact, 'subject', '') or ''
                obj = getattr(fact, 'object', '') or ''
                predicate = getattr(fact, 'predicate', '') or ''
                vocab.add(subject.lower())
                vocab.add(obj.lower())
                vocab.add(predicate.lower())

        # Add from codebook cache
        vocab.update(self._codebook._cache.keys())

        # Add custom vocabulary
        vocab.update(self._custom_vocabulary)

        return vocab

    def _get_entity_type(self, word: str) -> str:
        """Determine the type of entity."""
        if self._fact_store:
            facts = self._fact_store.get_all_facts() if hasattr(self._fact_store, 'get_all_facts') else []
            for fact in facts:
                subject = getattr(fact, 'subject', '').lower()
                obj = getattr(fact, 'object', '').lower()
                if subject == word:
                    return "fact_subject"
                if obj == word:
                    return "fact_object"
        if word in self._custom_vocabulary:
            return "learned"
        return "known_concept"

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split."""
        import re

        text = re.sub(r"[^\w\s']", " ", text.lower())
        return [t.strip("'") for t in text.split() if t]

    def add_entity(self, entity: str) -> None:
        """
        Add new entity to vocabulary (learned from conversation).

        Args:
            entity: New entity word to add
        """
        entity_lower = entity.lower().strip()
        if entity_lower and entity_lower not in self._stopwords:
            self._custom_vocabulary.add(entity_lower)
            # Pre-cache in codebook
            self._codebook.encode(entity_lower)

    def get_vocabulary_size(self) -> int:
        """Return total vocabulary size."""
        return len(self._build_vocabulary())

    def __repr__(self) -> str:
        return f"EntityExtractor(vocab={self.get_vocabulary_size()}, threshold={self._threshold})"

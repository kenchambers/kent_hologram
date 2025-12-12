"""
FactStore: Subject-Predicate-Object triple management.

Provides structured fact storage using S-P-O triples on top of the
holographic memory trace. Facts are encoded as nested bindings:
bind(bind(subject, predicate), object)

Supports citation tracking for bounded hallucination.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from hologram.config.constants import SURPRISE_THRESHOLD
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace
from hologram.memory.memory_trace import MemoryTrace


@dataclass
class Fact:
    """
    A subject-predicate-object fact with metadata.

    Attributes:
        subject: The subject of the fact (e.g., "France")
        predicate: The relationship/property (e.g., "capital")
        object: The object/value (e.g., "Paris")
        confidence: Source confidence [0, 1]
        source: Optional citation/source reference
        timestamp: When the fact was stored
        surprise_score: Surprise score when fact was learned (0.0 = known, 1.0 = novel)
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    surprise_score: Optional[float] = None

    def __str__(self) -> str:
        citation = f" [{self.source}]" if self.source else ""
        return f"{self.subject} --{self.predicate}--> {self.object}{citation}"


class FactStore:
    """
    Manages subject-predicate-object facts using holographic storage.

    Encodes facts as nested bindings:
    key = bind(subject_vector, predicate_vector)
    store(key, object_vector)

    Supports bidirectional queries:
    - Forward: "What is the {predicate} of {subject}?" (query method)
    - Reverse: "What has {predicate} of {object}?" (query_subject method)

    Maintains metadata for citation enforcement and tracks the
    vocabulary of all stored subjects and objects for cleanup/retrieval.

    Attributes:
        _memory: Underlying MemoryTrace for holographic storage
        _codebook: Codebook for converting strings to vectors
        _facts: List of stored facts (metadata)
        _value_vocab: Set of all object strings for retrieval
        _subject_vocab: Set of all subject strings for reverse queries

    Example:
        >>> space = VectorSpace(dimensions=10000)
        >>> codebook = Codebook(space)
        >>> fs = FactStore(space, codebook)
        >>> fs.add_fact("France", "capital", "Paris")
        >>> answer, confidence = fs.query("France", "capital")
        >>> answer
        'Paris'
        >>> subject, conf = fs.query_subject("capital", "Paris")
        >>> subject
        'France'
    """

    def __init__(self, space: VectorSpace, codebook: Codebook):
        """
        Initialize fact store.

        Args:
            space: VectorSpace for dimensionality
            codebook: Codebook for string->vector conversion
        """
        self._memory = MemoryTrace(space)
        self._codebook = codebook
        self._facts: list[Fact] = []
        self._value_vocab: set[str] = set()
        self._subject_vocab: set[str] = set()  # Track unique subjects
        # Exact match index: normalized_key -> Fact
        self._exact_index: dict[str, Fact] = {}
        # Cached encoded value vectors (for performance)
        self._value_vectors_cache: dict[str, torch.Tensor] = {}
        # Cached encoded subject vectors (for reverse queries)
        self._subject_vectors_cache: dict[str, torch.Tensor] = {}
        # Optional FAISS prefilter for large vocabularies (lazily created)
        self._value_faiss = None
        self._value_faiss_threshold = 500  # Use FAISS prefilter when vocab exceeds this

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
        confidence: float = 1.0
    ) -> Optional[Fact]:
        """
        Add a fact to the store.

        Encodes as: bind(bind(subject, predicate), object)
        and stores in memory trace.

        Args:
            subject: Subject string
            predicate: Predicate/relationship string
            obj: Object/value string
            source: Optional citation source
            confidence: Confidence in this fact [0, 1]

        Returns:
            The created Fact object if new, None if duplicate

        Example:
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fact = fs.add_fact("Earth", "shape", "round", source="Science")
            >>> fact.subject
            'Earth'
        """
        # Normalize for indexing (but preserve original in metadata)
        subject_norm = self._normalize(subject)
        predicate_norm = self._normalize(predicate)
        
        # Check for duplicate: same subject+predicate combination
        exact_key = f"{subject_norm}:{predicate_norm}"
        if exact_key in self._exact_index:
            # Fact already exists - return None to indicate duplicate
            existing_fact = self._exact_index[exact_key]
            # Update object if different (allows fact correction)
            if self._normalize(existing_fact.object) != self._normalize(obj):
                # Object changed - update the fact
                existing_fact.object = obj
                # Re-encode and store updated value with surprise gating
                o_vec = self._codebook.encode(obj)
                s_vec = self._codebook.encode(subject_norm)
                p_vec = self._codebook.encode(predicate_norm)
                key = Operations.bind(s_vec, p_vec)
                surprise = self._memory.store_with_surprise(
                    key,
                    o_vec,
                    learning_rate=confidence,
                    surprise_threshold=SURPRISE_THRESHOLD
                )
                existing_fact.surprise_score = surprise
                self._value_vocab.add(obj)
                self._value_vectors_cache[obj] = o_vec
            return None  # Duplicate subject+predicate, no new fact learned
        
        # Encode components as vectors (use normalized forms for better matching)
        s_vec = self._codebook.encode(subject_norm)
        p_vec = self._codebook.encode(predicate_norm)
        o_vec = self._codebook.encode(obj)

        # Create key: bind(subject, predicate)
        key = Operations.bind(s_vec, p_vec)

        # NEW: Store with surprise gating (Titans-inspired)
        surprise = self._memory.store_with_surprise(
            key,
            o_vec,
            learning_rate=confidence,  # Use fact confidence as learning rate
            surprise_threshold=SURPRISE_THRESHOLD
        )

        # Track metadata (preserve original text)
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source,
            surprise_score=surprise  # Track surprise for analysis
        )

        # Only count as new fact if surprise was above threshold
        if surprise >= SURPRISE_THRESHOLD:
            self._facts.append(fact)
            self._value_vocab.add(obj)
            self._subject_vocab.add(subject)  # Track subject for reverse queries
            self._exact_index[exact_key] = fact
            self._value_vectors_cache[obj] = o_vec
            self._subject_vectors_cache[subject] = s_vec  # Cache subject vector
            if self._value_faiss:
                try:
                    self._value_faiss.store(o_vec, {"value": obj})
                except Exception:
                    pass
            return fact

        # Duplicate/low surprise - no new learning occurred
        return None

    def _normalize(self, text: str) -> str:
        """
        Normalize text for consistent matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized text (lowercase, stripped)
        """
        return text.lower().strip()

    def query(self, subject: str, predicate: str) -> tuple[str, float]:
        """
        Query: 'What is the {predicate} of {subject}?'

        Multi-strategy retrieval:
        1. Exact Match: Check normalized key directly (O(1))
        2. Resonance Search: If exact match fails, use HDC resonance with normalized vectors

        Args:
            subject: Subject to query
            predicate: Predicate to query

        Returns:
            Tuple of (answer, confidence) where:
            - answer: The object string (or best match)
            - confidence: Cosine similarity to best match [0, 1]

        Example:
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fs.add_fact("France", "capital", "Paris")
            >>> answer, conf = fs.query("France", "capital")
            >>> answer
            'Paris'
            >>> conf > 0.9
            True
        """
        if len(self._value_vocab) == 0:
            # No facts stored yet
            return "", 0.0

        # Strategy 1: Exact normalized match (fastest)
        subject_norm = self._normalize(subject)
        predicate_norm = self._normalize(predicate)
        exact_key = f"{subject_norm}:{predicate_norm}"
        
        if exact_key in self._exact_index:
            fact = self._exact_index[exact_key]
            # Perfect match - return with high confidence
            return fact.object, 1.0

        # Strategy 2: HDC resonance search (fuzzy matching)
        # Use normalized forms for encoding to improve matching
        s_vec = self._codebook.encode(subject_norm)
        p_vec = self._codebook.encode(predicate_norm)
        key = Operations.bind(s_vec, p_vec)

        value_list = sorted(self._value_vocab)  # Deterministic ordering

        # Optional FAISS prefilter for large vocabularies
        candidate_values = value_list
        if len(value_list) > self._value_faiss_threshold:
            candidate_values = self._prefilter_values_with_faiss(value_list, key)

        # Build candidates from cache or encode on-the-fly
        candidates_list = []
        for v in candidate_values:
            if v in self._value_vectors_cache:
                candidates_list.append(self._value_vectors_cache[v])
            else:
                vec = self._codebook.encode(v)
                self._value_vectors_cache[v] = vec
                candidates_list.append(vec)
        
        candidates = torch.stack(candidates_list)

        # Find best match via resonance
        similarities = self._memory.resonance(key, candidates)
        best_idx = torch.argmax(similarities).item()
        confidence = float(similarities[best_idx].item())

        return candidate_values[best_idx], confidence

    def query_subject(self, predicate: str, obj: str) -> tuple[str, float]:
        """
        Reverse query: 'What has {predicate} of {object}?'
        
        This enables finding subjects by their properties, essential for
        code genealogy (e.g., finding callers of a function).
        
        Multi-strategy retrieval:
        1. Exact Match: Check all facts for predicate+object match
        2. Resonance Search: Use HDC to find best subject match
        
        Args:
            predicate: Predicate to query
            obj: Object value to search for
            
        Returns:
            Tuple of (subject, confidence) where:
            - subject: The subject string (or best match)
            - confidence: Cosine similarity to best match [0, 1]
            
        Example:
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fs.add_fact("Paris", "country", "France")
            >>> subject, conf = fs.query_subject("country", "France")
            >>> subject
            'Paris'
        """
        if len(self._subject_vocab) == 0:
            # No facts stored yet
            return "", 0.0
        
        # Strategy 1: Exact match via metadata (fastest)
        predicate_norm = self._normalize(predicate)
        obj_norm = self._normalize(obj)
        
        # Search through exact index for matching predicate+object
        for key, fact in self._exact_index.items():
            if (self._normalize(fact.predicate) == predicate_norm and 
                self._normalize(fact.object) == obj_norm):
                return fact.subject, 1.0
        
        # Strategy 2: HDC resonance search (fuzzy matching)
        # This is more complex for reverse queries, but follows the principle:
        # If: trace = bind(bind(subject, predicate), object)
        # Then: bind(trace, object) ≈ bind(subject, predicate)
        # And: bind(bind(trace, object), predicate) ≈ subject
        
        p_vec = self._codebook.encode(predicate_norm)
        o_vec = self._codebook.encode(obj_norm)
        
        # Get all candidate subject vectors
        subject_list = sorted(self._subject_vocab)
        
        # Build candidates from cache or encode on-the-fly
        candidates_list = []
        for s in subject_list:
            if s in self._subject_vectors_cache:
                candidates_list.append(self._subject_vectors_cache[s])
            else:
                vec = self._codebook.encode(self._normalize(s))
                self._subject_vectors_cache[s] = vec
                candidates_list.append(vec)
        
        candidates = torch.stack(candidates_list)
        
        # For reverse query, we need to check which subject best matches
        # Create query key: bind(predicate, object)
        query_key = Operations.bind(p_vec, o_vec)
        
        # Find best match via resonance
        similarities = self._memory.resonance(query_key, candidates)
        best_idx = torch.argmax(similarities).item()
        confidence = float(similarities[best_idx].item())
        
        return subject_list[best_idx], confidence

    def get_facts_by_subject(self, subject: str) -> list[Fact]:
        """
        Get all facts about a subject (from metadata).

        This is O(n) metadata lookup, not a holographic query.
        Used for citation and debugging.

        Args:
            subject: Subject to search for

        Returns:
            List of all facts with this subject

        Example:
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fs.add_fact("Paris", "country", "France")
            >>> fs.add_fact("Paris", "population", "2.2M")
            >>> facts = fs.get_facts_by_subject("Paris")
            >>> len(facts)
            2
        """
        return [f for f in self._facts if f.subject == subject]
    
    def get_facts_by_object(self, obj: str) -> list[Fact]:
        """
        Get all facts with a given object value (from metadata).
        
        This is O(n) metadata lookup for reverse searches.
        Useful for finding all entities with a particular property value.
        
        Args:
            obj: Object value to search for
            
        Returns:
            List of all facts with this object
            
        Example:
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fs.add_fact("Paris", "country", "France")
            >>> fs.add_fact("Lyon", "country", "France")
            >>> facts = fs.get_facts_by_object("France")
            >>> len(facts)
            2
        """
        obj_norm = self._normalize(obj)
        return [f for f in self._facts if self._normalize(f.object) == obj_norm]

    def get_facts_by_predicate(self, predicate: str) -> list[Fact]:
        """
        Get all facts with a given predicate.

        Args:
            predicate: Predicate to search for

        Returns:
            List of all facts with this predicate
        """
        return [f for f in self._facts if f.predicate == predicate]

    def get_citation(self, fact: Fact) -> str:
        """
        Format citation for a fact.

        Args:
            fact: Fact to cite

        Returns:
            Citation string

        Example:
            >>> fact = Fact("Earth", "shape", "round", source="NASA")
            >>> fs = FactStore(VectorSpace(), Codebook(VectorSpace()))
            >>> fs.get_citation(fact)
            '[NASA] Earth --shape--> round'
        """
        source_str = f"[{fact.source}] " if fact.source else ""
        return f"{source_str}{fact}"

    @property
    def fact_count(self) -> int:
        """Get total number of stored facts."""
        return len(self._facts)

    def get_all_facts(self) -> list:
        """Get all stored facts.

        Returns:
            List of all Fact objects stored in memory.
        """
        return list(self._facts)

    @property
    def vocabulary_size(self) -> int:
        """Get number of unique object values."""
        return len(self._value_vocab)

    @property
    def saturation_estimate(self) -> float:
        """Get memory saturation estimate from underlying trace."""
        return self._memory.saturation_estimate

    def __repr__(self) -> str:
        return (
            f"FactStore(facts={self.fact_count}, "
            f"vocabulary={self.vocabulary_size}, "
            f"saturation={self.saturation_estimate:.2%})"
        )

    # ------------------------------------------------------------------ #
    # Optional FAISS prefilter for value cleanup at scale
    # ------------------------------------------------------------------ #
    def _ensure_value_faiss(self, value_list: list[str]) -> None:
        """Create or expand the value FAISS index if threshold exceeded."""
        if self._value_faiss is None:
            try:
                from hologram.persistence.faiss_adapter import FaissAdapter
            except ImportError:
                return

            try:
                self._value_faiss = FaissAdapter(
                    self._memory._space.dimensions, "/tmp/hologram_value_faiss"
                )
            except Exception:
                self._value_faiss = None
                return

            for v in value_list:
                vec = self._value_vectors_cache.get(v) or self._codebook.encode(v)
                self._value_vectors_cache[v] = vec
                try:
                    self._value_faiss.store(vec, {"value": v})
                except Exception:
                    continue
        else:
            # Add any missing values to the existing index
            indexed_meta = getattr(self._value_faiss, "metadata", {})
            for v in value_list:
                if isinstance(indexed_meta, dict) and v in indexed_meta.values():
                    continue
                vec = self._value_vectors_cache.get(v) or self._codebook.encode(v)
                self._value_vectors_cache[v] = vec
                try:
                    self._value_faiss.store(vec, {"value": v})
                except Exception:
                    continue

    def _prefilter_values_with_faiss(self, value_list: list[str], key_vec: torch.Tensor) -> list[str]:
        """Use FAISS to prefilter candidate values before resonance cleanup."""
        self._ensure_value_faiss(value_list)
        if not self._value_faiss:
            return value_list

        try:
            results = self._value_faiss.query(key_vec, k=min(100, len(value_list)))
        except Exception:
            return value_list

        candidates = []
        for _, _, meta in results:
            value = meta.get("value")
            if value:
                candidates.append(value)

        return candidates or value_list


class HierarchicalFactStore:
    """
    Two-tier fact storage: hot HDC memory + FAISS cold storage.

    - Hot layer: FactStore (O(1) exact lookups, ~100 fact fuzzy capacity)
    - Cold layer: FaissAdapter (O(log n) similarity search, large capacity)
    """

    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        faiss_path: str = "/tmp/hologram_faiss",
        hot_confidence_threshold: float = 0.7,
    ):
        try:
            from hologram.persistence.faiss_adapter import FaissAdapter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FAISS is required for HierarchicalFactStore. "
                "Install faiss-cpu to enable cold storage."
            ) from exc

        self._space = space
        self._codebook = codebook
        self._hot = FactStore(space, codebook)
        self._cold = FaissAdapter(dimensions=space.dimensions, persist_path=faiss_path)
        self._hot_confidence_threshold = hot_confidence_threshold

        # Track exact keys stored in cold layer to avoid duplicates
        self._cold_index: dict[str, int] = {}

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Optional[Fact]:
        """
        Store fact in both hot and cold layers (deduplicated by subject+predicate).
        """
        fact = self._hot.add_fact(subject, predicate, obj, source=source, confidence=confidence)

        # Always ensure cold layer has this key/object
        subject_norm = self._hot._normalize(subject)
        predicate_norm = self._hot._normalize(predicate)
        exact_key = f"{subject_norm}:{predicate_norm}"

        if exact_key not in self._cold_index:
            key_vec = Operations.bind(
                self._codebook.encode(subject_norm),
                self._codebook.encode(predicate_norm),
            )
            vec_id = self._cold.store(
                key_vec,
                {"subject": subject, "predicate": predicate, "object": obj, "source": source or "unknown"},
            )
            self._cold_index[exact_key] = vec_id

        return fact

    def query(self, subject: str, predicate: str) -> tuple[str, float]:
        """
        Query hot memory first; fall back to FAISS if confidence is low.
        """
        answer, conf = self._hot.query(subject, predicate)
        if conf >= self._hot_confidence_threshold:
            return answer, conf

        # Cold search
        subject_norm = self._hot._normalize(subject)
        predicate_norm = self._hot._normalize(predicate)
        key_vec = Operations.bind(
            self._codebook.encode(subject_norm),
            self._codebook.encode(predicate_norm),
        )

        results = self._cold.query(key_vec, k=1)
        if not results:
            return answer, conf

        _, score, meta = results[0]
        return meta.get("object", ""), float(score)

    def query_subject(self, predicate: str, obj: str) -> tuple[str, float]:
        """Reverse query delegated to hot layer."""
        return self._hot.query_subject(predicate, obj)

    def get_facts_by_subject(self, subject: str) -> list[Fact]:
        """Delegated subject metadata lookup."""
        return self._hot.get_facts_by_subject(subject)

    def get_facts_by_predicate(self, predicate: str) -> list[Fact]:
        """Delegated predicate metadata lookup."""
        return self._hot.get_facts_by_predicate(predicate)

    def get_facts_by_object(self, obj: str) -> list[Fact]:
        """Delegated object metadata lookup."""
        return self._hot.get_facts_by_object(obj)

    def save(self) -> None:
        """Persist cold layer and deduplication index."""
        self._cold.save()
        index_path = Path(self._cold.persist_path) / "cold_index.json"
        index_path.write_text(json.dumps(self._cold_index))

    def load(self) -> None:
        """Restore cold layer and deduplication index."""
        self._cold.load()
        index_path = Path(self._cold.persist_path) / "cold_index.json"
        if index_path.exists():
            self._cold_index = json.loads(index_path.read_text())

    @property
    def fact_count(self) -> int:
        return self._hot.fact_count

    @property
    def vocabulary_size(self) -> int:
        return self._hot.vocabulary_size

    @property
    def saturation_estimate(self) -> float:
        return self._hot.saturation_estimate

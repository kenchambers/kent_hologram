"""
Ingestion Router: Smart pre-filtering for dual-stream memory.

Routes incoming text to the appropriate memory stream:
- High-confidence triples (>0.8) → Direct to semantic store (FactStore)
- Everything else → Episodic buffer for salience filtering

Based on cognitive science research:
- Complementary Learning Systems (McClelland): Fast episodic + slow semantic
- Global Workspace Theory (Baars): Attentional bottleneck for consolidation

The router extracts structured knowledge from unstructured text using
spaCy NER and dependency parsing, scoring confidence based on:
- Named entity presence (proper nouns, entities)
- Triple completeness (subject + predicate + object)
- Syntactic clarity (clear dependency structure)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RouteDecision(Enum):
    """Where to route extracted knowledge."""
    SEMANTIC = "semantic"      # High confidence → direct to FactStore
    EPISODIC = "episodic"      # Low confidence → episodic buffer
    DISCARD = "discard"        # Too noisy to store


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from text."""

    subject: str
    predicate: str
    object: str
    confidence: float
    route: RouteDecision
    source_text: str
    source_id: str
    timestamp: float = field(default_factory=time.time)

    # Extraction metadata
    has_named_entity: bool = False
    is_complete_triple: bool = False
    syntactic_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"({self.subject} --{self.predicate}--> {self.object}) "
            f"[conf={self.confidence:.2f}, route={self.route.value}]"
        )


class IngestionRouter:
    """
    Smart router for dual-stream memory architecture.

    Extracts structured knowledge from text and routes to appropriate
    memory stream based on confidence scoring.

    Key behaviors:
    - High-confidence triples (>0.8) → Semantic store (ChromaDB/FactStore)
    - Medium-confidence (0.3-0.8) → Episodic buffer for salience filtering
    - Low-confidence (<0.3) → Discarded as noise

    Example:
        >>> router = IngestionRouter(fact_store, consolidation_manager)
        >>> results = router.ingest("Paris is the capital of France.")
        >>> results[0].route
        RouteDecision.SEMANTIC
        >>> results[0].subject
        'Paris'
    """

    def __init__(
        self,
        fact_store=None,
        consolidation_manager=None,
        codebook=None,
        semantic_threshold: float = 0.8,
        episodic_threshold: float = 0.3,
        use_spacy: bool = True,
    ):
        """
        Initialize the ingestion router.

        Args:
            fact_store: FactStore for semantic memory (optional)
            consolidation_manager: ConsolidationManager for episodic buffer (optional)
            codebook: Codebook for encoding (required if consolidation_manager provided)
            semantic_threshold: Confidence threshold for semantic stream (default 0.8)
            episodic_threshold: Confidence threshold for episodic stream (default 0.3)
            use_spacy: Whether to use spaCy for NER (default True)
        """
        self._fact_store = fact_store
        self._consolidation_manager = consolidation_manager
        self._codebook = codebook
        self._semantic_threshold = semantic_threshold
        self._episodic_threshold = episodic_threshold

        # Initialize spaCy if available
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("IngestionRouter: spaCy loaded successfully")
            except (ImportError, OSError) as e:
                logger.warning(f"spaCy not available: {e}. Using heuristic extraction.")

        # Statistics
        self._total_processed = 0
        self._semantic_routed = 0
        self._episodic_routed = 0
        self._discarded = 0

    def ingest(
        self,
        text: str,
        source_id: str = "unknown",
        max_triples: int = 10,
    ) -> List[ExtractionResult]:
        """
        Ingest text and route extracted knowledge to appropriate streams.

        Args:
            text: Input text to process
            source_id: Source identifier (e.g., "gutenberg_moby_dick_ch5")
            max_triples: Maximum triples to extract per text chunk

        Returns:
            List of ExtractionResult objects with routing decisions
        """
        if not text or not text.strip():
            return []

        self._total_processed += 1

        # Extract triples from text
        extractions = self._extract_triples(text, source_id, max_triples)

        # Route each extraction
        results = []
        for extraction in extractions:
            # Compute confidence and decide route
            confidence = self._compute_confidence(extraction)
            route = self._decide_route(confidence)

            extraction.confidence = confidence
            extraction.route = route

            # Execute routing
            self._execute_route(extraction)

            results.append(extraction)

            # Update stats
            if route == RouteDecision.SEMANTIC:
                self._semantic_routed += 1
            elif route == RouteDecision.EPISODIC:
                self._episodic_routed += 1
            else:
                self._discarded += 1

        return results

    def ingest_batch(
        self,
        texts: List[str],
        source_id: str = "unknown",
        max_triples_per_text: int = 10,
    ) -> List[ExtractionResult]:
        """
        Batch ingest multiple text chunks.

        Args:
            texts: List of text chunks
            source_id: Source identifier for all chunks
            max_triples_per_text: Maximum triples per chunk

        Returns:
            Combined list of all extraction results
        """
        all_results = []
        for i, text in enumerate(texts):
            chunk_source = f"{source_id}_chunk{i}"
            results = self.ingest(text, chunk_source, max_triples_per_text)
            all_results.extend(results)

        logger.info(
            f"Batch ingestion complete: {len(texts)} chunks, "
            f"{len(all_results)} extractions "
            f"(semantic={self._semantic_routed}, episodic={self._episodic_routed}, "
            f"discarded={self._discarded})"
        )

        return all_results

    def _extract_triples(
        self,
        text: str,
        source_id: str,
        max_triples: int,
    ) -> List[ExtractionResult]:
        """
        Extract subject-predicate-object triples from text.

        Uses spaCy NER and dependency parsing if available,
        falls back to heuristic extraction otherwise.
        """
        if self._nlp is not None:
            return self._extract_with_spacy(text, source_id, max_triples)
        else:
            return self._extract_heuristic(text, source_id, max_triples)

    def _extract_with_spacy(
        self,
        text: str,
        source_id: str,
        max_triples: int,
    ) -> List[ExtractionResult]:
        """Extract triples using spaCy NER and dependency parsing."""
        doc = self._nlp(text)
        extractions = []

        # Strategy 1: Named entity + verb + object pattern
        for ent in doc.ents:
            # Find verbs related to this entity
            for token in doc:
                if token.pos_ == "VERB" and token.head.text == ent.text:
                    # Find object of the verb
                    for child in token.children:
                        if child.dep_ in ("dobj", "attr", "pobj"):
                            extraction = ExtractionResult(
                                subject=ent.text,
                                predicate=token.lemma_,
                                object=child.text,
                                confidence=0.0,  # Computed later
                                route=RouteDecision.DISCARD,  # Decided later
                                source_text=text[:200],
                                source_id=source_id,
                                has_named_entity=True,
                                is_complete_triple=True,
                                syntactic_score=0.8,
                            )
                            extractions.append(extraction)

                            if len(extractions) >= max_triples:
                                return extractions

        # Strategy 2: Subject-verb-object from dependency tree
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text
                    elif child.dep_ in ("dobj", "attr", "pobj"):
                        obj = child.text

                if subject and obj:
                    # Check if subject is a named entity
                    is_ent = any(
                        ent.text.lower() == subject.lower()
                        for ent in doc.ents
                    )

                    extraction = ExtractionResult(
                        subject=subject,
                        predicate=token.lemma_,
                        object=obj,
                        confidence=0.0,
                        route=RouteDecision.DISCARD,
                        source_text=text[:200],
                        source_id=source_id,
                        has_named_entity=is_ent,
                        is_complete_triple=True,
                        syntactic_score=0.7,
                    )
                    extractions.append(extraction)

                    if len(extractions) >= max_triples:
                        return extractions

        # Strategy 3: "X is Y" patterns (copula)
        for token in doc:
            if token.lemma_ == "be" and token.pos_ == "AUX":
                subject = None
                complement = None

                for child in token.head.children:
                    if child.dep_ == "nsubj":
                        subject = child.text
                    elif child.dep_ in ("attr", "acomp"):
                        complement = child.text

                # Also check main token's children
                if not subject or not complement:
                    for child in token.children:
                        if child.dep_ == "nsubj" and not subject:
                            subject = child.text
                        elif child.dep_ in ("attr", "acomp") and not complement:
                            complement = child.text

                if subject and complement:
                    is_ent = any(
                        ent.text.lower() == subject.lower()
                        for ent in doc.ents
                    )

                    extraction = ExtractionResult(
                        subject=subject,
                        predicate="is",
                        object=complement,
                        confidence=0.0,
                        route=RouteDecision.DISCARD,
                        source_text=text[:200],
                        source_id=source_id,
                        has_named_entity=is_ent,
                        is_complete_triple=True,
                        syntactic_score=0.6,
                    )
                    extractions.append(extraction)

                    if len(extractions) >= max_triples:
                        return extractions

        return extractions

    def _extract_heuristic(
        self,
        text: str,
        source_id: str,
        max_triples: int,
    ) -> List[ExtractionResult]:
        """Fallback heuristic extraction without spaCy."""
        extractions = []

        # Simple pattern: "X is Y" or "X is the Y of Z"
        import re

        # Pattern 1: "X is Y"
        is_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?([^.]+)",
            re.IGNORECASE
        )

        for match in is_pattern.finditer(text):
            subject = match.group(1).strip()
            obj = match.group(2).strip()

            # Check for "X is the Y of Z" pattern
            of_match = re.match(r"(\w+)\s+of\s+(.+)", obj)
            if of_match:
                predicate = of_match.group(1)
                real_subject = of_match.group(2).strip()
                extraction = ExtractionResult(
                    subject=real_subject,
                    predicate=predicate,
                    object=subject,
                    confidence=0.0,
                    route=RouteDecision.DISCARD,
                    source_text=text[:200],
                    source_id=source_id,
                    has_named_entity=subject[0].isupper(),
                    is_complete_triple=True,
                    syntactic_score=0.5,
                )
            else:
                extraction = ExtractionResult(
                    subject=subject,
                    predicate="is",
                    object=obj[:50],  # Truncate long objects
                    confidence=0.0,
                    route=RouteDecision.DISCARD,
                    source_text=text[:200],
                    source_id=source_id,
                    has_named_entity=subject[0].isupper(),
                    is_complete_triple=True,
                    syntactic_score=0.4,
                )

            extractions.append(extraction)

            if len(extractions) >= max_triples:
                break

        return extractions

    def _compute_confidence(self, extraction: ExtractionResult) -> float:
        """
        Compute confidence score for an extraction.

        Factors:
        - Named entity presence: +0.3
        - Complete triple: +0.3
        - Syntactic clarity: +0.4 * syntactic_score
        """
        score = 0.0

        # Named entity bonus (proper nouns are more reliable)
        if extraction.has_named_entity:
            score += 0.3

        # Complete triple bonus
        if extraction.is_complete_triple:
            score += 0.3

        # Syntactic quality score
        score += 0.4 * extraction.syntactic_score

        return min(1.0, score)

    def _decide_route(self, confidence: float) -> RouteDecision:
        """Decide routing based on confidence."""
        if confidence >= self._semantic_threshold:
            return RouteDecision.SEMANTIC
        elif confidence >= self._episodic_threshold:
            return RouteDecision.EPISODIC
        else:
            return RouteDecision.DISCARD

    def _execute_route(self, extraction: ExtractionResult) -> None:
        """Execute the routing decision."""
        if extraction.route == RouteDecision.SEMANTIC:
            self._route_to_semantic(extraction)
        elif extraction.route == RouteDecision.EPISODIC:
            self._route_to_episodic(extraction)
        # DISCARD: do nothing

    def _route_to_semantic(self, extraction: ExtractionResult) -> None:
        """Route high-confidence extraction directly to semantic store."""
        if self._fact_store is not None:
            try:
                self._fact_store.add_fact(
                    subject=extraction.subject,
                    predicate=extraction.predicate,
                    obj=extraction.object,
                    source=extraction.source_id,
                    confidence=extraction.confidence,
                )
                logger.debug(
                    f"Routed to semantic: {extraction.subject} "
                    f"--{extraction.predicate}--> {extraction.object}"
                )
            except Exception as e:
                logger.warning(f"Failed to route to semantic: {e}")

    def _route_to_episodic(self, extraction: ExtractionResult) -> None:
        """Route medium-confidence extraction to episodic buffer."""
        if self._consolidation_manager is not None and self._codebook is not None:
            try:
                from hologram.core.operations import Operations

                # Encode as key-value pair
                s_vec = self._codebook.encode(extraction.subject.lower())
                p_vec = self._codebook.encode(extraction.predicate.lower())
                o_vec = self._codebook.encode(extraction.object.lower())

                key = Operations.bind(s_vec, p_vec)

                # Store in episodic buffer with salience from confidence
                salience = extraction.confidence * 0.7  # Discount confidence
                self._consolidation_manager.store(
                    key=key,
                    value=o_vec,
                    value_label=extraction.object,
                    source_id=extraction.source_id,
                    salience=salience,
                )
                logger.debug(
                    f"Routed to episodic: {extraction.subject} "
                    f"--{extraction.predicate}--> {extraction.object} "
                    f"(salience={salience:.2f})"
                )
            except Exception as e:
                logger.warning(f"Failed to route to episodic: {e}")

    def get_stats(self) -> dict:
        """Get router statistics."""
        return {
            "total_processed": self._total_processed,
            "semantic_routed": self._semantic_routed,
            "episodic_routed": self._episodic_routed,
            "discarded": self._discarded,
            "semantic_threshold": self._semantic_threshold,
            "episodic_threshold": self._episodic_threshold,
            "spacy_available": self._nlp is not None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_processed = 0
        self._semantic_routed = 0
        self._episodic_routed = 0
        self._discarded = 0

    def __repr__(self) -> str:
        return (
            f"IngestionRouter(semantic_threshold={self._semantic_threshold}, "
            f"episodic_threshold={self._episodic_threshold}, "
            f"spacy={self._nlp is not None})"
        )

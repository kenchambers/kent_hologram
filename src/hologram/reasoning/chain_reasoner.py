"""
Chain Reasoning: Multi-step reasoning via indexed lookup (attention-style).

Implements transformer-like attention mechanism over facts for multi-hop reasoning.
Each step queries ChromaDB/FAISS individually to avoid superposition noise from
bundled traces.

This is the "chain reasoning" counterpart to the bundled MemoryTrace:
- MemoryTrace (Hopfield): Single-step, graceful degradation under corruption
- ChainReasoner (Attention): Multi-step, precise chains with hard cleanup

Uses residual context accumulation (like transformer residual stream) to
disambiguate multi-step chains.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from hologram.core.operations import Operations


@dataclass
class ChainStep:
    """A single step in a reasoning chain."""
    subject: str
    predicate: str
    answer: str
    confidence: float


@dataclass
class ChainResult:
    """Result of multi-step chain reasoning."""
    steps: List[ChainStep]
    final_answer: str
    min_confidence: float

    def __bool__(self):
        """Allow truthiness check on ChainResult."""
        return bool(self.steps) and self.min_confidence > 0.0


class ChainReasoner:
    """
    Multi-step reasoning engine using indexed fact lookup.

    Implements transformer-style attention over individual facts, avoiding
    the superposition noise of bundled memory traces. Each reasoning step:

    1. Queries ChromaDB/FAISS for the next fact (attention over K-V pairs)
    2. Uses hard cleanup to extract clean answer (argmax, not softmax)
    3. Accumulates context via residual bundling (transformer residual stream)

    This maps directly to transformer components:
    - ChromaDB query = multi-head attention over (K, V) pairs
    - Hard cleanup = argmax (bounded hallucination)
    - Context bundling = residual connection

    Attributes:
        _codebook: Codebook for encoding concepts
        _chroma_store: ChromaFactStore for indexed fact lookup (or FAISS adapter)
        _threshold: Minimum confidence to accept a chain step (default: 0.35)

    Example:
        >>> reasoner = ChainReasoner(codebook, chroma_store)
        >>> # Single-hop: "What is the capital of France?"
        >>> result = reasoner.chain_reason("France", ["capital"])
        >>> print(result.final_answer)  # "Paris"
        >>>
        >>> # Multi-hop: "What continent is Paris's country on?"
        >>> result = reasoner.chain_reason_with_context("Paris", ["country", "continent"])
        >>> print(result.final_answer)  # "Europe"
    """

    def __init__(
        self,
        codebook,
        chroma_store,  # ChromaFactStore or FAISS adapter
        threshold: float = 0.35,
    ):
        """
        Initialize chain reasoner.

        Args:
            codebook: Codebook for encoding concepts to vectors
            chroma_store: ChromaFactStore (or FAISS adapter) for indexed lookup
            threshold: Minimum confidence threshold for chain steps (default: 0.35)
        """
        self._codebook = codebook
        self._chroma_store = chroma_store
        self._threshold = threshold

    def chain_reason(
        self,
        start: str,
        relation_chain: List[str],
        max_depth: int = 5,
    ) -> Optional[ChainResult]:
        """
        Multi-step reasoning via indexed lookup (attention-style).

        Each step queries ChromaDB/FAISS individually — no superposition noise.
        Equivalent to a transformer doing multi-hop attention.

        Args:
            start: Starting concept (e.g., "France")
            relation_chain: Sequence of relations to follow (e.g., ["capital", "population"])
            max_depth: Maximum chain length (safety limit)

        Returns:
            ChainResult with all steps, or None if chain breaks

        Example:
            >>> # "What is the capital of France?"
            >>> result = reasoner.chain_reason("France", ["capital"])
            >>> # result.final_answer == "Paris"
            >>>
            >>> # "What is the population of France's capital?"
            >>> result = reasoner.chain_reason("France", ["capital", "population"])
            >>> # result.steps[0]: France --capital--> Paris
            >>> # result.steps[1]: Paris --population--> 2.1 million
        """
        if len(relation_chain) > max_depth:
            return None

        current = start
        steps = []
        min_confidence = 1.0

        for relation in relation_chain:
            # Query ChromaDB/FAISS (attention over individual facts)
            answer, conf = self._chroma_store.query(current, relation, n_results=1)

            if not answer or conf < self._threshold:
                # Chain breaks → refuse (bounded hallucination)
                return None

            # Record step
            steps.append(ChainStep(
                subject=current,
                predicate=relation,
                answer=answer,
                confidence=conf,
            ))

            # Move to next step (hard cleanup - answer is already clean from query)
            current = answer
            min_confidence = min(min_confidence, conf)

        return ChainResult(
            steps=steps,
            final_answer=current,
            min_confidence=min_confidence,
        )

    def chain_reason_with_context(
        self,
        start: str,
        relation_chain: List[str],
        max_depth: int = 5,
        top_k: int = 3,
    ) -> Optional[ChainResult]:
        """
        Multi-step reasoning with residual context for disambiguation.

        Transformers accumulate context via residual connections: h = h + f(h).
        This lets later steps "remember" earlier decisions. When a chain step
        is ambiguous (e.g., "What continent is Paris's country on?"), the
        context vector biases toward answers coherent with chain history.

        Args:
            start: Starting concept
            relation_chain: Sequence of relations to follow
            max_depth: Maximum chain length
            top_k: Number of candidates to consider per step (for disambiguation)

        Returns:
            ChainResult with all steps, or None if chain breaks

        Example:
            >>> # Disambiguation via context
            >>> result = reasoner.chain_reason_with_context("Paris", ["country", "continent"])
            >>> # Step 1: Paris --country--> France (context = Paris + France)
            >>> # Step 2: France --continent--> Europe (context biases toward geographic answer)
        """
        if len(relation_chain) > max_depth:
            return None

        current = start
        context_vec = self._codebook.encode(start)  # Residual stream initialization
        steps = []
        min_confidence = 1.0

        for relation in relation_chain:
            # Query for top-k candidates (like multi-head attention)
            # Note: ChromaFactStore.query() returns (answer, confidence) for n_results=1
            # For multiple results, we need to query differently
            # For now, use single query and extend later if needed

            # Simple version: query single best match
            answer, conf = self._chroma_store.query(current, relation, n_results=1)

            if not answer or conf < self._threshold:
                # Chain breaks → refuse
                return None

            # If we wanted to use top_k disambiguation, we'd query multiple candidates
            # and use context_vec to rank them. For MVP, single query is sufficient.
            # TODO: Extend ChromaFactStore to support query_multiple() for top-k

            # Record step
            steps.append(ChainStep(
                subject=current,
                predicate=relation,
                answer=answer,
                confidence=conf,
            ))

            # Residual connection: accumulate into context
            # (like transformer's h_{t+1} = h_t + Attention(h_t))
            step_vec = self._codebook.encode(answer)
            context_vec = Operations.bundle(context_vec, step_vec)

            # Move to next step
            current = answer
            min_confidence = min(min_confidence, conf)

        return ChainResult(
            steps=steps,
            final_answer=current,
            min_confidence=min_confidence,
        )

    def query_similar(
        self,
        subject: str,
        predicate: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Query for multiple similar facts (for disambiguation).

        This is a helper that queries ChromaDB for top-k candidates,
        which can then be re-ranked using context vectors.

        Args:
            subject: Subject to query
            predicate: Predicate to query
            top_k: Number of results to return

        Returns:
            List of (answer, confidence) tuples, sorted by confidence

        Note:
            This is a placeholder. Full implementation requires extending
            ChromaFactStore to support multiple results with metadata.
        """
        # For now, just return single best match
        # TODO: Extend ChromaFactStore.query() to support n_results > 1
        answer, conf = self._chroma_store.query(subject, predicate, n_results=1)
        if answer:
            return [(answer, conf)]
        return []

    def __repr__(self) -> str:
        fact_count = getattr(self._chroma_store, 'fact_count', '?')
        return f"ChainReasoner(facts={fact_count}, threshold={self._threshold})"

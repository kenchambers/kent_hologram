# Dual-Stream Memory Architecture for Kent Hologram
## Integrating Hippocampal (Episodic) and Neocortical (Semantic) Learning

**Design Status:** Proposal (Ready for Implementation Planning)
**Integration Target:** Gutenberg ingestion + Conversational training
**Cognitive Foundations:** McClelland complementary learning systems, Baars global workspace theory, Plate holographic reduced representations

---

## Executive Summary

Your HDC system currently treats all facts equally: raw Gutenberg chunks bundle directly into holographic memory, causing saturation and interference ("Steele in the composition..."). This proposal separates memory into two specialized streams:

1. **Episodic Stream (Hippocampus-like)**: Fast pattern-separated storage of specific episodes (book passages, user utterances). Includes temporal and source context binding.

2. **Semantic Stream (Neocortex-like)**: Slow, integrated, generalizable knowledge extracted via consolidation. Grounded, persistent, serves as resonator fact-check.

3. **Attentional Selection (Baars Workspace)**: Bottleneck mechanism selecting which episodic memories are "worth" consolidating. Prevents saturation by filtering on novelty/relevance/salience.

**Key Innovation:** Use distinct role vectors for semantic, episodic, and contextual components. They bundle together without interference but can be selectively unbind or gated.

---

## Architectural Diagram

### System Overview

```
                          USER INPUT
                              ↓
                    ┌─────────────────────┐
                    │  Ingestion Router   │
                    │  (Smart filter)     │
                    └──┬──────────────┬───┘
                       │              │
        High confidence │              │ Medium confidence
                 facts  │              │ episodes
                       ↓              ↓
    ┌──────────────────────────────────────────────────┐
    │                                                   │
    │  SEMANTIC STREAM (Neocortex)  EPISODIC STREAM   │
    │  ═══════════════════════════  ═══════════════════ │
    │  • ChromaDB storage           • MemoryTrace      │
    │  • NeuralMemory (trained)     • Working memory   │
    │  • Persistent facts           • Short-lived      │
    │  • Slow learning              • Fast learning    │
    │  • Integrated structure       • Pattern-separ.   │
    │  • General knowledge          • Specific events  │
    │  • High confidence            • + source tags    │
    │                                                   │
    │  S-V-O with R_semantic        Episodes bundled   │
    │  [Permanent, recoverable]     [Temporal, noisy]  │
    │                                                   │
    └──────────────────────────────────────────────────┘
                       ↑              ↑
                       │              │
                  ┌────┴──────────────┴────┐
                  │  CONSOLIDATION PHASE   │
                  │  (Sleep / Background)  │
                  │  • Interleaved replay  │
                  │  • Pattern extraction  │
                  │  • Salience filtering  │
                  └────────────────────────┘
                              ↑
                    ┌─────────┴──────────┐
                    │                    │
              ┌─────▼─────┐      ┌──────▼────┐
              │ ATTENTIONAL│      │ WORKSPACE │
              │ SELECTION  │      │ (Broadcast)
              │ • Novelty  │      │ What's    │
              │ • Relevance│      │ currently │
              │ • Rarity   │      │ in focus? │
              │ • Emotion  │      │           │
              └────────────┘      └──────┬────┘
                                        │
                                   ┌────▼────────┐
                                   │ RESONATOR   │
                                   │ OUTPUT GATE │
                                   │ • Verify    │
                                   │   facts     │
                                   │ • Gate      │
                                   │   hallucin. │
                                   └─────────────┘
```

### Memory State Machine

```
                     INGESTION
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    [User input]   [Gutenberg]     [System]
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                  [Routing Decision]
                        │
        ┌───────────────┼───────────────┐
        │ High conf     │ Medium conf   │
        │ facts         │ episodes      │
        ▼               ▼               │
    [Semantic]     [Episodic]          │
    [ChromaDB]     [MemoryTrace]       │
        │               │              │
        │ (Persistent)  │ (Temporary)  │
        │               │              │
        │       ┌───────┴──────┐       │
        │       ▼              ▼       │
        │   [Workspace Selection]     │
        │   [Salience Filter]         │
        │       │                      │
        │       │ Selected episodes    │
        │       │ (High salience)      │
        │       ▼                      │
        │   [Consolidation]           │
        │   [Sleep phase]             │
        │   [Interleaved replay]      │
        │   [Pattern extraction]      │
        │       │                      │
        │       ▼                      │
        └─→ [Semantic Memory]         │
            [Neural integration]      │
                                      │
            [Decay episodic] ←────────┘
            (Can now fade)
```

---

## Component Data Structures

### 1. Episodic Memory Bundle

```python
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class EpisodeBundled:
    """
    A single episode: book passage, user utterance, or system event.

    Stored in working memory (MemoryTrace).
    Associated with distinct role vectors to prevent interference.
    """

    # Core content
    passage_vector: torch.Tensor
    """Encoded summary of passage: codebook.encode("Ahab captained...")"""

    # Source binding: WHERE did this come from?
    source_book_id: str  # e.g., "gutenberg_moby_dick"
    source_chapter: Optional[str]  # e.g., "chapter_5"
    source_section: Optional[str]  # e.g., "beginning"
    source_type: str  # "gutenberg" | "user" | "system"

    # Time binding: WHEN did we encounter this?
    timestamp: float  # Unix timestamp or episode index
    episode_index: int  # Relative order in stream

    # Salience: How important is this episode?
    salience: float  # [0.0, 1.0] computed dynamically
    confidence: float  # How confident is passage summary? [0.0, 1.0]

    # Access tracking: For decay and replay selection
    retrieval_count: int  # How often was this retrieved?
    last_accessed: float  # Timestamp of last retrieval
    consolidated_at: Optional[float]  # When did it consolidate? (None if not yet)

    # The actual holographic vector (fully bundled with roles)
    episodic_bundle: torch.Tensor
    """
    = bind(passage_vector, R_episode)
    + bind(source_binding, R_source)
    + bind(temporal_vector, R_time)
    + bind(context_vector, R_context)
    """

    # For debugging/introspection
    human_readable: str  # E.g., "moby_dick_ch5: Ahab commanded..."


def create_episode_bundle(
    passage: str,
    codebook,
    ops,
    source_id: str,
    source_chapter: Optional[str] = None,
    timestamp: float = None,
    context_vector: Optional[torch.Tensor] = None,
    salience: float = 0.5,
) -> EpisodeBundled:
    """
    Factory function to create a bundled episode vector.

    Separates concerns:
    - Passage gets R_episode role
    - Source metadata gets R_source role
    - Time gets R_time role
    - Context gets R_context role

    No interference because roles are orthogonal.
    """

    # Encode passage
    passage_vector = codebook.encode(passage)

    # Create source binding
    source_book = codebook.encode(source_id)
    source_book_vec = ops.bind(source_book, codebook.get_role("SOURCE_BOOK"))

    if source_chapter:
        chapter_vec = codebook.encode(source_chapter)
        source_chapter_vec = ops.bind(chapter_vec, codebook.get_role("SOURCE_CHAPTER"))
    else:
        source_chapter_vec = torch.zeros_like(passage_vector)

    source_binding = source_book_vec + source_chapter_vec

    # Temporal encoding
    if timestamp is None:
        timestamp = time.time()

    temporal_vec = ops.permute(
        codebook.get_role("TIME_BASIS"),
        shifts=int(timestamp) % 10000  # Keep shifts bounded
    )

    # Context: user's current focus/mood/intent
    if context_vector is None:
        context_vector = torch.ones_like(passage_vector) * 0.1

    # Bundle with distinct roles (no interference)
    episodic_bundle = ops.bundle(
        ops.bind(passage_vector, codebook.get_role("EPISODE")),
        ops.bind(source_binding, codebook.get_role("SOURCE")),
        ops.bind(temporal_vec, codebook.get_role("EPISODE_TEMPORAL")),
        ops.bind(context_vector, codebook.get_role("CONTEXT")),
    )

    return EpisodeBundled(
        passage_vector=passage_vector,
        source_book_id=source_id,
        source_chapter=source_chapter,
        source_section=None,
        source_type="gutenberg" if "gutenberg" in source_id else "user",
        timestamp=timestamp,
        episode_index=0,  # Updated by buffer manager
        salience=salience,
        confidence=0.7,
        retrieval_count=0,
        last_accessed=timestamp,
        consolidated_at=None,
        episodic_bundle=episodic_bundle,
        human_readable=f"{source_id}: {passage[:50]}...",
    )
```

### 2. Semantic Fact with Source Tracking

```python
@dataclass
class SemanticFact:
    """
    A generalized, high-confidence fact extracted from consolidation.

    Stored persistently in ChromaDB + NeuralMemory.
    Serves as ground truth for resonator verification.
    """

    # Core triple
    subject: str
    predicate: str
    object: str

    # Vectors (for HDC operations)
    subject_vector: torch.Tensor
    predicate_vector: torch.Tensor
    object_vector: torch.Tensor

    # Semantic triple vector (what gets stored in ChromaDB)
    semantic_vector: torch.Tensor
    """
    = bind(subject_vector, R_semantic_subject)
    + bind(predicate_vector, R_semantic_predicate)
    + bind(object_vector, R_semantic_object)
    """

    # Source provenance: Which episodes led to this?
    sources: List[dict]  # [{"book": "moby_dick", "chapter": "5", ...}]
    source_agreement: float  # What % of sources agree?

    # Confidence: How certain are we this is true?
    confidence: float  # [0.0, 1.0], based on # sources + agreement

    # Timing
    learned_at: float  # When was this fact consolidated?
    first_encountered: float  # When did we first see evidence?

    # Integration
    integration_level: float  # How well does it fit with existing facts?
    """
    0.0 = Isolated fact
    0.5 = Somewhat connected
    1.0 = Tightly integrated with knowledge graph
    """

    retrieval_count: int  # How often retrieved? (for weighting)

    def to_dense_triple(self) -> str:
        """For human readability."""
        return f"({self.subject}, {self.predicate}, {self.object})"
```

### 3. Salience Score (Attention Mechanism)

```python
def compute_salience(
    episode: EpisodeBundled,
    existing_facts: List[SemanticFact],
    user_intent: torch.Tensor,
    system_state,
    codebook,
    ops,
) -> float:
    """
    Compute how "important" an episode is for consolidation.

    Factors:
    - Novelty: Does it contradict/extend/fill gaps in knowledge?
    - Relevance: Does it match current user interests?
    - Rarity: How common is this fact globally?
    - Emotional: Was user emotionally engaged?

    Returns: [0.0, 1.0] scalar for workspace selection.
    """

    # 1. NOVELTY SCORE
    # Extract predicted triples from episode
    predicted_triples = _predict_triples_from_episode(episode)

    # Compare to existing facts
    novelty_scores = []
    for triple in predicted_triples:
        # Does this triple already exist?
        existing = [f for f in existing_facts
                   if f.subject == triple.subject
                   and f.predicate == triple.predicate
                   and f.object == triple.object]

        if existing:
            # Old knowledge → low novelty
            novelty_scores.append(0.1)
        elif any(f.subject == triple.subject for f in existing_facts):
            # Partial match → medium novelty
            novelty_scores.append(0.5)
        else:
            # Completely new entity → high novelty
            novelty_scores.append(0.9)

    novelty = sum(novelty_scores) / max(1, len(novelty_scores)) if novelty_scores else 0.3

    # 2. RELEVANCE SCORE
    # Does user care about this topic?
    relevance = float(
        ops.cosine_similarity(
            episode.passage_vector,
            user_intent  # From workspace context
        ).item()
    )
    relevance = max(0.0, relevance)  # Clip to [0, 1]

    # 3. RARITY SCORE
    # Uncommon facts are worth preserving
    predicted_subjects = [t.subject for t in predicted_triples]
    occurrence_counts = [
        sum(1 for f in existing_facts if f.subject == s)
        for s in predicted_subjects
    ]

    if occurrence_counts:
        # Inverse of occurrence count (rare = high score)
        rarity = 1.0 / (1.0 + sum(occurrence_counts) / len(occurrence_counts))
    else:
        rarity = 1.0  # Completely new = very rare

    # 4. EMOTIONAL WEIGHT (for conversations)
    emotional_weight = 0.1
    if episode.source_type == "user":
        # User input might be emotionally charged
        # Could parse sentiment here (for now, fixed weight)
        emotional_weight = 0.2
    elif episode.source_type == "system":
        # System messages less emotional
        emotional_weight = 0.05

    # 5. WEIGHTED COMBINATION
    salience = (
        0.4 * novelty +
        0.3 * relevance +
        0.2 * rarity +
        0.1 * emotional_weight
    )

    return max(0.0, min(1.0, salience))


def select_for_workspace(
    episodic_buffer: List[EpisodeBundled],
    existing_facts: List[SemanticFact],
    user_intent: torch.Tensor,
    system_state,
    workspace_capacity: int = 50,
    salience_threshold: float = 0.3,
    codebook = None,
    ops = None,
) -> Tuple[List[EpisodeBundled], torch.Tensor]:
    """
    Select top-K episodes for consolidation (Global Workspace bottleneck).

    Implements Baars' theory: only ~50 items in conscious workspace at once.
    Selection based on salience (novelty, relevance, rarity).

    Returns:
    - selected: List of EpisodeBundled in workspace
    - workspace_vector: Bundled vector of all selected
    """

    # Score all episodes
    scored_episodes = []
    for episode in episodic_buffer:
        score = compute_salience(
            episode,
            existing_facts,
            user_intent,
            system_state,
            codebook,
            ops,
        )
        episode.salience = score  # Update in-place
        scored_episodes.append((episode, score))

    # Filter by threshold (ignore very low salience)
    above_threshold = [
        (ep, score) for ep, score in scored_episodes
        if score >= salience_threshold
    ]

    if not above_threshold:
        # No salient episodes → return empty workspace
        return [], ops.bundle()  # Empty bundle

    # Sort by score (descending), break ties by recency
    above_threshold.sort(
        key=lambda x: (-x[1], -x[0].last_accessed)
    )

    # Take top-K
    selected = [ep for ep, _ in above_threshold[:workspace_capacity]]

    # Bundle selected episodes into workspace vector
    workspace_vector = ops.bundle(*[ep.episodic_bundle for ep in selected])

    return selected, workspace_vector
```

---

## Ingestion Router: Smart Pre-Filter

```python
class IngestingRouter:
    """
    Routes Gutenberg data to appropriate stream.
    Prevents raw saturation by extracting structure first.
    """

    def __init__(self, codebook, ops, nlp=None):
        self.codebook = codebook
        self.ops = ops
        self.nlp = nlp  # spaCy model for NER/POS
        self.resonator = None  # For decomposition

    def route_gutenberg_chunk(
        self,
        text: str,
        book_id: str,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
    ) -> dict:
        """
        Process Gutenberg chunk: extract facts, route to streams.

        Returns: {
            'semantic_facts': List[SemanticFact],  # High confidence
            'episodic_bundle': EpisodeBundled,     # Medium confidence
        }
        """

        # Step 1: Extract entities and relations
        entities = self._extract_entities(text)
        relations = self._extract_relations(text)
        triples = self._extract_triples(text)

        # Step 2: Separate high/medium confidence
        semantic_facts = []
        episodic_bundles = []

        for triple in triples:
            if triple.confidence >= 0.8:
                # High confidence → Semantic stream directly
                fact = SemanticFact(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object,
                    subject_vector=self.codebook.encode(triple.subject),
                    predicate_vector=self.codebook.encode(triple.predicate),
                    object_vector=self.codebook.encode(triple.object),
                    sources=[{
                        'book': book_id,
                        'chapter': chapter,
                        'section': section,
                        'confidence': triple.confidence,
                    }],
                    source_agreement=1.0,
                    confidence=min(0.95, triple.confidence * 1.1),  # Boost
                    learned_at=time.time(),
                    first_encountered=time.time(),
                    integration_level=0.0,  # Will be computed later
                )
                fact.semantic_vector = self._compute_semantic_vector(fact)
                semantic_facts.append(fact)

        # Step 3: Create episodic episode for whole passage
        chunk_summary = self._summarize_chunk(text, entities, relations)

        if chunk_summary:
            episode = create_episode_bundle(
                passage=chunk_summary,
                codebook=self.codebook,
                ops=self.ops,
                source_id=book_id,
                source_chapter=chapter,
                timestamp=time.time(),
                context_vector=None,
                salience=0.5,  # To be updated by workspace selection
            )
            episodic_bundles.append(episode)

        return {
            'semantic_facts': semantic_facts,
            'episodic_bundles': episodic_bundles,
        }

    def _extract_entities(self, text: str) -> List[dict]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return []
        doc = self.nlp(text)
        return [
            {'text': ent.text, 'label': ent.label_}
            for ent in doc.ents
        ]

    def _extract_relations(self, text: str) -> List[dict]:
        """Extract relations (simplified: entity pairs with predicates)."""
        # Placeholder: real implementation would use relation extraction
        return []

    def _extract_triples(self, text: str) -> List[dict]:
        """
        Extract (subject, predicate, object) triples.
        Real implementation would use SRL (semantic role labeling).
        """
        # Placeholder
        return []

    def _summarize_chunk(
        self,
        text: str,
        entities: List[dict],
        relations: List[dict],
    ) -> str:
        """Create a brief summary of the chunk for episodic memory."""
        # Placeholder: could use extractive/abstractive summarization
        return text[:200] if len(text) > 200 else text

    def _compute_semantic_vector(self, fact: SemanticFact) -> torch.Tensor:
        """Compute the bundled semantic vector."""
        return self.ops.bundle(
            self.ops.bind(fact.subject_vector, self.codebook.get_role("SEMANTIC_SUBJECT")),
            self.ops.bind(fact.predicate_vector, self.codebook.get_role("SEMANTIC_PREDICATE")),
            self.ops.bind(fact.object_vector, self.codebook.get_role("SEMANTIC_OBJECT")),
        )
```

---

## Consolidation: Interleaved Replay + Pattern Extraction

```python
class InterleavedReplayConsolidationManager(ConsolidationManager):
    """
    Enhanced consolidation with:
    - Interleaved replay (old facts + new facts mixed)
    - Pattern extraction from episodes
    - Prevents catastrophic forgetting
    - Promotes integration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semantic_facts = []  # All consolidated facts
        self._episodic_buffer = None  # Reference to buffer

    def set_episodic_buffer(self, buffer):
        """Reference to episodic buffer for replay selection."""
        self._episodic_buffer = buffer

    def _do_consolidation(self, snapshot: ConsolidationSnapshot):
        """
        Main consolidation loop (runs in background thread).

        Process:
        1. Extract semantic structure from salient episodes
        2. Sample old facts (interleaved replay)
        3. Mix new + old, shuffle
        4. Train neural network on mixed batch
        5. Update semantic storage
        6. Decay episodic buffer
        """

        logger.info(f"Consolidation triggered: {len(snapshot.facts)} pending facts")

        # Phase 1: Extract semantic structure from episodes
        new_semantic_facts = self._extract_semantic_from_episodes(
            snapshot.facts,
            salience_threshold=0.3
        )
        logger.info(f"Extracted {len(new_semantic_facts)} new semantic facts")

        # Phase 2: Interleaved replay - sample old facts
        if self._semantic_facts:
            old_facts = self._semantic_facts_sample(
                count=min(len(new_semantic_facts), 20),
                weighted_by_recency=True
            )
        else:
            old_facts = []

        logger.info(f"Sampled {len(old_facts)} old facts for replay")

        # Phase 3: Mix and shuffle
        mixed_batch = new_semantic_facts + old_facts
        random.shuffle(mixed_batch)

        # Phase 4: Train neural network on mixed batch
        if mixed_batch:
            loss = self._neural_memory.train_batch(mixed_batch, epochs=3)
            logger.info(f"Neural training loss: {loss:.4f}")

        # Phase 5: Update semantic storage
        for fact in new_semantic_facts:
            self._upsert_semantic_fact(fact)
            self._semantic_facts.append(fact)

        # Phase 6: Decay episodic buffer
        # Episodes that consolidated can fade
        for fact in snapshot.facts:
            if self._episodic_buffer:
                self._episodic_buffer.decay_episode(fact, factor=0.5)

    def _extract_semantic_from_episodes(
        self,
        episodes: List[EpisodeBundled],
        salience_threshold: float = 0.3,
    ) -> List[SemanticFact]:
        """
        Given episodic memories, extract generalizable semantic structure.

        Example: Multiple Moby Dick chapters mention Ahab:
        - Ch5: "Ahab was the captain"
        - Ch12: "Ahab commanded the Pequod"
        - Ch23: "Ahab obsessed over Moby Dick"

        Extract:
        - ("Ahab", "was_captain_of", "Pequod") [confidence 0.9]
        - ("Ahab", "obsessed_over", "Moby Dick") [confidence 0.7]

        Not verbatim, but generalized meaning.
        """

        # Filter by salience
        salient = [ep for ep in episodes if ep.salience >= salience_threshold]
        logger.info(f"Processing {len(salient)} salient episodes")

        # Decompose each episode using resonator
        decomposed_triples = []
        for episode in salient:
            triples = self._decompose_episode_to_triples(episode)
            for triple in triples:
                triple['sources'] = [{
                    'book': episode.source_book_id,
                    'chapter': episode.source_chapter,
                    'timestamp': episode.timestamp,
                }]
            decomposed_triples.extend(triples)

        # Cluster similar triples (same S/V/O)
        clusters = self._cluster_triples(decomposed_triples)

        # Generalize within clusters
        semantic_facts = []
        for cluster in clusters:
            if len(cluster) >= 1:  # Even 1 occurrence worth storing
                fact = self._generalize_cluster(cluster)
                if fact.confidence >= 0.4:  # Minimum threshold
                    semantic_facts.append(fact)

        return semantic_facts

    def _decompose_episode_to_triples(
        self,
        episode: EpisodeBundled
    ) -> List[dict]:
        """
        Use resonator to decompose episode into (S, V, O) triples.

        Returns list of {'subject': ..., 'predicate': ..., 'object': ...}
        """
        # Placeholder: would call resonator.resonate()
        # For now, just return empty
        return []

    def _cluster_triples(self, triples: List[dict]) -> List[List[dict]]:
        """
        Cluster similar triples (same S/V/O).
        Returns: List of clusters, where each cluster has similar triples.
        """
        # Group by (subject, predicate, object)
        clusters = {}
        for triple in triples:
            key = (triple['subject'], triple['predicate'], triple['object'])
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(triple)

        return list(clusters.values())

    def _generalize_cluster(self, cluster: List[dict]) -> SemanticFact:
        """
        Given a cluster of similar triples, generalize to one semantic fact.

        Compute:
        - Best consensus (subject, predicate, object)
        - Confidence from agreement
        - Sources from all triples
        """
        s = cluster[0]['subject']
        v = cluster[0]['predicate']
        o = cluster[0]['object']

        sources = []
        for triple in cluster:
            if 'sources' in triple:
                sources.extend(triple['sources'])

        confidence = min(0.99, 0.5 + 0.4 * len(cluster) / 5)  # More sources = higher conf

        fact = SemanticFact(
            subject=s,
            predicate=v,
            object=o,
            subject_vector=self._codebook.encode(s),
            predicate_vector=self._codebook.encode(v),
            object_vector=self._codebook.encode(o),
            sources=sources,
            source_agreement=1.0 / len(cluster),
            confidence=confidence,
            learned_at=time.time(),
            first_encountered=min(src['timestamp'] for src in sources) if sources else time.time(),
            integration_level=0.0,
        )

        fact.semantic_vector = self._compute_semantic_vector(fact)
        return fact

    def _semantic_facts_sample(
        self,
        count: int,
        weighted_by_recency: bool = True,
    ) -> List[SemanticFact]:
        """Sample old facts for interleaved replay."""
        if not self._semantic_facts:
            return []

        if weighted_by_recency:
            # Recent facts more likely to be sampled
            weights = [
                1.0 / (1.0 + time.time() - fact.learned_at)
                for fact in self._semantic_facts
            ]
            weights = [w / sum(weights) for w in weights]
        else:
            weights = [1.0 / len(self._semantic_facts)] * len(self._semantic_facts)

        return random.choices(
            self._semantic_facts,
            weights=weights,
            k=min(count, len(self._semantic_facts))
        )

    def _upsert_semantic_fact(self, fact: SemanticFact):
        """Store fact in ChromaDB and neural memory."""
        # ChromaDB storage (persistent)
        # Neural memory training (already done in train_batch)
        pass

    def _compute_semantic_vector(self, fact: SemanticFact) -> torch.Tensor:
        """Compute bundled semantic vector."""
        return self._ops.bundle(
            self._ops.bind(fact.subject_vector, self._codebook.get_role("SEMANTIC_SUBJECT")),
            self._ops.bind(fact.predicate_vector, self._codebook.get_role("SEMANTIC_PREDICATE")),
            self._ops.bind(fact.object_vector, self._codebook.get_role("SEMANTIC_OBJECT")),
        )
```

---

## Dual-Stream Resonator (Output Verification Gate)

```python
class DualStreamResonator(Resonator):
    """
    Extended Resonator that gates output through fact verification.

    Prevents hallucination by checking: Does this triple exist?
    - In semantic memory (high confidence)
    - In episodic buffer (medium confidence)
    - Nowhere (hallucination)
    """

    def __init__(
        self,
        base_resonator: Resonator,
        semantic_facts: List[SemanticFact],
        episodic_buffer: List[EpisodeBundled],
    ):
        self._base = base_resonator
        self._semantic_facts = semantic_facts
        self._episodic_buffer = episodic_buffer

    def resonate(
        self,
        thought: torch.Tensor,
        noun_vocabulary: List[str],
        verb_vocabulary: List[str],
    ) -> ResonatorResult:
        """
        Factorize thought vector, then verify facts.

        Returns enhanced ResonatorResult with:
        - grounding_confidence: How well grounded in memory?
        - grounding_source: "semantic", "episodic", or "hallucinated"
        - verified_fact: The underlying SemanticFact (if exists)
        """

        # Step 1: Standard ALS decomposition
        result = self._base.resonate(thought, noun_vocabulary, verb_vocabulary)

        # Step 2: Verify triple against fact store
        verification = self._verify_triple(
            result.subject_word,
            result.verb_word,
            result.object_word
        )

        # Step 3: Add verification to result
        result.grounding_confidence = verification['confidence']
        result.grounding_source = verification['source']
        result.verified_fact = verification.get('fact', None)

        return result

    def _verify_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> dict:
        """
        Check: Does this S-P-O triple exist?

        Returns: {
            'confidence': float [0.0, 1.0],
            'source': "semantic" | "episodic" | "hallucinated",
            'fact': SemanticFact if found, else None,
        }
        """

        # First check semantic facts (high confidence)
        for fact in self._semantic_facts:
            if (fact.subject.lower() == subject.lower() and
                fact.predicate.lower() == predicate.lower() and
                fact.object.lower() == obj.lower()):

                return {
                    'confidence': fact.confidence,
                    'source': 'semantic',
                    'fact': fact,
                }

        # Then check episodic buffer (medium confidence, discounted)
        for episode in self._episodic_buffer:
            # Simplified: check if passage mentions all three
            passage_lower = episode.passage_vector.lower() if hasattr(episode.passage_vector, 'lower') else episode.human_readable.lower()
            if (subject.lower() in passage_lower and
                predicate.lower() in passage_lower and
                obj.lower() in passage_lower):

                return {
                    'confidence': 0.5,  # Episodic confidence lower
                    'source': 'episodic',
                    'fact': None,
                }

        # No match = hallucination
        return {
            'confidence': 0.0,
            'source': 'hallucinated',
            'fact': None,
        }
```

---

## Example: Processing Moby Dick Chapter

```python
# SCENARIO: User uploads Moby Dick Chapter 5
# "Captain Ahab stood at the helm. He commanded the Pequod..."

# 1. INGESTION
chunk = {
    'text': "Captain Ahab stood at the helm...",
    'book_id': 'gutenberg_moby_dick',
    'chapter': 'chapter_5',
    'source_type': 'gutenberg',
}

# 2. ROUTING
router = IngestingRouter(codebook, ops)
routes = router.route_gutenberg_chunk(
    chunk['text'],
    chunk['book_id'],
    chunk['chapter']
)

# Routes returns:
# {
#   'semantic_facts': [
#       ('Ahab', 'commanded', 'Pequod', conf=0.85),
#       ('Pequod', 'is_ship', 'whaling', conf=0.75),
#   ],
#   'episodic_bundles': [
#       EpisodeBundled(passage="Ahab commanded...", salience=0.65),
#   ]
# }

# 3. SEMANTIC STREAM (Direct storage)
for fact in routes['semantic_facts']:
    semantic_facts.append(fact)
    chromadb_store.add(fact)

# 4. EPISODIC STREAM (Working memory)
episodic_buffer_manager.add_episode(
    routes['episodic_bundles'][0],
    priority=0.65  # salience
)

# 5. WORKSPACE SELECTION (Later, during consolidation)
selected_episodes, workspace_vector = select_for_workspace(
    episodic_buffer_manager.get_episodes(),
    semantic_facts,
    user_intent=torch.ones(10000) * 0.1,  # Generic intent
)

# If Moby Dick chapter has high salience (user reading it):
# → Selected for workspace
# → Consolidated into semantic memory
# → Episode fades but semantic knowledge persists

# 6. CONSOLIDATION (Sleep phase)
consolidation_mgr.consolidate(snapshot)

# Consolidation:
# a. Extracts: ("Ahab", "commanded", "Pequod") from episode
# b. Interleaves with old "ship captain" facts
# c. Trains neural network on mixed batch
# d. Stores fact with source tracking
# e. Decays episodic memory (can now fade)

# 7. RETRIEVAL (User asks "Who commanded the Pequod?")
user_query = "Who commanded the Pequod?"
thought = compose_thought(user_query, codebook)

result = dual_stream_resonator.resonate(
    thought,
    noun_vocabulary=['Ahab', 'Starbuck', ...],
    verb_vocabulary=['commanded', 'saw', ...],
)

# Resonator returns:
# ResonatorResult(
#     subject='Ahab',
#     verb='commanded',
#     object='Pequod',
#     converged=True,
#     grounding_confidence=0.95,  # Found in semantic facts
#     grounding_source='semantic',
#     verified_fact=SemanticFact(...),
# )

# System speaks: "Ahab" (not a guess, it's grounded!)
```

---

## Key Advantages

| Problem | Solution | Benefit |
|---------|----------|---------|
| Raw Gutenberg saturation | Episodic buffer with capacity limits | Prevents "Steele in composition" garbage |
| All facts equal | Dual streams: semantic vs episodic | Books ≠ conversations, both integrate |
| Can't extract meaning | Interleaved replay consolidation | Generalizes from specific episodes |
| Resonator hallucinates | Output gate + fact verification | "I don't know" beats false confidence |
| Loses context | Temporal + source binding | Remembers WHERE facts came from |
| Catastrophic forgetting | Interleaved replay | 38% less forgetting (NeuroDream) |

---

## Next Steps

1. **Review this architecture** - Does it align with your vision?
2. **Clarify priorities:**
   - Saturation prevention (most urgent)?
   - Hallucination prevention (critical)?
   - Consolidation effectiveness (nice-to-have)?
3. **Decide implementation order:**
   - Phase 1: Build episodic buffer + salience scoring
   - Phase 2: Enhance consolidation with replay
   - Phase 3: Add resonator verification gate
   - Phase 4: Implement ingestion router
4. **Begin Phase 1** with detailed component specification

Would you like me to elaborate on any component, discuss trade-offs, or help plan Phase 1 implementation?

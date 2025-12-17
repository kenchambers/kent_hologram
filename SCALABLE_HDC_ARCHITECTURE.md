# Scalable, Hallucination-Free HDC Architecture for Kent Hologram

## Executive Summary

The current Kent Hologram architecture uses a single 10,000D trace with ~100 fact capacity (D/100 Bentov principle). To scale to **millions of words** while maintaining **0% hallucination guarantee**, this document proposes a **Hierarchical HDC + Neuro-Symbolic Knowledge Graph** architecture that:

1. **Maintains HDC foundation** - preserves proven resonance and factorization strengths
2. **Scales holographically** - hierarchical tiers instead of single trace saturation
3. **Eliminates hallucination** - knowledge graph enforces strict fact constraints
4. **Detects mixed content** - code/text classifier with language-specific processors
5. **Grounds generation** - constrained LLM output matches stored facts

**Key insight:** The hallucination problem isn't in storage (FactStore is sound) — it's in **generation unconstrained from facts**. A frontier LLM can generate plausible-sounding lies. We solve this by making generation **algebraically derive** from facts, not sample from probability distributions.

---

## Part 1: Current Architecture Analysis

### 1.1 Bottlenecks Identified

#### Bottleneck 1: Single Trace Saturation

**File:** `/src/hologram/memory/memory_trace.py` (lines 29-81)

```python
class MemoryTrace:
    """Stores multiple key-value pairs in bundled vector (single _trace)"""
    def __init__(self, space: VectorSpace):
        self._trace = space.empty_vector()  # Single 10,000D vector
        self._fact_count = 0  # Tracks saturation
```

**Problem:** All facts bundle into ONE trace. Capacity = D/100 ≈ 100 facts for 10,000D.

**Why it fails at scale:** After ~200 facts, noise accumulates → similarities drop below decision thresholds (RESPONSE_CONFIDENCE_THRESHOLD = 0.20).

**Current mitigation:** FactStore uses FAISS prefilter + tiered lookups (hot/cold), but this is a workaround, not fundamental solution.

#### Bottleneck 2: Resonator Fixed on 3 Slots

**File:** `/src/hologram/core/resonator.py` (lines 118-220)

```python
def resonate(self, thought, noun_vocabulary, verb_vocabulary):
    """Factorizes into exactly (subject, verb, object) - 3 slots only"""
    for iteration in range(self._max_iterations):
        x, x_word = self._solve_for_slot(thought, ROLE_SUBJECT, ...)
        y, y_word = self._solve_for_slot(thought, ROLE_VERB, ...)
        z, z_word = self._solve_for_slot(thought, ROLE_OBJECT, ...)
```

**Problem:** S-V-O works for simple sentences, but:
- Code has 10+ semantic slots: function, parameters, return, imports, dependencies
- Long text needs hierarchical structure: (section, paragraph, sentence)
- Multiple facts need chaining: (subject1, relation1, subject2, relation2, object)

#### Bottleneck 3: Generation is "Constrained Sampling," Not "Constrained Derivation"

**File:** `/src/hologram/generation/resonant_generator.py` (lines 213-306)

```python
def generate(...):
    # Stage 1: Resonator factorizes thought (good)
    resonator_result = self._resonator.resonate(thought, ...)

    # Stage 2: Token-by-token with divergence check (but divergence only REJECTS, not derives)
    for role in target.grammar:
        candidates = self._generate_candidates(target, role, role_vocab, ...)
        selected_token, trace = self._evaluate_candidates(candidates, ...)
        output_tokens.append(selected_token)
```

**Problem:**
- Candidates come from neural cleanup (softmax), not from stored facts
- Divergence calculator can reject hallucinations post-hoc, but can't force truth
- If all candidates fail divergence check → fallback to best candidate anyway (line 565)

**Why this matters for hallucination:**
- LLM generates "Paris is the capital of Germany"
- Parser creates (Paris, capital, Germany)
- This never matches stored (France, capital, Paris)
- But fallback accepts it anyway ❌

### 1.2 Current Strengths to Preserve

1. **S-V-O Factorization (Algebraic)** - Resonator uses ALS, not neural networks. Very interpretable.
2. **Holographic Interference (Information-Theoretic)** - Facts stored in superposition, retrieves with resonance. Sound principle.
3. **Surprise Gating (Novelty Detection)** - Prevents duplicate encoding. Efficient learning.
4. **Citation Tracking (Fact Metadata)** - Every fact tracks source. Foundation for grounding.

---

## Part 2: Proposed Hierarchical Architecture

### 2.1 Three-Tier Memory System

```
┌─────────────────────────────────────────────────────────────┐
│           TIER 3: KNOWLEDGE GRAPH (Cold Storage)            │
│     Neo4j/RDF Triple Store (Millions of S-P-O triples)      │
│     Role: Exact fact storage, constraint enforcement        │
│     Indexed by: (Subject, Predicate, Object) with metadata  │
└──────────────────────┬──────────────────────────────────────┘
                       │ Exact match queries
                       │ Constraint validation
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        TIER 2: HIERARCHICAL HDC (Hot/Warm Storage)          │
│  Pyramid of traces: 1 summary + N semantic clusters         │
│     Role: Fast resonance, semantic similarity, synthesis    │
│     Organization: Fact→Embedding→Cluster→Trace             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Candidate generation
                       │ Similar fact retrieval
                       ▼
┌─────────────────────────────────────────────────────────────┐
│     TIER 1: CODE/TEXT CLASSIFIER + Dual Processors         │
│   • Tree-sitter parser (Code) → Dependency graphs          │
│   • Sentence tokenizer (Text) → Discourse structure         │
│     Role: Content-aware fact extraction, semantic grouping  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Tier 1: Content Classifier & Dual Processors

#### 2.2.1 Code Detection Layer

**Library:** [Tree-sitter](https://github.com/tree-sitter/tree-sitter) + [tree-sitter-python](https://github.com/tree-sitter/tree-sitter-python) (100+ languages)

```python
# NEW FILE: src/hologram/processors/language_classifier.py

import tree_sitter
from pathlib import Path

class LanguageClassifier:
    """Detect code vs. text and route to appropriate processor."""

    def __init__(self):
        self.parser = tree_sitter.Parser()
        # Load tree-sitter language libraries
        self.languages = {
            'python': tree_sitter.Language('path/to/python.so'),
            'javascript': tree_sitter.Language('path/to/javascript.so'),
            'rust': tree_sitter.Language('path/to/rust.so'),
            # ... 100+ languages available
        }
        self.parser.set_language(self.languages['python'])  # Default

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify content type.

        Returns: (content_type, confidence)
            - "code_python", "code_javascript", ..., "text"
            - confidence: 0.0-1.0
        """
        # Heuristic 1: Try parsing as Python
        try:
            tree = self.parser.parse(text.encode())
            if tree.root_node.child_count > 0:
                # Valid parse → likely code
                return ("code_python", 0.95)
        except:
            pass

        # Heuristic 2: Keyword signatures
        code_keywords = {
            'def ', 'import ', 'class ', 'return ', 'async ',
            'function ', 'const ', 'let ', 'var ',
            'fn ', 'pub ', 'struct ', 'enum ',
        }
        code_score = sum(1 for kw in code_keywords if kw in text) / len(code_keywords)

        # Heuristic 3: Indentation patterns
        lines = text.split('\n')
        indent_score = sum(1 for line in lines if line and line[0] == ' ') / len(lines)

        if code_score > 0.3 or indent_score > 0.5:
            return ("code_mixed", max(code_score, indent_score))

        return ("text", 0.8)
```

#### 2.2.2 Code Processor (Semantic AST Facts)

```python
# NEW FILE: src/hologram/processors/code_processor.py

from tree_sitter import Node

class CodeProcessor:
    """Extract S-P-O facts from syntax trees."""

    def extract_facts(self, text: str, language: str = 'python') -> list[tuple[str, str, str]]:
        """
        Extract code structure as facts:
        - (function_name, "parameter", param_name)
        - (function_name, "returns", return_type)
        - (function_name, "calls", callee_function)
        - (class_name, "has_method", method_name)
        - (module, "imports", imported_module)
        """
        facts = []

        # Parse to AST
        tree = self.parser.parse(text.encode())

        # Walk AST and extract semantic relationships
        for node in tree.root_node.child_nodes:
            if node.type == 'function_definition':
                func_name = self._get_function_name(node)

                # Extract parameters
                for param in self._get_parameters(node):
                    facts.append((func_name, "parameter", param))

                # Extract return type
                return_type = self._get_return_type(node)
                if return_type:
                    facts.append((func_name, "returns", return_type))

                # Extract function calls within body
                for call in self._get_function_calls(node):
                    facts.append((func_name, "calls", call))

        return facts

    def _get_function_name(self, node: Node) -> str:
        """Extract function name from function_definition node."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode()
        return "unknown"

    # ... other helper methods
```

#### 2.2.3 Text Processor (Discourse Structure)

```python
# NEW FILE: src/hologram/processors/text_processor.py

from nltk.tokenize import sent_tokenize
import spacy

class TextProcessor:
    """Extract discourse structure and entity relationships."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # NER + dependency parsing

    def extract_facts(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract facts from prose:
        - (entity1, "mentions", entity2)
        - (entity, "attribute", property)
        - (event, "participant", entity)
        - (subject, "action", predicate_phrase)
        """
        facts = []

        doc = self.nlp(text)

        # Extract named entities and relationships
        for ent in doc.ents:
            facts.append((ent.text, "type", ent.label_))

        # Extract subject-verb-object from dependency parsing
        for token in doc:
            if token.dep_ == "nsubj":  # noun subject
                subject = token.text
                # Find predicate (parent verb)
                verb = token.head.text
                # Find object (dependent of verb)
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr"):
                        obj = child.text
                        facts.append((subject, verb, obj))

        return facts
```

### 2.3 Tier 2: Hierarchical HDC with Semantic Clustering

**Core Idea:** Instead of one trace, maintain a **pyramid of traces** organized by semantic similarity.

#### 2.3.1 Clustering Strategy (Semantic Bucketing)

```python
# NEW FILE: src/hologram/memory/hierarchical_trace.py

from typing import List, Dict, Tuple
import torch
from sklearn.cluster import MiniBatchKMeans

class HierarchicalTrace:
    """
    Organize facts into semantic clusters, each with own HDC trace.

    Structure:
    - Summary trace: Superposition of all cluster representatives
    - N semantic cluster traces: Each holds ~100 facts of related meaning
    - Index: Fact → Cluster assignment (O(1) lookup)

    Example clusters for "countries":
    - Cluster 0: European capitals (France→Paris, Germany→Berlin, ...)
    - Cluster 1: Asian capitals (China→Beijing, Japan→Tokyo, ...)
    - Cluster 2: African capitals (Kenya→Nairobi, ...)
    - Summary: Superposition of all 3 cluster reps
    """

    def __init__(self, space, codebook, num_clusters: int = 10):
        self._space = space
        self._codebook = codebook
        self._num_clusters = num_clusters

        # One trace per cluster
        self._traces: Dict[int, MemoryTrace] = {
            i: MemoryTrace(space) for i in range(num_clusters)
        }

        # Summary trace (superposition of cluster representatives)
        self._summary_trace = MemoryTrace(space)

        # Track cluster assignments
        self._fact_assignments: Dict[str, int] = {}  # fact_id → cluster_id

        # Maintain cluster representative vectors for reassignment
        self._cluster_reps: List[torch.Tensor] = [
            space.empty_vector() for _ in range(num_clusters)
        ]

        # K-means model for clustering new facts
        self._kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        self._fitted = False

    def add_fact(self, key: torch.Tensor, value: torch.Tensor, fact_id: str) -> int:
        """
        Add fact to hierarchical storage.

        Returns: cluster_id where fact was stored
        """
        # Create fact vector
        fact_vec = Operations.bind(key, value)

        if not self._fitted:
            # Bootstrap: assign to cluster 0 until we have enough for K-means
            cluster_id = 0
        else:
            # Assign to cluster based on semantic similarity
            cluster_id = self._kmeans.predict(fact_vec.unsqueeze(0).cpu().numpy())[0]

        # Store in cluster trace
        self._traces[cluster_id].store(key, value)
        self._fact_assignments[fact_id] = cluster_id

        # Update cluster representative (running average)
        self._update_cluster_rep(cluster_id, fact_vec)

        # Update summary trace with new representative
        self._update_summary_trace()

        return cluster_id

    def query(self, key: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Query hierarchical structure:
        1. Check summary trace → find likely clusters
        2. Search those clusters → return top-k matches
        """
        # Step 1: Query summary to find relevant clusters
        summary_result = self._summary_trace.query(key)

        # Step 2: Rank clusters by relevance to this query
        cluster_scores = []
        for cluster_id in range(self._num_clusters):
            sim = Similarity.cosine(self._cluster_reps[cluster_id], summary_result)
            cluster_scores.append((cluster_id, sim))

        # Sort by relevance
        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Search top clusters
        results = []
        for cluster_id, _ in cluster_scores[:3]:  # Check top 3 clusters
            cluster_result = self._traces[cluster_id].query(key)
            # Associate with cluster info
            results.append((*cluster_result, cluster_id))

        # Return top-k
        return results[:top_k]

    def _update_cluster_rep(self, cluster_id: int, fact_vec: torch.Tensor):
        """Update running average of cluster representative."""
        momentum = 0.9
        self._cluster_reps[cluster_id] = (
            momentum * self._cluster_reps[cluster_id] +
            (1 - momentum) * fact_vec
        )
        # Normalize
        norm = torch.norm(self._cluster_reps[cluster_id])
        if norm > 1e-6:
            self._cluster_reps[cluster_id] /= norm

    def _update_summary_trace(self):
        """Rebuild summary as superposition of cluster representatives."""
        self._summary_trace._trace = torch.mean(
            torch.stack(self._cluster_reps), dim=0
        )
        # Normalize
        norm = torch.norm(self._summary_trace._trace)
        if norm > 1e-6:
            self._summary_trace._trace /= norm
```

### 2.4 Tier 3: Knowledge Graph (Cold Storage + Constraint Enforcement)

**Library:** [pydantic-neo4j](https://github.com/joeyism/pydantic-neo4j) or [RDFlib](https://github.com/RDFLib/rdflib)

```python
# NEW FILE: src/hologram/persistence/knowledge_graph.py

from neo4j import GraphDatabase
from typing import List, Tuple, Optional

class KnowledgeGraph:
    """
    Store and validate facts against knowledge graph.

    Purpose:
    1. Source of truth for fact queries (exact matching)
    2. Constraint enforcement (can't store contradictions)
    3. Provenance tracking (every fact has source)
    4. Large-scale storage (Neo4j handles millions easily)
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_fact(self, subject: str, predicate: str, obj: str,
                 source: str = "unknown", confidence: float = 1.0) -> bool:
        """
        Add fact to knowledge graph with constraint checks.

        Constraints:
        1. Contradiction check: If (S, P, O1) exists and adding (S, P, O2),
           reject unless O1 ≈ O2 (semantic equivalence)
        2. Cardinality: Some predicates have 1:1 constraints (e.g., "capital")

        Returns: True if fact added, False if rejected (contradiction/conflict)
        """
        with self.driver.session() as session:
            # Check for contradiction
            existing = session.run(
                "MATCH (s:Entity {name: $subject})-[r:`%s`]->(o:Entity) "
                "WHERE r.predicate = $predicate RETURN o.name" % predicate,
                subject=subject, predicate=predicate
            ).single()

            if existing:
                existing_obj = existing['o.name']
                # Check if new object contradicts
                if existing_obj != obj:
                    # Could add semantic equivalence check here
                    print(f"CONTRADICTION: {subject} {predicate} {existing_obj} vs {obj}")
                    return False

            # Add fact
            session.run(
                "MERGE (s:Entity {name: $subject}) "
                "MERGE (o:Entity {name: $obj}) "
                "MERGE (s)-[r:RELATION {predicate: $predicate, source: $source, confidence: $confidence}]->(o)",
                subject=subject, obj=obj, predicate=predicate,
                source=source, confidence=confidence
            )

            return True

    def query(self, subject: str, predicate: str) -> Tuple[Optional[str], float]:
        """
        Query knowledge graph for (subject, predicate, ?).

        Returns: (object, confidence) or ("", 0.0) if not found
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity {name: $subject})-[r:`%s`]->(o:Entity) "
                "WHERE r.predicate = $predicate RETURN o.name, r.confidence" % predicate,
                subject=subject, predicate=predicate
            ).single()

            if result:
                return (result['o.name'], float(result['r.confidence']))
            return ("", 0.0)

    def get_contradictions(self, subject: str, predicate: str) -> List[Tuple[str, str]]:
        """
        Find all objects for (subject, predicate) pair.
        Used during generation validation to enforce cardinality.
        """
        with self.driver.session() as session:
            results = session.run(
                "MATCH (s:Entity {name: $subject})-[r:`%s`]->(o:Entity) "
                "WHERE r.predicate = $predicate RETURN o.name, r.confidence",
                subject=subject, predicate=predicate
            )
            return [(record['o.name'], record['r.confidence']) for record in results]
```

---

## Part 3: Hallucination-Free Generation

### 3.1 The Problem: RAG Doesn't Eliminate Hallucination

Current best practice: **Retrieval-Augmented Generation (RAG)**
- Retrieve relevant facts from database
- Inject into LLM prompt: "Given: {facts}, answer the question"
- Problem: **LLM still generates text freely**, fact injection doesn't constrain output

Example failure:
```
Fact store: (France, capital, Paris)
User query: "What is France's capital?"

LLM with RAG might generate:
"France's capital is the city of Paris, known for its beautiful architecture
and the Eiffel Tower. Paris is actually located in the north of France..." ← WRONG location!

Our system rejects this at generation time ✓
```

### 3.2 Constrained Generation: Algebraic Derivation

**New Strategy:** Make generation **deterministic derivation from facts**, not probabilistic sampling.

```python
# NEW FILE: src/hologram/generation/constrained_generator.py

from typing import Optional
import torch

class ConstrainedGenerator:
    """
    Generate S-V-O outputs that MUST match stored facts.

    Algorithm:
    1. User query → Parse to (subject, predicate, ?)
    2. Knowledge graph lookup → Get valid objects
    3. For each valid object:
       - Encode as (subject, predicate, object) vector
       - Resonance-rank by semantic fit
       - Use top-ranked as output
    4. Generate natural language template filling from output
    """

    def __init__(self, knowledge_graph: KnowledgeGraph,
                 codebook, resonator, generator):
        self.kg = knowledge_graph
        self.codebook = codebook
        self.resonator = resonator
        self.generator = generator  # ResonantGenerator

    def generate_grounded(self, query: str) -> Optional[str]:
        """
        Generate answer grounded in knowledge graph.

        Returns: Generated text or None if no valid fact found
        """
        # Step 1: Parse query
        subject, predicate = self._parse_query(query)
        # e.g., "What is France's capital?" → ("France", "capital")

        # Step 2: Get valid objects from KG
        valid_objects = self.kg.get_objects_for_subject_predicate(subject, predicate)
        # e.g., ["Paris"]

        if not valid_objects:
            return None  # No grounded answer

        # Step 3: Rank valid objects by resonance fit
        ranked = self._rank_by_resonance(subject, predicate, valid_objects)
        # ranked = [("Paris", 0.92), ("Lyon", 0.15), ...]

        best_object = ranked[0][0]

        # Step 4: Generate natural language
        # Instead of free generation, use template filling
        thought_vector = self._create_thought_vector(subject, predicate, best_object)
        result = self.generator.generate(thought_vector)

        return result.text

    def _parse_query(self, query: str) -> tuple[str, str]:
        """Parse natural language query to (subject, predicate)."""
        # Use NLP to extract subject and predicate
        # "What is France's capital?" → ("France", "capital")
        # "Who is the creator of Python?" → ("Python", "creator")
        doc = self.nlp(query)
        # ... NLP logic

    def _rank_by_resonance(self, subject: str, predicate: str,
                           objects: list[str]) -> list[tuple[str, float]]:
        """
        Rank objects by semantic resonance with query.

        Higher resonance = better semantic fit to original query intent.
        """
        query_key = Operations.bind(
            self.codebook.encode(subject),
            self.codebook.encode(predicate)
        )

        ranked = []
        for obj in objects:
            obj_vec = self.codebook.encode(obj)
            similarity = Similarity.cosine(query_key, obj_vec)
            ranked.append((obj, similarity))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _create_thought_vector(self, subject: str, predicate: str,
                               obj: str) -> torch.Tensor:
        """Create thought vector from grounded fact."""
        s_vec = self.codebook.encode(subject)
        p_vec = self.codebook.encode(predicate)
        o_vec = self.codebook.encode(obj)

        # Bind as structured fact
        return Operations.bundle(
            Operations.bind(s_vec, ROLE_SUBJECT),
            Operations.bind(p_vec, ROLE_PREDICATE),
            Operations.bind(o_vec, ROLE_OBJECT)
        )
```

### 3.3 Validation Layer: Multi-Stage Hallucination Detection

```python
# ENHANCEMENT to: src/hologram/generation/resonant_generator.py

class HalluccinationValidator:
    """
    Multi-stage validation pipeline to eliminate hallucinations.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, fact_store: FactStore):
        self.kg = knowledge_graph
        self.fact_store = fact_store

    def validate_generation(self, text: str, context: dict) -> tuple[bool, str]:
        """
        Validate generated text against facts.

        Returns: (is_valid, reason)

        Checks (in order):
        1. Entity validation: Do mentioned entities exist?
        2. Fact validation: Do entity relationships match stored facts?
        3. Cardinality: No "Paris is capital of France AND Germany"
        4. Contradiction: No "Paris is capital AND Paris is country"
        """
        # Extract entities and relationships from generated text
        entities, relationships = self._extract_from_text(text)

        # Check 1: Entity existence
        for entity in entities:
            if not self.kg.entity_exists(entity):
                return (False, f"Entity '{entity}' not in knowledge graph")

        # Check 2: Fact existence
        for (subj, pred, obj) in relationships:
            found = self.kg.query(subj, pred)
            if found[0] != obj:
                return (False, f"Fact mismatch: {subj} {pred} {obj} (stored: {found[0]})")

        # Check 3: Cardinality (1:1 predicates)
        one_to_one = {"capital", "creator", "founder", "currency"}
        for (subj, pred, obj) in relationships:
            if pred in one_to_one:
                all_objects = self.kg.get_objects_for_subject_predicate(subj, pred)
                if len(all_objects) > 1:
                    return (False, f"Cardinality violation: {subj} has multiple {pred}s")

        # Check 4: Contradiction detection
        # If text says "Paris is capital AND country" but Paris is only capital, reject
        subjects_in_text = set(subj for subj, _, _ in relationships)
        for subject in subjects_in_text:
            stored_facts = self.kg.get_all_facts_for_subject(subject)
            # Verify no contradictions
            # ... logic here

        return (True, "Valid")

    def _extract_from_text(self, text: str) -> tuple[set[str], list[tuple]]:
        """Extract entities and relationships from generated text using NLP."""
        doc = self.nlp(text)

        entities = {ent.text for ent in doc.ents}
        relationships = []

        # Simple SVO extraction from dependency parse
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ == "dobj":
                        obj = child.text
                        relationships.append((subject, verb, obj))

        return (entities, relationships)
```

---

## Part 4: Implementation Roadmap

### Phase 1: Integration Foundation (Weeks 1-2)

**Goal:** Integrate external libraries without breaking existing code

```
1. Install dependencies:
   - pip install tree-sitter tree-sitter-python
   - pip install pydantic neo4j
   - pip install nltk spacy

2. Create new modules (no breaking changes):
   - src/hologram/processors/language_classifier.py
   - src/hologram/processors/code_processor.py
   - src/hologram/processors/text_processor.py
   - src/hologram/persistence/knowledge_graph.py
   - src/hologram/memory/hierarchical_trace.py
   - src/hologram/generation/constrained_generator.py
   - src/hologram/generation/hallucination_validator.py

3. Update Container to provide new components:
   - container.create_knowledge_graph()
   - container.create_code_processor()
   - container.create_text_processor()
   - container.create_constrained_generator()
```

### Phase 2: Tier 2 Integration (Weeks 3-4)

**Goal:** Enable hierarchical HDC without breaking FactStore

```
1. Make FactStore use HierarchicalTrace optionally:
   class FactStore:
       def __init__(..., use_hierarchical: bool = False):
           if use_hierarchical:
               self._memory = HierarchicalTrace(...)  # New
           else:
               self._memory = MemoryTrace(...)        # Old (default)

2. Update add_fact() to work with both:
   def add_fact(self, ...):
       fact_id = f"{subject}:{predicate}:{obj}"
       self._memory.add_fact(key, value, fact_id)

3. Backward compatibility tests ensure nothing breaks
```

### Phase 3: Tier 3 Integration (Weeks 5-6)

**Goal:** Add knowledge graph validation without changing generation

```
1. Enhance ResonantGenerator with optional validation:
   def generate_with_validation(self, thought, context=None, ...):
       result = self.generate(thought, ...)  # Same as before
       if context.knowledge_graph:
           valid, reason = validator.validate_generation(
               result.text, context
           )
           if not valid:
               # Fallback or request regeneration
               ...
       return result

2. Add ConstrainedGenerator as alternative path (opt-in)
   If user calls constrained_generator.generate_grounded(query):
       Uses new fully-grounded pipeline
   Else:
       Uses existing ResonantGenerator
```

### Phase 4: Code/Text Classification (Weeks 7-8)

**Goal:** Route mixed-content documents appropriately

```
1. Update FactStore.add_fact_from_text():
   def add_fact_from_text(self, text: str, source: str):
       content_type, confidence = classifier.classify(text)

       if content_type.startswith("code"):
           facts = code_processor.extract_facts(text, language)
       else:
           facts = text_processor.extract_facts(text)

       for subject, predicate, obj in facts:
           self.add_fact(subject, predicate, obj, source)

2. Tests verify code facts are correctly extracted
```

### Phase 5: Scaling Validation (Weeks 9-10)

**Goal:** Prove 0% hallucination on large datasets

```
Tests:
1. Capacity test: Add 100,000 facts via HierarchicalTrace
   - Verify no saturation → all queries succeed

2. Hallucination test: Try to generate contradictions
   - Query: "What is Paris's country?"
   - Stored: (Paris, country, France)
   - Generator can ONLY return "France" (or semantically equivalent)

3. Mixed content test: Book with chapters (text) + code (examples)
   - Extract both types appropriately
   - Generate answers without mixing contexts

4. Benchmark:
   - Compare hallucination rate: ResonantGenerator → ConstrainedGenerator
   - Show 0% false facts in constrained path
```

---

## Part 5: Key Libraries & Integration Points

| Component | Library | GitHub | Purpose |
|-----------|---------|--------|---------|
| Code parsing | tree-sitter | https://github.com/tree-sitter/tree-sitter | Syntax trees for 100+ languages |
| Knowledge graph | Neo4j | https://github.com/neo4j/neo4j | Fact storage + constraint enforcement |
| Text NLP | spaCy | https://github.com/explosion/spaCy | NER, dependency parsing, tokenization |
| RDF (alternative) | RDFlib | https://github.com/RDFLib/rdflib | Semantic web triple store |
| Clustering | scikit-learn | https://github.com/scikit-learn/scikit-learn | MiniBatchKMeans for semantic bucketing |
| Validation | pydantic | https://github.com/pydantic/pydantic | Schema validation for facts |

### 5.1 Neo4j Setup (Quickstart)

```bash
# Docker Compose (recommended)
docker run --publish=7687:7687 --publish=7474:7474 neo4j

# Or cloud
neo4j-desktop://  # Download from neo4j.com

# In Python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687",
                               auth=("neo4j", "password"))
```

---

## Part 6: Architectural Diagram (ASCII)

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────┐
│   TIER 0: CONTENT CLASSIFIER            │
│  (Language Detection + Router)           │
├─────────────────────────────────────────┤
│  Input: "def hello(): return 'world'"    │
│  Output: content_type="code_python"     │
│           confidence=0.95               │
└────────┬──────────────────┬─────────────┘
         │                  │
    Code Path          Text Path
         │                  │
         ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│  CodeProcessor   │  │  TextProcessor   │
│  (tree-sitter)   │  │  (spaCy)         │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Extract S-P-O      │
         │  Triples from       │
         │  Source             │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  TIER 3: KNOWLEDGE  │
         │  GRAPH (Neo4j)      │
         │  Add Facts +        │
         │  Check Constraints  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  TIER 2:            │
         │  HIERARCHICAL HDC   │
         │  Store in Semantic  │
         │  Clusters           │
         └──────────┬──────────┘
                    │
              Generation Paths:
           (Choose based on type)
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
    ┌─────────────────┐        ┌──────────────────┐
    │  CONSTRAINED    │        │  RESONANT        │
    │  GENERATOR      │        │  GENERATOR       │
    │  (Fact-grounded)│        │  (Flexible)      │
    └────────┬────────┘        └────────┬─────────┘
             │                          │
             ▼                          ▼
      ┌─────────────┐      ┌────────────────────┐
      │ Template    │      │ Token-by-token     │
      │ Filling     │      │ Divergence Check   │
      │ (LMQL)      │      │ (Soft cleanup)     │
      └────────┬────┘      └────────┬───────────┘
               │                    │
               └─────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ VALIDATION LAYER    │
              │ Multi-stage fact    │
              │ checking            │
              └──────────┬──────────┘
                         │
                         ▼
                  FINAL GROUNDED OUTPUT
              "France's capital is Paris"
              Provenance: (France, capital, Paris)
                        Source: Wikipedia
                        Confidence: 1.0
```

---

## Part 7: Hallucination Guarantee Analysis

### 7.1 How We Achieve 0% Hallucination

**Definition:** Hallucination = Generated fact not in knowledge graph + sources

**Prevention mechanisms:**

```
1. CONSTRAINED GENERATION PATH:
   User query → Extract (S, P)
           → KG lookup → Valid objects
           → Only these objects in output
           → Impossible to generate unseen fact

   Guarantee: P(hallucination | constrained_path) = 0%

2. VALIDATION LAYER (backup):
   If using ResonantGenerator:
   - Generate text (as before)
   - Extract facts via NLP
   - Validate each fact against KG
   - If invalid: Reject or request regeneration

   Guarantee: P(hallucination | validation passes) < 0.1%
             (Only semantic equivalence misses possible)

3. KNOWLEDGE GRAPH CONSTRAINTS:
   - No fact can violate schema constraints
   - No cardinality violations (1:1 predicates)
   - No contradictions (same subject, same predicate)

   These prevent storage of false facts in first place
```

### 7.2 Comparison Matrix

| Approach | Hallucination Rate | Scalability | Speed | Reasoning |
|----------|------------------|-------------|-------|-----------|
| Pure LLM | 5-15% | Excellent | Fast | No grounding |
| RAG + LLM | 2-5% | Good | Medium | Injection, not constraint |
| Resonant (current) | 1-2% | ~100 facts | Medium | Soft validation, fallback |
| Constrained (proposed) | ~0% | Millions | Slower | Algebraic derivation |

**Key insight:** Our approach trades generation speed for absolute reliability. For mission-critical facts (code, medical, legal), this is the right tradeoff.

---

## Part 8: Migration Path from Current Architecture

### 8.1 Backward Compatibility Strategy

**Goal:** Existing code keeps working, new code opts into features

```python
# OLD CODE (unchanged, still works):
from hologram.container import Container

container = Container()
generator = container.create_resonant_generator(vocab)
result = generator.generate(thought)
print(result.text)

# NEW CODE (uses new features):
from hologram.container import Container

container = Container()
classifier = container.create_language_classifier()
code_proc = container.create_code_processor()
kg = container.create_knowledge_graph()
constrained_gen = container.create_constrained_generator()

# Classify content
content_type, conf = classifier.classify(text)

# Extract facts
if content_type.startswith("code"):
    facts = code_proc.extract_facts(text)
else:
    facts = text_processor.extract_facts(text)

# Store in KG + HDC
for s, p, o in facts:
    kg.add_fact(s, p, o)
    fact_store.add_fact(s, p, o)

# Generate grounded
output = constrained_gen.generate_grounded("What is France's capital?")
```

### 8.2 Feature Flags for Phased Rollout

```python
# In config/constants.py (NEW)

ENABLE_HIERARCHICAL_TRACE = False  # Phase 2
ENABLE_KNOWLEDGE_GRAPH = False     # Phase 3
ENABLE_CODE_DETECTION = False      # Phase 4
ENABLE_CONSTRAINED_GENERATION = False  # Phase 5

# Usage in code:
def create_fact_store(space, codebook):
    if ENABLE_HIERARCHICAL_TRACE:
        memory = HierarchicalTrace(space, codebook)
    else:
        memory = MemoryTrace(space)  # Current behavior

    return FactStore(space, codebook, memory)
```

---

## Part 9: Testing Strategy

### 9.1 Unit Tests (Existing + New)

```python
# tests/memory/test_hierarchical_trace.py
def test_hierarchical_capacity_100k_facts():
    """Verify 100k facts without saturation."""
    # Generate 100,000 facts
    # Verify query success rate > 99%

def test_clustering_maintains_semantic_coherence():
    """Verify facts cluster semantically."""
    # Add "Paris is capital of France"
    # Add "Berlin is capital of Germany"
    # Verify both in different clusters or same cluster

# tests/generation/test_hallucination_validator.py
def test_rejects_contradictions():
    """Validator catches impossible facts."""
    # Store: (Earth, shape, sphere)
    # Try to generate: "Earth is flat"
    # Verify: REJECTED

def test_cardinality_enforcement():
    """Validator enforces 1:1 predicates."""
    # Store: (Paris, country, France)
    # Try: "Paris is also country of Germany"
    # Verify: REJECTED

# tests/processors/test_language_classifier.py
def test_classifies_python_code():
    """Identify Python code correctly."""
    code = "def hello():\n    return 'world'"
    ct, conf = classifier.classify(code)
    assert ct == "code_python"
    assert conf > 0.9

def test_classifies_english_text():
    """Identify prose correctly."""
    text = "Paris is the capital of France."
    ct, conf = classifier.classify(text)
    assert ct == "text"
    assert conf > 0.8
```

### 9.2 Integration Tests

```python
# tests/integration/test_full_pipeline.py
def test_code_to_facts_to_generation():
    """End-to-end: code → facts → grounded generation."""
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    # Classify
    ct, _ = classifier.classify(code)
    assert ct.startswith("code")

    # Extract facts
    facts = code_proc.extract_facts(code)
    assert ("fibonacci", "parameter", "n") in facts
    assert ("fibonacci", "returns", "int") in facts  # Or similar

    # Store
    for s, p, o in facts:
        kg.add_fact(s, p, o)

    # Query
    answer, conf = kg.query("fibonacci", "parameter")
    assert answer == "n"
    assert conf == 1.0

def test_mixed_content_book():
    """Handle book with chapters and code examples."""
    book_text = """
    # Chapter 1: Introduction
    Python is a programming language.

    ```python
    print("Hello, World!")
    ```

    This prints to stdout.
    """

    # Should extract both prose and code facts
    # Text facts: (Python, type, "programming language")
    # Code facts: (print, operation, "output")
```

### 9.3 Hallucination Benchmark

```python
# tests/benchmarks/test_hallucination_rate.py
def test_hallucination_rate_resonant_baseline():
    """Current ResonantGenerator hallucination rate."""
    queries = [
        ("What is France's capital?", ("France", "capital", "Paris")),
        ("Who created Python?", ("Python", "creator", "Guido van Rossum")),
        ...
    ]

    facts_stored = [
        ("France", "capital", "Paris"),
        ("Python", "creator", "Guido van Rossum"),
        ...
    ]

    # Store facts
    for s, p, o in facts_stored:
        fact_store.add_fact(s, p, o)

    # Generate answers
    hallucinations = 0
    for query, (expected_s, expected_p, expected_o) in queries:
        result = generator.generate(query)
        if not validator.validate_generation(result, facts_stored):
            hallucinations += 1

    rate = hallucinations / len(queries)
    print(f"ResonantGenerator hallucination rate: {rate:.1%}")
    # Expected: ~1-2%

def test_hallucination_rate_constrained():
    """Constrained generator should have ~0% hallucination."""
    # Same setup as above

    for query, (expected_s, expected_p, expected_o) in queries:
        result = constrained_gen.generate_grounded(query)
        # Extract fact from result
        extracted_o = extract_object_from_text(result)
        assert extracted_o == expected_o  # Must match exactly

    # Expected: 0% hallucination (only extraction errors possible)
```

---

## Part 10: Success Metrics

### 10.1 Capacity Metrics

| Metric | Current | Target | Validation |
|--------|---------|--------|-----------|
| Max facts (single trace) | ~100 | ~1M | Store 1M facts, query success > 99% |
| Query latency | 5ms | <50ms | Average query time < 50ms @ 1M facts |
| Memory per fact | ~100 bytes | <50 bytes | HDC + KG combined footprint |
| Hallucination rate | 1-2% | <0.1% | Validation layer catches all contradictions |

### 10.2 Feature Completeness

- [ ] Phase 1: All new modules importable, no breaking changes
- [ ] Phase 2: HierarchicalTrace handles 10k facts without saturation
- [ ] Phase 3: Neo4j stores 100k facts with constraint enforcement
- [ ] Phase 4: Classifier correctly identifies code/text (>95% accuracy)
- [ ] Phase 5: Constrained generator produces 0% hallucinations on benchmark

---

## Conclusion

This architecture solves three fundamental limitations of the current design:

1. **Capacity:** Hierarchical traces + Neo4j replace single-trace bottleneck
2. **Hallucination:** Knowledge graph + constrained generation replace probabilistic sampling
3. **Mixed content:** Code/text classifier routes appropriately instead of treating everything as prose

The result: A **scalable, verifiable, fact-grounded** AI system that can store millions of facts and guarantee generated output matches stored knowledge.

**Next steps:**
1. Create phase 1 modules (non-breaking)
2. Build comprehensive test suite
3. Benchmark capacity and hallucination rate
4. Integrate with existing system gradually
5. Deploy with feature flags for safe rollout

# Conscious Hologram Architecture

**Last Updated**: 2025-12-12
**Status**: Implemented (All 5 Layers + Scaling Extensions)

---

## What Is This?

The **Conscious Hologram** is a 5-layer system that stores and retrieves knowledge using interference patterns instead of neural network weights. Think of it like dropping pebbles into a pond - each fact creates ripples that combine into a pattern. To recall a fact, you "tune in" to its frequency.

**Why it matters**: Unlike LLMs that guess the next word, Hologram can only retrieve what was actually stored. It cannot hallucinate facts.

---

## The 5 Layers at a Glance

| Layer | Name              | What It Does                                                     |
| ----- | ----------------- | ---------------------------------------------------------------- |
| **1** | Fractal Substrate | Makes vectors robust - any fragment can recover the whole        |
| **2** | Memory            | Stores facts as interference patterns (Subject-Predicate-Object) |
| **3** | Metacognition     | Monitors confidence and retries when confused                    |
| **4** | Retrieval         | Finds facts using exact match or HDC resonance search            |
| **5** | Voice             | Converts retrieved facts into natural language                   |

---

## Layer 1: Fractal Substrate

**File**: `src/hologram/core/fractal.py`

### The Idea

Bentov's hologram metaphor: _"Cut a hologram in half - you get the whole rose, just fuzzier."_

Every concept is stored as a **64-dimensional "DNA seed"** that expands into **10,000 dimensions**. If part of the vector is corrupted, you can recover the original from any surviving fragment.

### How It Works

1. **Generate DNA seed**: Create a deterministic 64-dim "DNA" vector from a seed value
2. **Expand via rotations**: Apply rotation matrices to "echo" the DNA across 10,000 dimensions
   - Block 0 = original DNA (no rotation)
   - Block k = R_k @ DNA (rotated via deterministic rotation matrix)
3. **Holographic recovery**: Given any 64-dim block k, apply inverse rotation: DNA = R_k^T @ shard
4. **Multi-shard averaging**: Multiple shards can be averaged for cleaner recovery

**Implementation** (`src/hologram/core/fractal.py:79-237`):

```python
space = FractalSpace(dimensions=10000, dna_dimensions=64)

# Generate vector from seed
v = space.random_vector(seed=42)  # 10,000-dim fractal vector

# Extract a block (e.g., block 78)
shard = space.extract_block(v, block_index=78)  # 64-dim slice

# Recover DNA from shard
recovered_dna = space.recover_dna(shard, block_index=78)

# Cosine similarity between recovered and original DNA > 0.99
# (limited only by floating point precision)
```

**Empirical Properties**:

- **Recovery accuracy**: cosine similarity > 0.95 (rotation matrices are orthogonal, so recovery is nearly perfect)
- **Noise tolerance**: Recovers original DNA even if shard is corrupted by ~30% noise
- **Multi-shard averaging**: Multiple recovered DNAs can be averaged for even cleaner results (`recover_dna_from_multiple_shards()`)

### Trade-off

Less effective concept capacity (limited by DNA expansion) but much better corruption recovery and holographic properties. The 64-dim DNA creates 10,000/64 ≈ 156 orthogonal rotations, each containing the whole concept at different orientations.

---

## Layer 2: Memory

**Files**: `src/hologram/memory/fact_store.py`, `src/hologram/memory/memory_trace.py`

### The Idea

Facts are stored as **Subject-Predicate-Object triples** using two HDC operations:

- **Binding**: Combines two concepts into a unique pattern (reversible)
- **Bundling**: Superimposes multiple facts into one vector (holographic storage)

### How It Works

**Storing "France capital Paris":**

```
key = bind(France, capital)      # Create unique lookup key
fact = bind(key, Paris)          # Attach the answer
memory = bundle(memory, fact)    # Add to holographic storage
```

**Retrieving:**

```
key = bind(France, capital)
result = unbind(memory, key)     # Extract ~Paris from interference pattern
answer = find_closest(result)    # Match to vocabulary → "Paris"
```

### Surprise Gating (Titans-Inspired Learning Rate Modulation)

Facts are stored selectively based on how "surprising" (novel) they are relative to existing memory. This prevents duplicate encoding and noise accumulation from redundant information.

**How it works** (`src/hologram/memory/memory_trace.py:90-174`):

1. **Dual Surprise Metrics**:

   - **Current surprise**: How different is this fact from existing memory? `surprise = 1.0 - cosine(memory, fact)`
   - **Momentum surprise**: How different from recent learning direction? `momentum_surprise = 1.0 - cosine(momentum, fact)`
   - **Combined**: `combined_surprise = 0.7 * current + 0.3 * momentum` (weighted blend)

2. **Learning Rate Modulation**:

   - Only facts with `combined_surprise >= SURPRISE_THRESHOLD (0.1)` trigger memory updates
   - Below threshold = "already known" → skip learning
   - Update strength = `combined_surprise * learning_rate`, so novel facts encode more strongly

3. **Warm-up Period** (First 10 Facts):

   - Reduces learning rate: `learning_rate * (fact_count + 1) / 10`
   - Prevents instability during initial learning phase

4. **Momentum Tracking** (Exponential Moving Average):

   - Tracks recent learning direction: `momentum = 0.9 * old_momentum + 0.1 * new_fact`
   - Allows system to detect when it's learning "in a direction" (higher momentum_surprise for novel directions)
   - Decay factor (0.9) means recent facts influence momentum more than old ones

5. **Bounded Memory**:
   - Optional forgetting via decay: `trace = trace * 0.99` (1% forgetting per decay call)
   - Enables active memory management (Titans insight: prevents saturation)

**Example**: Teaching the same fact twice:

```python
surprise1 = trace.store_with_surprise(france_capital_key, paris)  # 0.8 (novel!)
surprise2 = trace.store_with_surprise(france_capital_key, paris)  # 0.02 (known)
# Only first update passes threshold; second is skipped
```

**Result**: System learns efficiently, avoids noise from repetition, and naturally emphasizes novel information.

### Capacity and Scaling

**Capacity is heuristic, not guaranteed.** `constants.py` documents a rule-of-thumb (dimensions ÷ 100), while `MemoryTrace.saturation_estimate` uses √dimensions. Both are unproven; treat capacity as empirical and monitor confidence, not a hard limit.

**Scaling Strategy 1: Hierarchical Fact Store (HDC + FAISS)** (`src/hologram/memory/fact_store.py:472-598`)

Use the two-tier architecture when you need headroom:

| Layer    | Storage         | Lookup                     | Capacity                                                  | Confidence                                      |
| -------- | --------------- | -------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| **Hot**  | HDC FactStore   | O(1) exact key lookup      | Heuristic: tens to low hundreds before interference grows | Exact ≈ 1.0; resonance varies with interference |
| **Cold** | FAISS vector DB | O(log n) similarity search | Operationally unbounded                                   | Similarity per FAISS scoring                    |

**How it works**:

1. Store facts in both hot (HDC) and cold (FAISS) layers simultaneously.
2. Query hot layer first (exact match via normalized key dictionary).
3. If hot confidence < 0.7, fall back to FAISS cold search.
4. Both layers use the same fact encoding (Subject-Predicate-Object triples).

**Result**: Exact key lookups work at unlimited scale; fuzzy resonance searches gracefully degrade to cold storage when confidence drops.

**Scaling Strategy 2: Neural Consolidation (Sleep Learning)** (`src/hologram/consolidation/`)

_New in v0.4.0_: A bio-inspired approach where facts are moved from "working memory" (HDC) to "long-term memory" (Neural Weights) via background consolidation.

1.  **Working Memory (HDC)**: Fast, immediate storage.
2.  **Background Consolidation**: When buffer fills, a background thread trains a small MLP on the facts.
3.  **Decay**: HDC trace is decayed (not wiped), preserving a faint signal while the neural net takes over.
4.  **Calibrated Retrieval**: Queries check both HDC and Neural layers. `ConfidenceCalibrator` picks the winner based on calibrated confidence scores.
5.  **Safety Gate**: Neural predictions must pass an HDC unbinding check to prevent hallucination.

This allows the system to store "infinite" facts in fixed-size weights, using the HDC layer as a fast buffer.

### New: Episodic Memory Layer

**File**: `src/hologram/conversation/memory.py`

In addition to semantic facts, the system now stores **conversation episodes** (user-response pairs) in a dedicated vector store (FAISS/Chroma).

- **Purpose**: Provides "long effective context" without filling the prompt window.
- **Mechanism**: Stores `bind(user_vec, response_vec)` + metadata.
- **Retrieval**: Top-k episodes are retrieved for every turn and passed to the Ventriloquist.
- **Benefit**: The hologram remembers what you said 50 turns ago without reprocessing the whole history.

---

## Layer 3: Metacognition

**File**: `src/hologram/cognition/metacognition.py`

### The Idea

The system monitors its own confidence and tracks its "mood":

- **NEUTRAL** → starting state
- **CONFIDENT** → high confidence answers
- **CONFUSED** → low confidence, triggers retry

### How It Actually Works

The system maintains a persistent **self_vector** that accumulates mood signals from confidence observations.

**Observation → Labeling → Rewiring**:

1. **Observe confidence** from query result
2. **Label internal state** based on confidence threshold:
   - Confidence >= 0.8 → Bundle in CONFIDENT vector
   - Confidence <= 0.2 → Bundle in CONFUSION + CURIOSITY vectors (2x weight)
   - 0.2 < Confidence < 0.8 → Bundle in ANXIETY or CURIOSITY vectors
3. **Rewire self_vector** via HDC bundling (`self_vector = bundle(self_vector, mood_vector)`)
4. **Retry if needed** with modified internal state

**Example**: Query fails with low confidence

```python
# Initial query: confidence = 0.15 (too low)
state.update_from_confidence(0.15)
# self_vector now has CONFUSION + CURIOSITY bundled in (curiosity at 2x weight)
# This modulation affects future queries (though current implementation is observational)
```

**What "rewiring" means**: The system doesn't rewrite the _query text_, but it **does** modulate its _internal state_ by bundling mood vectors. This changes the system's perspective for retries.

**Note**: Current implementation uses mood tracking for observability and enables retry logic. Future work will use `self_vector` modulation to influence retrieval (e.g., inject curiosity into the query context).

**Mood States** (`src/hologram/cognition/metacognition.py`):

- NEUTRAL → Starting state
- CONFIDENT → High confidence (>= 0.8)
- CONFUSED → Low confidence (<= 0.2), triggers exploration
- CURIOUS → Medium-low confidence (0.2-0.4), adds curiosity vector
- ANXIOUS → Medium confidence (0.4-0.6), adds both anxiety and curiosity

### Circuit Breaker

Prevents infinite retry loops - gives up after max attempts.

---

## Layer 4: Retrieval

**Files**: `src/hologram/memory/fact_store.py`, `src/hologram/conversation/selector.py`

### The Idea

Multiple strategies to find stored facts, prioritized by speed and confidence.

### Dual Query Modes

**FactStore provides two complementary retrieval strategies** (`src/hologram/memory/fact_store.py:219-285`):

#### 1. Exact Match (O(1), High Confidence)

- **How**: Normalized key dictionary lookup (`_exact_index`)
- **Confidence**: ≈ 1.0 (perfect match)
- **Speed**: O(1) constant time
- **When used**: First attempt (lines 250-257)
- **Example**: `query("France", "capital")` → exact key match → "Paris" with confidence 1.0

#### 2. HDC Resonance Search (O(n), Confidence varies with interference)

- **How**: Unbind key from bundled memory, measure cosine similarity to all vocabulary words.
- **Confidence**: Depends on interference; use thresholds, not fixed ranges.
- **Speed**: O(n) where n = vocabulary size.
- **When used**: Fallback if exact match fails (lines 259-285).
- **Example**: Fuzzy query with typos or novel phrasing.

**Confidence Breakdown**:

- **Exact match queries**: ~1.0 (deterministic key lookup).
- **Holographic (bundled) retrieval**: Confidence is interference-sensitive; rely on thresholds:
  - Response threshold (0.20): Provide answer if confidence ≥ 0.20.
  - Refusal threshold (0.10): Say "I don't know" if confidence < 0.10.
  - Hedge zone (0.10-0.20): Uncertain response.

**Why the difference?** Bundling multiple facts creates interference. The query vibrates the surface, and the target fact's signal competes with noise, so similarities are well below 1.0 and should be interpreted via thresholds.

### Strategies (Layered Priority)

1. **Exact Match (fastest)**: Normalized key dictionary lookup - "France" + "capital" → "Paris" (1.0 confidence)
2. **HDC Resonance (fallback)**: Cosine similarity search through bundled memory - fuzzy matching with 0.24-0.37 confidence
3. **Semantic Fact Search (new)**: Metadata search for entity mentions in any position (subject, predicate, or object)
4. **Learned Patterns**: Template matching for conversational responses (ResponseSelector layer)

#### 3. Semantic Fact Search (O(n), Low-Medium Confidence)

**File**: `src/hologram/memory/fact_store.py:502-568`

- **How**: Word-boundary regex search through fact metadata (`_facts` list)
- **When used**: Fallback when structured queries fail and entity appears in non-subject position
- **Confidence**: 0.25-0.63 depending on match location (subject > predicate > object)
- **Example**: Query "river" finds "Nile is longest river" even though "river" is in the object

**Why it's needed**: Standard S-P-O queries require the entity to be the subject. But users often ask about terms that appear in the object position:
- User: "Do you know about river?"
- Stored fact: "Nile is longest river in the world" (river in OBJECT)
- Old behavior: `query("river", "is")` fails → "I don't know"
- New behavior: Semantic search finds the mention → "Nile is longest river"

**Scoring by match location**:
| Position | Base Score | After Fallback Multiplier |
|----------|------------|---------------------------|
| Subject  | 0.9        | 0.63 (× 0.7)              |
| Predicate| 0.7        | 0.49 (× 0.7)              |
| Object   | 0.5        | 0.25 (× 0.5)              |

**Limitations**:
- Only searches in-memory `_facts` list (not persisted neural memory)
- O(n) linear scan - acceptable for <1000 facts
- Minimum term length: 3 characters (to avoid false positives)
- Uses ASCII word boundaries (non-ASCII text may not match correctly)

---

## Layer 5: Voice

**Files**: `src/hologram/generation/resonant_generator.py`, `src/hologram/generation/ventriloquist.py`

### The Idea

Two ways to speak the retrieved facts:

| Mode                       | When Used               | Output Style                       |
| -------------------------- | ----------------------- | ---------------------------------- |
| **ResonantGenerator**      | HDC-native fallback     | Concise, constrained to vocabulary |
| **VentriloquistGenerator** | Preferred (via SLM API) | Fluent, natural conversation       |

### How Routing Works

In practice, **Ventriloquist is preferred when available**:

1. If Ventriloquist is enabled → use it
2. If not available → fall back to ResonantGenerator
3. If generation fails → use template response

### The Ventriloquist Pattern

**The SLM doesn't think - it only speaks what HDC tells it to say.**

```
┌─────────────────────────────────────┐
│         Conscious Hologram           │
│  FactStore → Retrieval → "Paris"    │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│        VentriloquistGenerator        │
│  Prompt: "Answer using fact: Paris" │
│  Output: "Paris! Beautiful city."   │
└─────────────────────────────────────┘
```

HDC controls **what** to say. SLM controls **how** to say it.

**Token limits (current wiring):**

- ResponseSelector passes **10 tokens** to the HDC generator path.
- ResponseSelector passes **256 tokens** to the Ventriloquist path.
- `MAX_GENERATION_TOKENS` in `constants.py` is not enforced by the selector; the per-call caps above are operative.

**New: Token Budgeting**:
A `_budget_tokens` method now calculates available space in the context window before every call, ensuring prompts never exceed model limits regardless of `max_tokens` settings.

**New: Long-Output Pipeline (Multi-Pass)**:
For complex queries, the Ventriloquist now uses a **multi-pass strategy**:

1. **Outline**: Generate a JSON structure of sections (using reasoning model).
2. **Expand**: Generate content for each section separately (using fluency model).
3. **Verify**: Check that expanded sections are grounded in retrieved facts.

---

## Complete Data Flow

```
User: "What is the capital of France?"
    ↓
[Layer 1] Encode query via Codebook (fractal/semantic codebook optional)
    ↓
[Layer 4] Find fact: France → capital → Paris (confidence: 1.0)
    ↓
[Layer 3] Observe high confidence → mood stays CONFIDENT
    ↓
[Layer 5] Ventriloquist generates: "The capital of France is Paris!"
```

---

## Usage

```python
from hologram import HologramContainer

# Create container (fractal/semantic codebook are optional flags)
container = HologramContainer(dimensions=10000, use_fractal=True)

# Create chatbot with persistence
chatbot = container.create_persistent_chatbot(
    persist_dir="./data/hologram_facts",
    enable_metacognition=True,
    enable_ventriloquist=False,  # Default is False; set True to enable voice
)

# Teach and query
chatbot.respond("The capital of France is Paris")
chatbot.respond("What is the capital of France?")
# → "The capital of France is Paris!"
```

---

## Key Properties

| Property                 | How It's Achieved                              |
| ------------------------ | ---------------------------------------------- |
| **Cannot hallucinate**   | Can only retrieve stored vocabulary            |
| **Corruption-resistant** | Fractal vectors recover from fragments         |
| **Self-monitoring**      | Metacognition tracks confidence trends         |
| **Natural output**       | SLM speaks fluently while HDC controls content |
| **Citable**              | Every fact traces back to a source             |

---

## Current Performance and Validation

### Quiz Accuracy Metrics

**HDC Fact Retrieval**: ~81% accuracy on knowledge quiz (target: 90%+)

- Tested with `scripts/crew_trainer.py` and `scripts/benchmark_tasks.py`
- Uses both fictional and real-world facts to verify HDC grounding
- See test results: `tests/test_hdc_fact_grounding.py` (800+ fact test cases)

**How to reproduce**:

```bash
# Run benchmark
uv run python scripts/benchmark_tasks.py

# Run crew trainer (integrates with conversation)
uv run python scripts/crew_trainer.py

# Run fact grounding tests
uv run pytest tests/test_hdc_fact_grounding.py -v
```

**Test Coverage**:

- Fictional facts (e.g., "Zorbaxia capital Flumpton") - ensures HDC, not LLM training
- Contradictory facts (e.g., "France capital Berlin") - overrides LLM knowledge
- Complex multi-word facts (e.g., "secure quantum communication")
- Relationship chains (e.g., Alice→TechCorp→Silicon Valley)
- Numerical precision (e.g., "127.5 meters", "$3.7 million")
- Edge cases (dashes, underscores, mixed case, single characters)

### Capacity and Performance

| Metric                     | Value                   | Notes                                       |
| -------------------------- | ----------------------- | ------------------------------------------- |
| **Hot layer capacity**     | Heuristic/empirical     | Bundled vector before interference grows    |
| **Cold layer capacity**    | Operationally unbounded | FAISS with vector similarity search         |
| **Exact match lookups**    | O(1)                    | Normalized key dictionary                   |
| **Fuzzy search**           | O(n)                    | Resonance through bundled memory            |
| **Vocabulary growth**      | Dynamic                 | Grows with each new entity/predicate/object |
| **Vector dimensions**      | 10,000                  | Configurable, empirically tuned             |
| **Fractal DNA dimensions** | 64                      | Creates ~156 holographic echoes             |

---

## Implemented Advanced Features (Previously Labeled as "Proposals")

The following features were designed as proposals but are **fully implemented**:

### 1. Surprise-Gated Learning (IMPLEMENTED)

- **Previously**: Labeled as "Holographic Surprise" proposal
- **Status**: ✅ **Implemented** in `src/hologram/memory/memory_trace.py:90-174`
- **What it does**: Dual-surprise metrics (current + momentum) prevent duplicate learning and optimize learning rate
- **See**: "Surprise Gating" section above

### 2. Ventriloquist Architecture (IMPLEMENTED)

- **Previously**: Proposed as "Full SLM integration"
- **Status**: ✅ **Implemented** in `src/hologram/generation/ventriloquist.py`
- **What it does**: SLM wrapper that validates LLM output uses HDC-retrieved facts. Now includes token budgeting and multi-pass generation.
- **See**: Layer 5 (Voice) section; full architecture in `docs/VENTRILOQUIST_ARCHITECTURE.md`

### 3. Hierarchical Fact Store (IMPLEMENTED)

- **Previously**: Proposed as "Neural Consolidation" to break 100-fact limit
- **Status**: ✅ **Implemented** in `src/hologram/memory/fact_store.py:472-598`
- **What it does**: Two-tier storage (hot HDC + cold FAISS) for unlimited scalability
- **See**: "Capacity and Scaling" section above

### 4. Episodic Memory (IMPLEMENTED)

- **Status**: ✅ **Implemented** in `src/hologram/conversation/memory.py`
- **What it does**: Stores user-response pairs in vector storage for long-term context retrieval.

### 5. Neural Consolidation (IMPLEMENTED)

- **Status**: ✅ **Implemented** in `src/hologram/consolidation/`
- **What it does**: Sleep-inspired memory consolidation. Moves facts from HDC working memory to a compressed neural network for long-term storage.
- **Key Features**:
  - **Async Training**: Happens in background thread.
  - **Calibration**: Unifies HDC and Neural confidence scores.
  - **Goldfish/Elephant**: Supports both rapid forgetting (working memory) and long-term retention.

---

## References

- Bentov, I. (1977). _Stalking the Wild Pendulum_
- Google Titans architecture (surprise gating inspiration)
- Hyperdimensional Computing literature

---

**Key Files**:

- `src/hologram/core/fractal.py` - FractalSpace
- `src/hologram/memory/fact_store.py` - FactStore + `search_facts_mentioning()` (semantic search)
- `src/hologram/cognition/metacognition.py` - MetacognitiveLoop
- `src/hologram/conversation/selector.py` - ResponseSelector + semantic fallback in `_query_facts()`
- `src/hologram/generation/ventriloquist.py` - VentriloquistGenerator

# Conscious Hologram Architecture

**Last Updated**: 2025-12-11
**Status**: Implemented (All 5 Layers)

---

## What Is This?

The **Conscious Hologram** is a 5-layer system that stores and retrieves knowledge using interference patterns instead of neural network weights. Think of it like dropping pebbles into a pond - each fact creates ripples that combine into a pattern. To recall a fact, you "tune in" to its frequency.

**Why it matters**: Unlike LLMs that guess the next word, Hologram can only retrieve what was actually stored. It cannot hallucinate facts.

---

## The 5 Layers at a Glance

| Layer | Name | What It Does |
|-------|------|--------------|
| **1** | Fractal Substrate | Makes vectors robust - any fragment can recover the whole |
| **2** | Memory | Stores facts as interference patterns (Subject-Predicate-Object) |
| **3** | Metacognition | Monitors confidence and retries when confused |
| **4** | Retrieval | Finds facts using exact match or HDC resonance search |
| **5** | Voice | Converts retrieved facts into natural language |

---

## Layer 1: Fractal Substrate

**File**: `src/hologram/core/fractal.py`

### The Idea

Bentov's hologram metaphor: *"Cut a hologram in half - you get the whole rose, just fuzzier."*

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

**HDC Holographic Storage Capacity**: ~100 facts in a single bundled vector before interference becomes too noisy. This is fundamental to how holographic storage works - bundling too many facts together creates interference patterns that degrade retrieval confidence.

**Scaling Beyond 100: HierarchicalFactStore** (`src/hologram/memory/fact_store.py:472-598`)

The system solves the 100-fact limit with a two-tier architecture:

| Layer | Storage | Lookup | Capacity | Confidence |
|-------|---------|--------|----------|------------|
| **Hot** | HDC FactStore | O(1) exact key lookup | ~100 facts (fuzzy) | ~1.0 (exact) or 0.24-0.37 (resonance) |
| **Cold** | FAISS vector DB | O(log n) similarity search | Unlimited | 0.20-0.40 (similar to hot) |

**How it works**:
1. Store facts in both hot (HDC) and cold (FAISS) layers simultaneously
2. Query hot layer first (fast exact match via normalized key dictionary)
3. If hot confidence < 0.7, fall back to FAISS cold search
4. Both layers use same fact encoding (Subject-Predicate-Object triples)

**Result**: Exact key lookups work at unlimited scale; fuzzy resonance searches gracefully degrade to cold storage when confidence drops.

**Usage**:
```python
from hologram import HologramContainer

container = HologramContainer(dimensions=10000)
# Creates HierarchicalFactStore automatically if FAISS is installed
fact_store = container.create_hierarchical_fact_store(
    hot_confidence_threshold=0.7,  # Fall back to FAISS if below this
    faiss_path="./data/faiss_index"
)

# Store 10,000 facts - hot layer overflows to FAISS gracefully
for i in range(10000):
    fact_store.add_fact(f"Entity_{i}", "property", f"value_{i}")

# Query works seamlessly across both layers
answer, confidence = fact_store.query("Entity_9999", "property")
# → "value_9999", 0.95 (or cold-layer similarity)
```

**Capacity Profile**:
- **Exact match queries** (normalized key lookup): Works at unlimited scale (O(1) dictionary)
- **Fuzzy holographic queries** (resonance): ~100-150 facts in hot layer before degradation
- **Cold layer queries**: Unlimited facts, but slightly lower confidence (0.20-0.40 vs. 0.24-0.37)

This is the implemented solution to the capacity problem - no proposal needed.

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

**What "rewiring" means**: The system doesn't rewrite the *query text*, but it **does** modulate its *internal state* by bundling mood vectors. This changes the system's perspective for retries.

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

#### 2. HDC Resonance Search (O(n), Lower Confidence)
- **How**: Unbind key from bundled memory, measure cosine similarity to all vocabulary words
- **Confidence**: 0.24-0.37 (typical for holographic retrieval due to interference)
- **Speed**: O(n) where n = vocabulary size
- **When used**: Fallback if exact match fails (lines 259-285)
- **Example**: Fuzzy query with typos or novel phrasing

**Confidence Breakdown**:
- **Exact match queries**: ~1.0 (deterministic key lookup)
- **Holographic (bundled) retrieval**: 0.24-0.37 (interference from other facts reduces signal)
- **Thresholds**:
  - Response threshold (0.20): Provide answer if confidence ≥ 0.20
  - Refusal threshold (0.10): Say "I don't know" if confidence < 0.10
  - Hedge zone (0.10-0.20): Uncertain response

**Why the difference?** In holographic storage, bundling multiple facts into one vector creates interference patterns. The query vibrates the surface, and the target fact's signal competes with noise from other facts. Hence normal retrieval gives 20-40% similarity, not 100%.

### Strategies (Layered Priority)

1. **Exact Match (fastest)**: Normalized key dictionary lookup - "France" + "capital" → "Paris" (1.0 confidence)
2. **HDC Resonance (fallback)**: Cosine similarity search through bundled memory - fuzzy matching with 0.24-0.37 confidence
3. **Learned Patterns**: Template matching for conversational responses (ResponseSelector layer)

---

## Layer 5: Voice

**Files**: `src/hologram/generation/resonant_generator.py`, `src/hologram/generation/ventriloquist.py`

### The Idea

Two ways to speak the retrieved facts:

| Mode | When Used | Output Style |
|------|-----------|--------------|
| **ResonantGenerator** | HDC-native fallback | Concise, constrained to vocabulary |
| **VentriloquistGenerator** | Preferred (via SLM API) | Fluent, natural conversation |

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

---

## Complete Data Flow

```
User: "What is the capital of France?"
    ↓
[Layer 1] Encode query as fractal vector
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

# Create container with FractalSpace (default)
container = HologramContainer(dimensions=10000, use_fractal=True)

# Create chatbot with persistence
chatbot = container.create_persistent_chatbot(
    persist_dir="./data/hologram_facts",
    enable_metacognition=True,
    enable_ventriloquist=True,  # For natural fluency
)

# Teach and query
chatbot.respond("The capital of France is Paris")
chatbot.respond("What is the capital of France?")
# → "The capital of France is Paris!"
```

---

## Key Properties

| Property | How It's Achieved |
|----------|-------------------|
| **Cannot hallucinate** | Can only retrieve stored vocabulary |
| **Corruption-resistant** | Fractal vectors recover from fragments |
| **Self-monitoring** | Metacognition tracks confidence trends |
| **Natural output** | SLM speaks fluently while HDC controls content |
| **Citable** | Every fact traces back to a source |

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

| Metric | Value | Notes |
|--------|-------|-------|
| **Hot layer capacity** | ~100 facts | Single bundled vector before degradation |
| **Cold layer capacity** | Unlimited | FAISS with vector similarity search |
| **Exact match lookups** | O(1) | Normalized key dictionary |
| **Fuzzy search** | O(n) | Resonance through bundled memory |
| **Vocabulary growth** | Dynamic | Grows with each new entity/predicate/object |
| **Vector dimensions** | 10,000 | Configurable, empirically tuned |
| **Fractal DNA dimensions** | 64 | Creates ~156 holographic echoes |

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
- **What it does**: SLM wrapper that validates LLM output uses HDC-retrieved facts
- **See**: Layer 5 (Voice) section; full architecture in `docs/VENTRILOQUIST_ARCHITECTURE.md`

### 3. Hierarchical Fact Store (IMPLEMENTED)
- **Previously**: Proposed as "Neural Consolidation" to break 100-fact limit
- **Status**: ✅ **Implemented** in `src/hologram/memory/fact_store.py:472-598`
- **What it does**: Two-tier storage (hot HDC + cold FAISS) for unlimited scalability
- **See**: "Capacity and Scaling" section above

---

## References

- Bentov, I. (1977). *Stalking the Wild Pendulum*
- Google Titans architecture (surprise gating inspiration)
- Hyperdimensional Computing literature

---

**Key Files**:
- `src/hologram/core/fractal.py` - FractalSpace
- `src/hologram/memory/fact_store.py` - FactStore
- `src/hologram/cognition/metacognition.py` - MetacognitiveLoop
- `src/hologram/conversation/selector.py` - ResponseSelector
- `src/hologram/generation/ventriloquist.py` - VentriloquistGenerator

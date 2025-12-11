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

1. Generate a dense 64-dim seed for each concept
2. Use rotation matrices to "echo" it across 10,000 dimensions
3. Any 64-dim block can reconstruct the original seed

### Trade-off

Less effective capacity (64 concepts vs 10,000) but much better corruption recovery.

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

### Surprise Gating

Inspired by Google Titans: only store facts that are "surprising" (novel). If you try to teach the same fact twice, it skips the duplicate.

### Capacity

Around 100 facts before interference becomes too noisy. Future "Neural Consolidation" proposal aims to break this limit.

---

## Layer 3: Metacognition

**File**: `src/hologram/cognition/metacognition.py`

### The Idea

The system monitors its own confidence and tracks its "mood":
- **NEUTRAL** → starting state
- **CONFIDENT** → high confidence answers
- **CONFUSED** → low confidence, triggers retry

### How It Actually Works

When confidence is low:
1. Updates internal mood to CONFUSED
2. Retries the same query (different patterns may match)
3. Takes the better result if retry improves confidence

**Note**: The current implementation observes and retries, but doesn't deeply "rewire" the query. The mood tracking is mostly for observability.

### Circuit Breaker

Prevents infinite retry loops - gives up after max attempts.

---

## Layer 4: Retrieval

**Files**: `src/hologram/memory/fact_store.py`, `src/hologram/conversation/selector.py`

### The Idea

Multiple strategies to find stored facts, prioritized by speed and confidence.

### Strategies

1. **Exact Match (fastest)**: Normalized key lookup - "France" + "capital" → "Paris"
2. **HDC Resonance (fallback)**: Cosine similarity search through bundled memory
3. **Learned Patterns**: Template matching for conversational responses

### Confidence Scores

For holographic storage, **20-40% similarity is normal** for stored facts (interference from other facts reduces the signal). The thresholds are:
- **Response**: 0.20 (sufficient signal detected)
- **Refusal**: 0.10 (below this, say "I don't know")

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

## Current Performance

- **Quiz Accuracy**: 81% (target: 90%+)
- **Fact Capacity**: ~100 per bundle (limited by interference)
- **Vocabulary**: Dynamic, grows with learning

---

## Future Proposals

1. **Holographic Surprise** ([PROPOSAL_1](../PROPOSAL_1_HOLOGRAPHIC_SURPRISE.md)) - Better noise reduction
2. **Ventriloquist Architecture** ([PROPOSAL_2](../PROPOSAL_2_VENTRILOQUIST_ARCHITECTURE.md)) - Full SLM integration
3. **Neural Consolidation** ([PROPOSAL_3](../PROPOSAL_3_NEURAL_CONSOLIDATION.md)) - Break the 100-fact limit

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

# Kent Hologram - Hyperdimensional Memory System

A next-generation AI chatbot using **Hyperdimensional Computing (HDC)** to store knowledge as interference patterns, providing bounded hallucination and deterministic retrieval.

---

## Overview

Unlike traditional LLMs that predict tokens probabilistically, Kent Hologram uses **algebraic memory** inspired by holographic principles. Facts are stored as interference patterns in 10,000-dimensional space, retrieved through resonance rather than prediction.

**Key Innovation**: Kent Hologram **cannot hallucinate outside its vocabulary**. It can only retrieve what it has learned, and it says "I don't know" when uncertain.

---

## Architecture

### The Holographic Memory Paradigm

Based on Itzhak Bentov's holographic consciousness theory:

1. **Concepts as Disturbances**: Each word/concept is a random 10,000-dimensional vector
2. **Memory as Interference**: Facts create interference patterns when superimposed
3. **Query as Resonance**: Retrieval vibrates the memory surface to extract patterns

### Five-Layer "Conscious Hologram" Stack

| Layer | Component | Role |
|-------|-----------|------|
| **1. Substrate** | FractalSpace | Every vector is holographically robust to corruption |
| **2. Memory** | FactStore + MemoryTrace | HDC binding/bundling for fact storage (S-P-O structure) |
| **3. Controller** | MetacognitiveLoop | Self-monitoring, confidence tracking, retry when confused |
| **4. Retrieval** | ResponseSelector | Multi-strategy lookup (exact match + HDC resonance) |
| **5. Voice** | Ventriloquist (SLM) | Natural language generation with fact grounding |

### HDC vs Traditional Embeddings

**Traditional Embeddings (Dense Vectors)**:
- Learned through gradient descent
- Probabilistic similarity (cosine/dot product)
- Prone to hallucination when interpolating

**Hyperdimensional Computing**:
- Algebraic operations (bind/bundle/unbind)
- Deterministic retrieval (same query → same answer)
- Bounded hallucination (can't invent facts)
- Graceful degradation (noise lowers confidence, not accuracy)

---

## Key Capabilities

### 1. Episodic Memory with HDC Vectors

Every conversation turn is encoded as a hypervector and bundled into memory. The system remembers:
- What you said (user input vector)
- How it responded (response vector)
- When it happened (temporal binding)
- Conversation context (bundled turn history)

**Example**:
```
You: "the capital of France is Paris"
Hologram: "Got it! I'll remember that France capital Paris."

[Internally stores: bind(France, capital) → Paris as interference pattern]

You: "What is the capital of France?"
Hologram: "The capital of France is Paris."
```

### 2. Intent Classification (HDC Learning)

Intent detection uses **learned prototypes**, not hardcoded keywords:
- GREETING, QUESTION, TEACHING, STATEMENT, COMMAND, FAREWELL
- Learns from examples (you can teach it new patterns)
- Confidence scoring on every classification

### 3. Fact Storage and Retrieval

**Two-Strategy Hybrid**:
1. **Exact Match** (O(1) fast path): Direct lookup for known facts
2. **Resonance Search** (O(N) fallback): HDC similarity for partial matches

**Storage Format**: Subject-Predicate-Object (S-P-O) triples
- "France" --capital--> "Paris"
- "Earth" --shape--> "round"
- "Sky" --color--> "blue"

### 4. Neural Consolidation (Sleep-Inspired Learning)

Facts transition through three stages:
1. **Working Memory**: New facts in ChromaDB (fast retrieval)
2. **Consolidation**: Background "sleep" process transfers to neural HDC memory
3. **Long-Term Memory**: Holographic storage in 10,000D space

**Benefits**:
- Facts persist forever without degradation
- Multi-threaded consolidation doesn't block chat
- Automatic deduplication and validation

### 5. Real-Time Activity Monitoring

The frontend dashboard shows:
- **Intent classification** (what the system thinks you're asking)
- **Confidence scores** (how certain it is)
- **Facts learned** (what knowledge was added)
- **Thinking processes** (resonance search, generation steps)

---

## Innovation Highlights

### What Makes This Different from Traditional Chatbots

**Traditional Chatbots (GPT, Claude, etc.)**:
- Predict next token based on statistical patterns
- Can hallucinate plausible-sounding nonsense
- No explicit memory (context window only)
- Probabilistic, non-deterministic

**Kent Hologram**:
- Retrieves facts algebraically from stored patterns
- Cannot invent information (bounded vocabulary)
- Persistent memory across sessions
- Deterministic, reproducible responses
- Citations for every fact (source tracking)

### Why HDC for Memory?

1. **Distributed Representation**: Each fact is spread across all 10,000 dimensions
2. **Superposition**: Multiple facts stored in same vector space (holographic bundling)
3. **Associative Recall**: Partial cues retrieve complete patterns
4. **Noise Tolerance**: 30% corruption still retrieves correct answer
5. **Constant-Time Operations**: No gradient descent or backprop needed

### Memory Consolidation Approach

Inspired by **neuroscience research** on sleep and memory:
- **Awake**: Store facts in fast working memory (ChromaDB)
- **Asleep**: Background consolidation to holographic long-term memory
- **Replay**: Strengthens important facts, prunes duplicates

This mimics how humans consolidate learning during sleep!

---

## Things to Test

### 1. **Teach It Facts**
Try telling Hologram factual statements:
- "the capital of Germany is Berlin"
- "Earth's shape is round"
- "dogs are mammals"

Watch it learn and confirm: "Got it! I'll remember that..."

### 2. **Test Episodic Memory**
Ask follow-up questions:
- "What is the capital of Germany?" (should recall what you taught)
- "What shape is Earth?" (retrieves from memory)

### 3. **Bounded Hallucination Test**
Ask about things it doesn't know:
- "What is the capital of Mars?"
- "Who is the president of Antarctica?"

It should say "I don't know" instead of guessing!

### 4. **Intent Classification**
Try different conversation styles:
- **Greetings**: "Hello!", "Hi there!"
- **Questions**: "What is X?", "Tell me about Y"
- **Teaching**: "X is Y", "The color of the sky is blue"
- **Statements**: "That's interesting", "I see"

Watch the intent classifier detect your intent type.

### 5. **Confidence Monitoring**
Pay attention to confidence scores in the dashboard:
- **High confidence (>60%)**: Hologram is certain
- **Medium confidence (20-60%)**: Normal holographic interference
- **Low confidence (<20%)**: Will refuse to answer

Note: Due to holographic interference, 20-40% confidence for stored facts is **normal and expected**!

### 6. **Persistence Across Sessions**
1. Teach it several facts
2. Refresh the page or close the tab
3. Return later and ask about those facts

Your facts should persist! (Stored in ChromaDB + neural memory)

### 7. **Watch Consolidation**
The dashboard shows:
- **Pending facts**: Waiting in working memory
- **Consolidated facts**: Transferred to long-term HDC storage
- **Total vocabulary**: Words the system knows

Watch facts move from pending → consolidated over time.

---

## Technical Details

### System Requirements
- 10,000-dimensional vector space
- ~100 fact capacity per holographic layer (scales with hierarchy)
- Background consolidation worker (non-blocking)
- Persistent storage (ChromaDB + PyTorch checkpoints)

### Safety Mechanisms
- **RefusalPolicy**: Refuses when confidence < threshold (0.10)
- **CitationEnforcer**: Every answer traces to a stored fact
- **ConfidenceScorer**: Quantifies certainty (calibrated for holographic interference)
- **Circuit Breaker**: Falls back on repeated API failures

### Generation Modes
1. **Ventriloquist (SLM)**: Uses Kimi K2 Thinking for fluent, natural responses
2. **Resonant Generator (HDC)**: Pure hyperdimensional token-by-token generation
3. **Template Fallback**: Pre-defined patterns when generation fails

---

## Limitations

**Current Constraints**:
- ~100 facts per layer before saturation (use HierarchicalFactStore for more)
- Context window: 8K tokens (Kimi K2 limit)
- Consolidation is async (facts may take seconds to transfer)
- English-only (vocabulary is language-specific)

**Why Confidence Scores Are "Low"**:
Holographic memory creates interference patterns. When you store 50 facts, they all interfere with each other (this is expected!). A confidence of 25% for a stored fact is **normal** because:
- Multiple facts bundled together lower individual similarity
- Thresholds are calibrated for holographic behavior (0.20 response, 0.10 refusal)
- Unknown queries produce <10% confidence and are correctly refused

Think of it like listening to a conversation at a party—you can hear it clearly despite background noise.

---

## Learn More

**Project Links**:
- [Full Documentation](../README.md)
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)
- [Semantic Search Guide](../SEMANTIC_SEARCH.md)
- [Training Guide](../TRAINING_GUIDE.md)

**Concepts**:
- [HDC Concepts](../docs/CONSCIOUS_HOLOGRAM_ARCH.md)
- [Bentov Holographic Model](https://en.wikipedia.org/wiki/Itzhak_Bentov)
- [Hyperdimensional Computing](https://redwood.berkeley.edu/research/brain-inspired-computing/)

---

**Version**: 0.4.0
**Technology**: Python 3.11+, PyTorch, torchhd, ChromaDB, React (frontend)
**License**: MIT

Built with **algebraic memory**, not statistical prediction.

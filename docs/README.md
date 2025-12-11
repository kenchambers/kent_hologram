# Hologram Documentation Index

Complete technical documentation for the Conscious Hologram HDC-based knowledge system.

---

## Start Here 🚀

### For New Users
1. **[CONSCIOUS_HOLOGRAM_ARCH.md](./CONSCIOUS_HOLOGRAM_ARCH.md)** - Architecture overview and 5-layer system design
   - Conceptual overview of how the system works
   - All 5 layers explained
   - Performance metrics and capacity information
   - Usage examples

### For Developers
2. **[CONVERSATIONAL_LEARNING.md](./CONVERSATIONAL_LEARNING.md)** - Layer 4: Conversation system deep-dive
   - Intent classification
   - Entity extraction
   - Response pattern learning (Hebbian)
   - Style tracking and adaptation
   - Complete API reference

3. **[VENTRILOQUIST_ARCHITECTURE.md](./VENTRILOQUIST_ARCHITECTURE.md)** - Layer 5: Generation system deep-dive
   - ResonantGenerator (HDC-native token-by-token generation)
   - VentriloquistGenerator (SLM fluency wrapper)
   - Hybrid routing and generation validation
   - Resonant Cavity architecture details

### For System Architects
4. **[HDC_LEARNING_PHILOSOPHY.md](./HDC_LEARNING_PHILOSOPHY.md)** - Design principles and philosophy
   - Why HDC instead of traditional NLP/ML
   - Anti-patterns to avoid
   - Learning approach (Hebbian, example-based, no hardcoding)
   - Guidelines for extending the system

### For Code Tasks
5. **[CODE_ENHANCEMENT_GUIDE.md](./CODE_ENHANCEMENT_GUIDE.md)** - Specialized feature: Code generation and genealogy
   - Using Hologram for code enhancement
   - Reverse queries for code genealogy
   - Fact-grounded code generation

---

## Document Map

### Architecture & Design

| Document | Audience | Key Topics |
|----------|----------|-----------|
| **CONSCIOUS_HOLOGRAM_ARCH.md** | Everyone | 5 layers, performance, capacity, usage |
| **HDC_LEARNING_PHILOSOPHY.md** | Architects, Extension Developers | Why HDC, design patterns, anti-patterns |
| **VENTRILOQUIST_ARCHITECTURE.md** | Generation System Developers | Layer 5, ResonantGenerator, VentriloquistGenerator |
| **CONVERSATIONAL_LEARNING.md** | Conversation System Developers | Layer 4, intent, entities, patterns, learning |
| **CODE_ENHANCEMENT_GUIDE.md** | Code Generation Users | Using Hologram for code tasks |

---

## Quick Reference

### The 5 Layers

1. **Layer 1: Fractal Substrate** (`src/hologram/core/fractal.py`)
   - Deterministic DNA expansion to 10,000 dimensions
   - Holographic recovery from any 64-dim fragment
   - See: CONSCIOUS_HOLOGRAM_ARCH.md → Layer 1

2. **Layer 2: Memory** (`src/hologram/memory/fact_store.py`, `memory_trace.py`)
   - Subject-Predicate-Object triples
   - Surprise-gated learning (Titans-inspired)
   - Hierarchical hot/cold storage (HDC + FAISS)
   - See: CONSCIOUS_HOLOGRAM_ARCH.md → Layer 2

3. **Layer 3: Metacognition** (`src/hologram/cognition/metacognition.py`)
   - Self-monitoring with mood states
   - Confidence-based retry loops
   - Internal state modulation via HDC bundling
   - See: CONSCIOUS_HOLOGRAM_ARCH.md → Layer 3

4. **Layer 4: Conversation** (`src/hologram/conversation/`)
   - Intent classification (learned prototypes)
   - Entity extraction (resonance matching)
   - Response selection with pattern learning
   - See: CONVERSATIONAL_LEARNING.md

5. **Layer 5: Generation** (`src/hologram/generation/`)
   - ResonantGenerator (HDC-native, bounded hallucination)
   - VentriloquistGenerator (SLM fluency wrapper)
   - Hybrid routing logic
   - See: VENTRILOQUIST_ARCHITECTURE.md

---

## Key Concepts

### Holographic Storage
- Facts bundled into single vector via superposition
- Retrieval via unbinding and cosine similarity
- Confidence: 0.24-0.37 for bundled facts (interference)
- See: CONSCIOUS_HOLOGRAM_ARCH.md → Layer 2

### Surprise Gating
- Dual surprise metrics (current + momentum)
- Prevents duplicate learning
- Learning rate modulation based on novelty
- See: CONSCIOUS_HOLOGRAM_ARCH.md → Surprise Gating

### Hebbian Learning
- Patterns strengthened if conversation flows naturally
- Patterns weakened if they cause confusion
- Emergent behavior through reinforcement
- See: CONVERSATIONAL_LEARNING.md → Pattern Learning

### Bounded Hallucination
- System can ONLY output facts from holographic memory
- Unknown queries produce low confidence
- Dual query modes: exact match (O(1), 1.0 confidence) vs. resonance (O(n), 0.24-0.37)
- See: CONSCIOUS_HOLOGRAM_ARCH.md → Layer 4

### Ventriloquist Pattern
- HDC controls **what** to say (factual grounding)
- SLM controls **how** to say it (fluency)
- No LLM hallucination because facts come from HDC
- See: VENTRILOQUIST_ARCHITECTURE.md

---

## Common Questions

**Q: How do I use Hologram for my project?**
→ Start with CONSCIOUS_HOLOGRAM_ARCH.md (Usage section), then see the example scripts in `examples/`

**Q: How does conversation learning work?**
→ CONVERSATIONAL_LEARNING.md has detailed explanation with API reference

**Q: How is fact capacity limited?**
→ CONSCIOUS_HOLOGRAM_ARCH.md → Capacity and Scaling explains the two-tier (HDC + FAISS) solution

**Q: Why doesn't Hologram hallucinate?**
→ See CONSCIOUS_HOLOGRAM_ARCH.md → Key Properties → "Cannot hallucinate" explanation

**Q: How do I add a new feature?**
→ HDC_LEARNING_PHILOSOPHY.md explains the design approach and anti-patterns to avoid

**Q: How is generation implemented?**
→ VENTRILOQUIST_ARCHITECTURE.md explains both ResonantGenerator and VentriloquistGenerator

---

## Performance & Validation

- **Quiz Accuracy**: ~81% (target: 90%+)
- **Hot layer capacity**: ~100 facts (unbundled vector)
- **Cold layer capacity**: Unlimited (FAISS)
- **Exact match lookups**: O(1), confidence ≈ 1.0
- **Fuzzy holographic search**: O(n), confidence ≈ 0.24-0.37

Test with: `uv run pytest tests/test_hdc_fact_grounding.py -v`

See: CONSCIOUS_HOLOGRAM_ARCH.md → Performance and Validation

---

## Implemented Advanced Features

The following features are **fully implemented** (not proposals):

✅ **Surprise-Gated Learning** - Dual-surprise metrics prevent duplicate learning and optimize learning rate
  - See: CONSCIOUS_HOLOGRAM_ARCH.md → Surprise Gating

✅ **Ventriloquist Architecture** - SLM wrapper that validates LLM output uses HDC-retrieved facts
  - See: VENTRILOQUIST_ARCHITECTURE.md

✅ **Hierarchical Fact Store** - Two-tier storage (hot HDC + cold FAISS) for unlimited scalability
  - See: CONSCIOUS_HOLOGRAM_ARCH.md → Capacity and Scaling

---

## File Structure

```
docs/
├── README.md (this file)
├── CONSCIOUS_HOLOGRAM_ARCH.md (main architecture)
├── CONVERSATIONAL_LEARNING.md (Layer 4 deep-dive)
├── VENTRILOQUIST_ARCHITECTURE.md (Layer 5 deep-dive)
├── HDC_LEARNING_PHILOSOPHY.md (design principles)
└── CODE_ENHANCEMENT_GUIDE.md (code generation feature)

src/hologram/
├── core/
│   ├── fractal.py (Layer 1)
│   ├── operations.py (HDC bind/bundle/unbind)
│   ├── codebook.py (deterministic hash→vector)
│   └── similarity.py
├── memory/
│   ├── fact_store.py (Layer 2, FactStore + HierarchicalFactStore)
│   └── memory_trace.py (holographic storage, surprise gating)
├── cognition/
│   └── metacognition.py (Layer 3)
├── conversation/
│   ├── chatbot.py (Layer 4 orchestration)
│   ├── intent.py (Intent classification)
│   ├── entity.py (Entity extraction)
│   ├── selector.py (Response selection)
│   ├── patterns.py (Pattern store with Hebbian learning)
│   ├── style_tracker.py (Style adaptation)
│   └── corpus.py (Learned response corpus)
├── generation/
│   ├── ventriloquist.py (Layer 5, SLM generation)
│   ├── resonant_generator.py (Layer 5, HDC generation)
│   ├── base.py (GenerationContext, Generator protocol)
│   └── circuit_breaker.py (Failure detection)
├── persistence/
│   └── faiss_adapter.py (FAISS cold storage)
└── config/
    └── constants.py (System hyperparameters)
```

---

## For Maintainers

**Documentation Update Notes** (Last Updated: 2025-12-11):

1. **CONSCIOUS_HOLOGRAM_ARCH.md** - Major revision:
   - ✅ Fixed capacity claim (added HierarchicalFactStore explanation)
   - ✅ Expanded surprise gating (was 3 sentences, now 2 pages)
   - ✅ Clarified metacognition rewiring
   - ✅ Added dual query modes explanation
   - ✅ Added empirical fractal recovery properties
   - ✅ Fixed "Future Proposals" section (relabeled as "Implemented Features")
   - ✅ Added "Performance and Validation" section

2. **All other docs** - Verified current, no changes needed:
   - VENTRILOQUIST_ARCHITECTURE.md ✅
   - CONVERSATIONAL_LEARNING.md ✅
   - HDC_LEARNING_PHILOSOPHY.md ✅
   - CODE_ENHANCEMENT_GUIDE.md ✅

---

## How to Contribute

When adding new features:

1. Update CONSCIOUS_HOLOGRAM_ARCH.md if the 5-layer system is affected
2. Update the relevant layer deep-dive (CONVERSATIONAL_LEARNING.md or VENTRILOQUIST_ARCHITECTURE.md)
3. Add code examples to documentation if it's a new API
4. Test documentation examples with actual code
5. Update this README.md if navigation changes

When implementing design changes:

1. See HDC_LEARNING_PHILOSOPHY.md for design principles
2. Avoid hardcoding - use learned prototypes and patterns
3. Leverage HDC operations (bind, bundle, unbind) where possible
4. Document your rationale in code comments and relevant doc

---

**Last Updated**: 2025-12-11
**Status**: Fully Implemented (All 5 Layers)
**Maintainer**: Ken Chambers

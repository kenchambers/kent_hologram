# Kent Hologram Training Architecture Diagrams

---

## 1. Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KENT HOLOGRAM TRAINING SYSTEM                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        THREE LEARNING PATHS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PATH 1: CONVERSATIONAL (crew_trainer.py)                           │
│  ───────────────────────────────────────────────                    │
│   Gemini (Topic Starter) ──┐                                         │
│   Claude (Discussant)      │  → CrewTrainer → Hologram (Learner)   │
│   Quiz Loop ───────────────┘                                         │
│                                                                       │
│  Output: Conversation logs + Facts (ChromaDB)                       │
│  Timeline: 50-100 rounds = 30-120 mins                              │
│  Cost: $0.25-0.50                                                   │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PATH 2: WEB TEACHING (within crew_trainer.py)                      │
│  ──────────────────────────────────────────────                     │
│   Topics ("World Capitals") ──┐                                      │
│   DuckDuckGo Search ──────────│→ WebTeacher → Fact Extraction       │
│   LLM Processing ─────────────┘              → Store Facts           │
│                                                                       │
│  Output: Bulk-loaded facts (ChromaDB)                               │
│  Timeline: 5-10 mins per topic                                      │
│  Cost: $0.01-0.10 per topic                                         │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  PATH 3: DOCUMENT INGESTION (ingest_gutenberg.py)                   │
│  ────────────────────────────────────────────────────               │
│   Project Gutenberg Dataset ────┐                                    │
│   (75,570 books, streaming) ───│→ Clean & Chunk → Extract Facts    │
│   Resume from Checkpoint ───────┘    (1000 char)   via LLM         │
│                                                                       │
│  Output: Large-scale facts (ChromaDB)                               │
│  Timeline: 50 books = 10-15 mins                                    │
│  Cost: $2-5 (mostly compute)                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Crew Trainer Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                       CrewTrainer Main Loop                       │
└──────────────────────────────────────────────────────────────────┘

INITIALIZATION
──────────────
  • Load API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY)
  • Create CrewAI agents (Gemini, Claude)
  • Initialize Hologram container (10,000 dims)
  • Load vocabulary from existing facts
  • Create persistent chatbot with generators
  • Initialize conversation logger
  • Load cadence memory (optional)

START CONVERSATION ROUNDS
──────────────────────────

  Round N:
  ────────

    1. TOPIC SELECTION
       ┌─────────────────────────────────────┐
       │ Category Rotation:                  │
       │ "World Capitals" →                  │
       │ "Famous Inventors" →                │
       │ "Science Facts" → ...               │
       └─────────────────────────────────────┘

    2. GEMINI TEACHES (via LLM)
       ┌─────────────────────────────────────────────┐
       │ System Prompt:                              │
       │   - Teach ONE fact per message             │
       │   - Format: "Capital of X is Y"            │
       │   - Quiz on the fact taught                │
       │   - Use assigned category                  │
       │   - Keep sentences short (<12 words)       │
       └─────────────────────────────────────────────┘
       Output: "The capital of France is Paris"

    3. HOLOGRAM RESPONDS
       ┌──────────────────────────────────────┐
       │ Intent Detection:                    │
       │  - Is this TEACHING intent?         │
       │  - Extract S-P-O triple             │
       │  - Store in FactStore               │
       │  - Log: ✓ Fact learned              │
       └──────────────────────────────────────┘

    4. CLAUDE REINFORCES (alternates with Gemini)
       ┌──────────────────────────────────────┐
       │ - Confirm the fact                  │
       │ - Add related fact                  │
       │ - Quiz on both facts                │
       └──────────────────────────────────────┘

    5. DISCUSSION LOOP (up to 8 turns)
       ┌──────────────────────────────────────┐
       │ Speaker Rotation:                    │
       │  70% Claude, 30% Gemini             │
       │                                      │
       │ Hologram:                            │
       │  - Listens to all messages          │
       │  - Learns facts while listening     │
       │  - Responds 20% of the time         │
       │  - Learns responses (atomic)        │
       │  - Tracks cadence patterns          │
       └──────────────────────────────────────┘

    6. PERIODIC CONSOLIDATION
       ┌──────────────────────────────────────┐
       │ Every 10 rounds:                     │
       │  - Neural consolidation             │
       │  - Save memory to disk               │
       │  - Update vocabulary stats           │
       │  - Print status report               │
       └──────────────────────────────────────┘

    7. STORAGE
       ┌──────────────────────────────────────┐
       │ ChromaDB:                            │
       │  - Facts (S-P-O triples)            │
       │  - Responses (learned sentences)    │
       │  - Vocabulary (nouns/verbs)         │
       │                                      │
       │ Memory:                              │
       │  - Neural memory (Torch tensors)    │
       │  - Cadence patterns                 │
       │  - Self-improvement patterns        │
       └──────────────────────────────────────┘

END: Ctrl+C or max_rounds reached
────────────────────────────────
  • Final consolidation
  • Save all memory
  • Print final statistics
  • Close connections
```

---

## 3. Codebook Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              CODEBOOK ARCHITECTURE DECISION TREE                 │
└─────────────────────────────────────────────────────────────────┘

DO YOU NEED SEMANTIC ANALOGIES?
│
├─ YES (e.g., Paris:France :: Tokyo:???)
│  │
│  └─ USE: SemanticCodebook
│     ├─ Embeddings: sentence-transformers (384d)
│     ├─ Projection: Fixed random matrix (384d → 10000d)
│     ├─ Result: "Tokyo" (confidence 0.7+)
│     ├─ Cost: Slower (embedding computation)
│     └─ Requirements: sentence-transformers library
│
└─ NO (e.g., slot filling, conversational learning)
   │
   └─ USE: Base Codebook
      ├─ Generation: Hash-seeded random vectors
      ├─ Similarity: Random orthogonal
      ├─ Strength: Slot filling works (coherence-based)
      ├─ Cost: Very fast
      └─ Requirements: None (pure)


┌─────────────────────────────────────────────────────────────────┐
│              CODEBOOK IMPLEMENTATION DETAILS                     │
└─────────────────────────────────────────────────────────────────┘

BASE CODEBOOK
──────────────────────────────────────────────────────────────────
concept: str → Vector (10000d)

  "Paris" ────┐
              ├─ hash(concept) ──┐
  "France" ───┤                  ├─ torch.randn(seed) ───> orthogonal
              │   (SHA-256)   ──┘    vectors
  "Tokyo" ────┤
              ├─ ~orthogonal (~0.0 similarity)
  "Japan" ────┘

  Pros:
    ✓ Deterministic (same input = same vector)
    ✓ Very fast (no dependencies)
    ✓ Slot completion works (coherence-based)
    ✓ Pure HDC (no semantic structure needed)

  Cons:
    ✗ No semantic similarity
    ✗ Analogies fail (random results)


SEMANTIC CODEBOOK
──────────────────────────────────────────────────────────────────
concept: str → Vector (10000d)

  "Paris" ───────┐
                 ├─ SentenceTransformer ──┐   384d
  "France" ──────┤ (all-MiniLM-L6-v2)    ├─ semantic
                 │                       │   embeddings
  "Tokyo" ───────┤                       │   (related
                 ├─ Fixed Projection ────┘    concepts)
  "Japan" ───────┘ (seed 0xCAFEBABE)     ↓
                                      10000d
                    Normalize → HDC vectors
                    (similar ≠ orthogonal!)

  Paris ≈ 0.6 similarity with France (semantically close!)
  Tokyo ≈ 0.6 similarity with Japan

  Pros:
    ✓ Semantic similarity preserved
    ✓ Analogies work (semantic relation captured)
    ✓ Better generalization

  Cons:
    ✗ Slower (embedding computation)
    ✗ Requires sentence-transformers
    ✗ Memory overhead
```

---

## 4. Fact Store Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       FACT STORE DESIGN                           │
└──────────────────────────────────────────────────────────────────┘

INPUT: (Subject, Predicate, Object) Triple
   ↓
   ├─ "France" (subject)
   ├─ "capital" (predicate)
   └─ "Paris" (object)

ENCODING PHASE
──────────────────────────────────────────────────────────────────

   Step 1: Encode each component
   france_vec = codebook.encode("France")      → 10000d vector
   pred_vec   = codebook.encode("capital")     → 10000d vector
   paris_vec  = codebook.encode("Paris")       → 10000d vector

   Step 2: Create nested binding
   key = bind(bind(france_vec, pred_vec))      → holographic key

   Step 3: Store value
   store(key, paris_vec)                       → in MemoryTrace

   Mathematical representation:
   MemoryTrace[bind(bind(S, P))] = O

   Where ⊗ = bind (element-wise multiplication in frequency domain)


RETRIEVAL PHASE
──────────────────────────────────────────────────────────────────

   FORWARD QUERY: "What is the capital of France?"

   query_input = bind(bind(france_vec, pred_vec))
   query_result = retrieve(query_input)        → noisy vector

   CLEANUP: Find closest vocabulary word
   candidates_similarity = [
     cos(query_result, paris_vec)     → 0.95 ✓ HIGH
     cos(query_result, london_vec)    → 0.15
     cos(query_result, tokyo_vec)     → 0.10
   ]

   Answer: "Paris" (confidence: 0.95)


   REVERSE QUERY: "What has capital of Paris?"

   query_input = bind(bind(paris_vec, inverse(pred_vec)))
   query_result = retrieve(query_input)        → noisy vector

   CLEANUP: Find closest vocabulary word
   Answer: "France" (confidence: 0.92)


STORAGE BACKEND
──────────────────────────────────────────────────────────────────

   Option 1: ChromaDB (Default)
   ├─ Dense vector storage
   ├─ Supports similarity search
   ├─ Works for 1K-100K facts
   └─ Simple setup

   Option 2: EmergentLayerFactStore (Scalable)
   ├─ Layered organization
   ├─ Efficient retrieval
   ├─ Supports 100K+ facts
   ├─ Flag: --emergent-layers
   └─ Better for large-scale


METADATA TRACKING
──────────────────────────────────────────────────────────────────

   Each Fact stores:
   ├─ subject: str               ("France")
   ├─ predicate: str             ("capital")
   ├─ object: str                ("Paris")
   ├─ confidence: float           (0.0-1.0)
   ├─ source: str                ("crew_training")
   ├─ timestamp: datetime         (when learned)
   └─ surprise_score: float       (novelty metric)
```

---

## 5. Optimal Training Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                 RECOMMENDED TRAINING SEQUENCE                     │
└──────────────────────────────────────────────────────────────────┘

TIME    PHASE               ACTIONS                    OUTCOMES
────────────────────────────────────────────────────────────────────

0:00    SETUP              • uv sync
                           • Load API keys
                           • Check dependencies

        STATUS: Ready

0:05    BOOTSTRAP          crew_trainer.py            Facts: 20-30
        PHASE 1            --max-rounds 50            Vocab: 200-300
        Conversations                                  Time: 30-40m

        WHY FIRST?
        ✓ Build vocabulary incrementally
        ✓ Natural diversity from dialogue
        ✓ Quiz feedback validates learning
        ✓ Low cost ($0.25)

0:45    (OPTIONAL)         crew_trainer.py             Facts: 20-50
        PHASE 2            --web-teach "Topics"       Cost: $0.01-0.10
        Web Teaching       (if cold-start needed)

        OR SKIP if you have initial vocab

1:00    DOCUMENT           ingest_gutenberg.py         Facts: 5,000-10,000
        PHASE 3            --max-books 50              Vocabulary boost
        Ingestion          (or --max-books 100)        Time: 10-15m

        WHY NOW?
        ✓ Large-scale fact injection
        ✓ Multiple perspectives
        ✓ Deep knowledge acquisition
        ✓ Prepares for consolidation

1:30    (OPTIONAL)         crew_trainer.py             Facts: 100-150
        PHASE 4            --max-rounds 50             Fine-tuning
        Fine-tuning        (continue conversations)    Time: 30-40m

        WHY AGAIN?
        ✓ Practice with learned facts
        ✓ Refine generation
        ✓ Monitor self-improvement
        ✓ Track learning curves

2:10    COMPLETE           • Final statistics
                           • Save all memory
                           • Analyze results


KNOWLEDGE GROWTH BY PHASE
────────────────────────────────────────────────────────────────────

        Conversations    Web       Books       Conversations  TOTAL
        (50 rounds)      Teach     (50-100)    (50 more)

Facts:      100          20-50    5,000-10,000  100         5,200-15,300
Vocab:      300-500      100-200  3,000-5,000   100         3,500-6,300
Time:       30-40m       5m       10-15m        30-40m       75-100m
Cost:       $0.25        $0.05    $2-5          $0.25        $2.75-5.50


WHAT GETS LEARNED
────────────────────────────────────────────────────────────────────

Phase 1 - Conversations:
  ├─ Basic facts (capitals, people)
  ├─ Vocabulary (common nouns/verbs)
  ├─ Grammatical patterns
  └─ Dialogue structure

Phase 3 - Books:
  ├─ Deep knowledge (100+ facts per book)
  ├─ Diverse vocabulary
  ├─ Complex sentence structures
  └─ Literary knowledge

Phase 4 - More conversations:
  ├─ Fine-tuning responses
  ├─ Practice with learned facts
  ├─ Generation quality
  └─ Self-improvement patterns
```

---

## 6. Memory Consolidation Timeline

```
┌──────────────────────────────────────────────────────────────────┐
│              MEMORY CONSOLIDATION PROCESS                         │
└──────────────────────────────────────────────────────────────────┘

LEARNING PHASE (Noisy)
──────────────────────────────────────────────────────────────────

Facts arrive: S-P-O triples
   ↓
Store in MemoryTrace (holographic)
   ↓
Each fact adds noise (interference)
   ↓
After 10 facts:
  ├─ Retrieval accuracy: ~70-80%
  ├─ Noise level: MEDIUM
  └─ Trigger consolidation


CONSOLIDATION PHASE (Trigger)
──────────────────────────────────────────────────────────────────

Option A: Automatic (default)
─────────────────────────────
Triggered after N facts (default: 10)
  ├─ Extract learned patterns
  ├─ Clean up interference
  ├─ Rebuild MemoryTrace
  └─ Reset for next batch

Option B: Explicit (periodic)
────────────────────────────
Every 10 conversation rounds:
  └─ chatbot.save_memory(persist_dir, force_consolidation=True)

Option C: Final (on exit)
───────────────────────
At end of training:
  └─ Merge all noisy traces into clean patterns


EFFECTS OF CONSOLIDATION
──────────────────────────────────────────────────────────────────

Before consolidation (10 facts):
  Query: "capital of France?"
  Noise level: MEDIUM
  Retrieval accuracy: 70-80%

After consolidation:
  Query: "capital of France?"
  Noise level: LOW
  Retrieval accuracy: 90-95%

After multiple consolidations (100 facts):
  Query: "capital of France?"
  Noise level: VERY LOW
  Retrieval accuracy: 95%+


CONSOLIDATION OVERHEAD
──────────────────────────────────────────────────────────────────

Time cost: ~100-200ms per consolidation
Memory cost: None (in-place operation)
Frequency: Every 10 facts by default
Total impact: <5% training time

RESULT: Automatic consolidation recommended
        (enabled by default, minimal overhead)
```

---

## 7. Test Pyramid

```
┌──────────────────────────────────────────────────────────────────┐
│                       TEST PYRAMID                                │
└──────────────────────────────────────────────────────────────────┘

                              TOP TESTS
                          (Integration)
                              /  \
                            /      \
                          /          \
                        E2E Tests    Feature Tests
                        20 tests     30 tests
                      (trainers)   (memory, etc)
                        ▲
                        │
                    MIDDLE TESTS
                    (Component)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
      Core HDC      Analogy Tests   Fact Store
      Tests         13 tests        Tests
      15 tests       ✓ ALL PASS     20 tests
      ✓ PASS
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▲
                        │
                    UNIT TESTS
                    (Functions)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    Codebook        Operations      Similarity
    Tests           Tests           Tests
    5 tests         5 tests         5 tests


RECOMMENDED RUN ORDER
──────────────────────────────────────────────────────────────────

1. QUICK CHECK (1 minute)
   $ pytest tests/test_codebook.py tests/reasoning/test_analogy_engine.py -v
   → 18 tests, all pass

2. FOUNDATION CHECK (5 minutes)
   $ pytest tests/core/ tests/test_hdc_fact_grounding.py -v
   → 30 tests, all pass

3. MEMORY CHECK (10 minutes)
   $ pytest tests/memory/ tests/test_neural_consolidation.py -v
   → 100+ tests, mostly pass

4. INTEGRATION CHECK (30 seconds)
   $ pytest tests/integration/ -v
   → 20 tests, all pass

5. FULL SUITE (10 minutes)
   $ pytest tests/ -v
   → 571 tests total


CONFIDENCE LEVELS
──────────────────────────────────────────────────────────────────

✓ Core functionality (100% pass)
  ├─ Codebook ✓
  ├─ Operations ✓
  ├─ Similarity ✓
  ├─ Analogy Engine ✓
  └─ Fact Store ✓

✓ Training pipeline (95%+ pass)
  ├─ Crew trainer ✓
  ├─ Web teacher ✓
  ├─ Memory consolidation ✓
  └─ Persistence ✓

✓ Advanced features (90%+ pass)
  ├─ Emergent layers ✓
  ├─ SWE reasoning ✓
  ├─ ARC solving ✓
  └─ Introspection ✓
```

---

## 8. Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────┐
│                  QUICK REFERENCE CARD                             │
└──────────────────────────────────────────────────────────────────┘

CODEBOOK SELECTION
──────────────────
Conversational learning → Base Codebook
Semantic analogies     → SemanticCodebook
Slot completion        → Base Codebook
Production system      → SemanticCodebook

TRAINING APPROACH
─────────────────
Cold start          → Web teach (5min) → Conversations (50 rounds)
With data           → Conversations (50 rounds) → Books (50 books)
Scale up            → Add --emergent-layers flag
Quick validation    → --max-rounds 10 --max-books 5

TEST COMMANDS
─────────────
Analogy tests      → pytest tests/reasoning/test_analogy_engine.py
Core HDC tests     → pytest tests/test_codebook.py tests/core/
Memory tests       → pytest tests/memory/ tests/test_neural_consolidation.py
Full suite         → pytest tests/ -v

TRAINING COMMANDS
──────────────────
Convo training     → python scripts/crew_trainer.py --max-rounds 50
Web teaching       → crew_trainer.py --web-teach "Topic1" "Topic2"
Code teaching      → crew_trainer.py --web-teach-code "Python" "Algorithms"
Book ingestion     → python scripts/ingest_gutenberg.py --max-books 50
Scalable mode      → Add --emergent-layers flag to any command

EXPECTED OUTCOMES
──────────────────
50 conversation rounds    → 100-150 facts, 500 vocabulary
50 books                  → 5,000-10,000 facts, 3,000+ vocabulary
Combined                  → 5,200-15,300 facts, 3,500-6,300 vocabulary
Time estimate             → 1.5-2.5 hours total
Cost estimate             → $2.75-5.50
```

---

**END OF DIAGRAMS**

# Conscious Hologram: Context and Token Flow Diagrams

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                   │
│                  "What is the capital of France?"                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
        ┌─────────────────▼────────────────────┐
        │   ConversationalChatbot              │
        │  ─────────────────────────────       │
        │  • Intent Classifier (HDC)           │
        │  • Entity Extractor (Regex)          │
        │  • Memory (Conversation history)     │
        │  • Style Tracker (User style)        │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │   ResponseSelector              │
        │  ─────────────────────────────  │
        │  1. Query FactStore (HDC)      │
        │  2. Route to Generator          │
        │     ├─ Prefer: SLM (Fluency)   │
        │     └─ Fallback: HDC/Templates │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────────────────┐
        │         DUAL GENERATION PATHS               │
        │                                               │
        │  A) SLM PATH (PREFERRED)                    │
        │  ├─ VentriloquistGenerator                  │
        │  ├─ Input: 34 tokens (system + user + fact) │
        │  ├─ API: Novita (8K window)                 │
        │  └─ Output: 256 tokens max                  │
        │                                               │
        │  B) HDC PATH (FALLBACK)                     │
        │  ├─ ResonantGenerator                       │
        │  ├─ Input: Thought vector (10K dims)        │
        │  ├─ Resonator: 100 iterations max           │
        │  └─ Output: 10 tokens max                   │
        │                                               │
        │  C) TEMPLATE PATH (ULTIMATE FALLBACK)       │
        │  ├─ ResponsePatternStore                    │
        │  └─ Output: Pre-defined template            │
        └──────────────┬──────────────────────────────┘
                       │
        ┌──────────────▼────────────────────┐
        │        RESPONSE OUTPUT              │
        │  ─────────────────────────────     │
        │  "The capital of France is Paris." │
        └───────────────────────────────────┘
```

---

## 2. Token Consumption Waterfall

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MESSAGE FLOW WITH TOKEN COSTS                      │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: INTENT + ENTITY CLASSIFICATION
  ├─ IntentClassifier.classify()
  │  ├─ Input encoding: hash → seed → torch.randn(10K dims)
  │  ├─ Bundling with prototypes
  │  └─ Cost: 0 SLM tokens (internal HDC)
  │
  └─ EntityExtractor.extract()
     ├─ Regex pattern matching
     └─ Cost: 0 tokens (string operations)

PHASE 2: FACT STORE QUERY
  ├─ FactStore.query("France", "capital")
  │  ├─ Encode subject: "France" → 10K dims (cached)
  │  ├─ Encode predicate: "capital" → 10K dims (cached)
  │  ├─ Bind vectors (element-wise multiply + permute)
  │  ├─ Resonance search: O(N) where N = vocabulary size
  │  └─ Cost: 0 SLM tokens (HDC vector operations)
  │
  └─ Returns: ("Paris", confidence=0.98)

PHASE 3: CONTEXT CONSTRUCTION (CRITICAL POINT)
  ├─ system_prompt:
  │  "You are a helpful assistant..."
  │  └─ ~20 tokens
  │
  ├─ user_prompt:
  │  "Question: What is the capital of France?\n\n"
  │  "Fact: Paris\n\n"
  │  "Answer the question using the fact above..."
  │  └─ ~14 tokens
  │
  └─ Total input: 34 tokens

PHASE 4: SLM API CALL (NOVITA)
  ├─ Input tokens: 34
  ├─ Model: moonshotai/kimi-k2-thinking (8K context window)
  ├─ Generation: max_tokens=256 (hard limit)
  └─ Output tokens: ~20 actual (depends on content)

PHASE 5: VALIDATION
  ├─ Check fact appears in response
  │  ├─ "Paris" in "The capital of France is Paris." ✓
  │  └─ Cost: 0 tokens (string matching)
  │
  └─ Returns: GenerationResult(text="...", tokens=[...])

PHASE 6: RETURN TO USER
  └─ "The capital of France is Paris."

────────────────────────────────────────────────────────────────────
TOTAL TOKENS THIS TURN:
  Input:  34 tokens
  Output: 20 tokens
  Sum:    54 tokens (well within 8K limit)
────────────────────────────────────────────────────────────────────
```

---

## 3. FactStore Query Process (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│              FACTSTORE QUERY: "What is capital of France?"        │
└─────────────────────────────────────────────────────────────────┘

User Question: "What is the capital of France?"
  │
  ├─ Extract entities: ["France", "capital"]
  │
  └─ Attempt queries in order:
     │
     ├─ [1] Try "France" + "capital"
     │  │
     │  ├─ STRATEGY 1: Exact Match (O(1))
     │  │  ├─ Normalize: "france" : "capital"
     │  │  ├─ Check exact_index dictionary
     │  │  ├─ HIT! Found in metadata
     │  │  └─ Return: ("Paris", confidence=1.0) ← FASTEST PATH
     │  │
     │  └─ (If exact match fails, would use STRATEGY 2)
     │     ├─ Encode subject: encode("france") → vec_s (10K dims)
     │     ├─ Encode predicate: encode("capital") → vec_p (10K dims)
     │     ├─ Create key: bind(vec_s, vec_p)
     │     ├─ Get all values: [vec_paris, vec_tokyo, ...]
     │     ├─ Compute similarities via resonance
     │     └─ Return: (best_match, max_similarity)
     │
     ├─ [2] Try "France" + "is"
     │  └─ No exact match, resonance search finds weak signals
     │
     └─ [3] Try from context / last turn
        └─ No additional context

INTERNAL STORAGE (MemoryTrace):
  ├─ Holds one superposed vector representing ALL facts
  ├─ "France" --capital--> "Paris" (bundled as interference pattern)
  ├─ "Paris" --population--> "2.2M" (also superposed)
  ├─ "Paris" --country--> "France" (also superposed)
  │
  └─ Capacity: ~100 facts before noise dominates signal
     (Empirically unproven, estimated from HDC literature)

CACHE LAYERS:
  ├─ _exact_index: {("france:capital"): Fact(...)} ← HIT
  ├─ _value_vectors_cache: {"paris": vec_paris, ...}
  ├─ _subject_vectors_cache: {"france": vec_france, ...}
  └─ Codebook cache: {"france": vec_france, ...}
```

---

## 4. VentriloquistGenerator Path (SLM)

```
┌──────────────────────────────────────────────────────────────────┐
│          VENTRILOQUIST GENERATOR: SLM-BASED GENERATION             │
└──────────────────────────────────────────────────────────────────┘

INPUT: GenerationContext
  ├─ query_text: "What is the capital of France?"
  ├─ fact_answer: "Paris"
  ├─ entities: ["capital", "France"]
  └─ style: StyleType.NEUTRAL

STEP 1: CONSTRUCT PROMPTS (No API cost yet)
  │
  ├─ System Prompt (~20 tokens):
  │  "You are a helpful assistant. Answer questions naturally and fluently
  │   using the provided facts. Be conversational but accurate."
  │
  ├─ User Prompt (~14 tokens):
  │  "Question: What is the capital of France?
  │
  │   Fact: Paris
  │
  │   Answer the question using the fact above..."
  │
  └─ Total input: 34 tokens (Safe, well within 8K window)

STEP 2: API CALL TO NOVITA
  │
  ├─ Model: moonshotai/kimi-k2-thinking (SLM, 8K context)
  ├─ Messages: [system_prompt, user_prompt]
  ├─ max_tokens: 256 (hard limit)
  ├─ temperature: 0.7 (sampling)
  │
  └─ Response from API:
     └─ "The capital of France is Paris."

STEP 3: TOKEN COUNTING (From API response)
  │
  ├─ Input tokens: 34
  ├─ Output tokens: 12 (actual generated)
  └─ Total: 46 tokens this turn

STEP 4: VALIDATION
  │
  ├─ Extract fact words: ["paris"] (words > 2 chars)
  ├─ Check in response: "paris" in "the capital of france is paris"
  ├─ Match ratio: 1/1 = 100% (exceeds 50% threshold) ✓
  │
  └─ PASSED VALIDATION

STEP 5: FORMAT RESULT
  │
  ├─ Convert tokens to GenerationResult
  ├─ Create trace records for each token
  ├─ Compute metrics
  │
  └─ Return: GenerationResult(
       text="The capital of France is Paris.",
       tokens=["The", "capital", "of", "France", "is", "Paris", "."],
       trace=[...],
       metrics=GenerationMetrics(total_tokens=7, ...)
     )

FALLBACK CHAIN (if generation fails):
  ├─ [1] If API error → return None
  ├─ [2] If validation fails (fact not in response) → return None
  ├─ [3] Circuit breaker opens after 3 failures in 10 attempts
  ├─ [4] Switch to ResonantGenerator (HDC)
  ├─ [5] Switch to template patterns
  └─ [6] Fallback template: "I'm not sure how to respond to that."

REASONING PATH (Alternative, for complex queries):
  ├─ Model: zai-org/glm-4.6v (Larger, 128K context)
  ├─ max_tokens: 512 (double fluency limit)
  ├─ temperature: 0.3 (more deterministic)
  ├─ response_format: {"type": "json_object"}
  │
  └─ Returns:
     {
       "reasoning_steps": ["Step 1: ...", "Step 2: ..."],
       "conclusion": "Final answer",
       "verified": true,
       "confidence": 0.9
     }
```

---

## 5. ResonantGenerator Path (HDC, Fallback)

```
┌──────────────────────────────────────────────────────────────────┐
│       RESONANT GENERATOR: HDC-NATIVE (FALLBACK) GENERATION          │
└──────────────────────────────────────────────────────────────────┘

INPUT: GenerationContext
  ├─ thought_vector: 10,000D hypervector
  │  (Bundled: Subject * SUBJECT_ROLE + Verb * VERB_ROLE + Object * OBJECT_ROLE)
  ├─ expected_subject: "France"
  └─ style: StyleType.NEUTRAL

STEP 1: RESONATOR FACTORIZATION
  │
  ├─ Input: thought_vector (compressed meaning)
  ├─ Goal: Extract subject, verb, object components
  │
  ├─ Loop (max 100 iterations):
  │  ├─ [Iteration 1]
  │  │  ├─ Current: random vector
  │  │  ├─ Target: thought_vector
  │  │  ├─ Similarity: 0.23 (low, continue)
  │  │  └─ Update current
  │  │
  │  ├─ [Iteration 2]
  │  │  ├─ Similarity: 0.67 (improving)
  │  │  └─ Update current
  │  │
  │  ├─ [Iteration 5]
  │  │  ├─ Similarity: 0.96 (excellent!)
  │  │  └─ Break (converged, 0.96 > 0.95 threshold)
  │  │
  │  └─ Result: Factorized components
  │     ├─ subject_component ≈ "France"
  │     ├─ verb_component ≈ "is"
  │     └─ object_component ≈ "capital"
  │
  └─ Cost: 5 iterations × 10,000 dims = 50K operations

STEP 2: TARGET ENCODER (Package Constraints)
  │
  ├─ Subject expectation: "France"
  ├─ Verb space: "is", "has", "was"
  ├─ Style: NEUTRAL (no emotional coloring)
  │
  └─ Target tensor: Constraints for generation

STEP 3: TOKEN GENERATION LOOP
  │
  ├─ Max tokens: 10 (hard limit)
  │
  ├─ [Token 1] "The"
  │  ├─ Sample from vocabulary (temperature=0.7)
  │  ├─ Encode: "the" → vec_the
  │  ├─ Compute divergence: similarity(vec_the, target)
  │  ├─ Divergence: 0.75 (ACCEPT)
  │  └─ Add to output
  │
  ├─ [Token 2] "capital"
  │  ├─ Sample candidate
  │  ├─ Divergence: 0.82 (ACCEPT)
  │  └─ Add to output
  │
  ├─ [Token 3] "of"
  │  ├─ Divergence: 0.65 (SOFT - accept with correction)
  │  └─ Add to output (slightly adjusted)
  │
  ├─ [Token 4] "France"
  │  ├─ Divergence: 0.88 (ACCEPT)
  │  └─ Add to output
  │
  ├─ [Token 5] "is"
  │  ├─ Divergence: 0.79 (ACCEPT)
  │  └─ Add to output
  │
  ├─ [Token 6] "Paris"
  │  ├─ Divergence: 0.91 (ACCEPT)
  │  └─ Add to output (high confidence!)
  │
  └─ STOP (Either max 10 tokens or natural boundary)

STEP 4: STYLE MODULATION (Sesame)
  │
  ├─ Current style: NEUTRAL
  ├─ Confidence: 0.85 (high)
  ├─ No disfluency injection needed (>0.35 threshold)
  │
  └─ Output: "The capital of France is Paris"

STEP 5: FORMAT RESULT
  │
  ├─ Text: "The capital of France is Paris"
  ├─ Tokens: ["The", "capital", "of", "France", "is", "Paris"]
  ├─ Trace: 6 generation traces (one per token)
  ├─ Metrics:
  │  ├─ total_tokens: 6
  │  ├─ accepted_first_try: 5
  │  ├─ accepted_with_correction: 1
  │  ├─ average_similarity: 0.81
  │  └─ convergence_iterations: 5
  │
  └─ Return: GenerationResult(...)

COST ANALYSIS:
  ├─ Resonator iterations: 5 × 10K dims = 50K ops
  ├─ Token generation: 6 tokens × 10K dims = 60K ops
  ├─ Style modulation: 6 tokens × 10K dims = 60K ops
  ├─ Total: ~170K floating-point operations
  ├─ Execution time: ~50ms (CPU, optimized)
  │
  └─ SLM cost: 0 API calls (no network latency!)

QUALITY NOTES:
  ├─ Advantage: Bounded hallucination (only uses learned facts)
  ├─ Disadvantage: Robotic output ("The capital of France is Paris")
  ├─ That's why Ventriloquist (SLM) is preferred!
  └─ HDC is fallback for resilience when SLM fails
```

---

## 6. Context Window Management

```
┌──────────────────────────────────────────────────────────────────┐
│           CONTEXT WINDOW: EXPLICIT VS IMPLICIT MANAGEMENT           │
└──────────────────────────────────────────────────────────────────┘

EXPLICIT LIMITS (Code enforces):
├─ VentriloquistGenerator max_tokens: 256 (hard limit)
├─ ResonantGenerator max_tokens: 10 (hard limit)
├─ Resonator max_iterations: 100 (hard limit)
├─ Circuit breaker: 3 failures → 60s cooldown
└─ Generation timeout: Implicit (API-dependent)

IMPLICIT LIMITS (No code enforcement):
├─ SLM context window: 8,000 tokens (Kimi K2)
│  ├─ No checking in VentriloquistGenerator
│  ├─ No warning if input too large
│  └─ API silently truncates if exceeded
│
├─ FactStore capacity: ~100 facts (estimated)
│  ├─ No saturation warning
│  ├─ No early stopping
│  └─ Degrades gracefully but silently
│
├─ Vocabulary size: No explicit limit
│  ├─ Resonance search becomes slow (O(N))
│  └─ No notification to user
│
└─ Session memory: Max 10 conversation turns (configurable)
   ├─ Older turns deleted automatically
   └─ No user notification

WORST-CASE SCENARIO:
┌─────────────────────────────────────────────────┐
│ Multiple complex facts + SLM context overflow    │
├─────────────────────────────────────────────────┤
│ Turn 1: 34 tokens input, 50 tokens output       │
│ Turn 2: 40 tokens input, 80 tokens output       │
│ Turn 3: 200 tokens input (many facts)           │
│ Turn 4: 350 tokens input (accumulating)         │
│ Turn 5: 800 tokens input (EXCEEDS 8K WINDOW?)   │
│                                                  │
│ What happens:                                   │
│ • No warning in code                            │
│ • API may truncate context silently             │
│ • Generation becomes less informed              │
│ • User doesn't know why quality dropped         │
└─────────────────────────────────────────────────┘

REALITY CHECK (Current implementation):
├─ Each SLM call is STATELESS
│  ├─ No conversation history sent to API
│  ├─ Only current turn + fact
│  └─ Previous context NOT included
│
└─ Therefore: Window overflow is UNLIKELY
   ├─ Each turn: ~34 + 256 = 290 tokens max
   ├─ Many turns needed to hit 8K limit
   └─ But possible with LONG FACTS

FACT ENCODING WORST CASE:
├─ Store 100 facts: "Subject --predicate--> Object"
├─ Each fact: ~7 tokens per fact
├─ Total: 700 tokens of facts alone
├─ Plus query: 20 tokens
├─ Plus system: 20 tokens
├─ TOTAL: 740 tokens (well within 8K) ✓
│
└─ But if REASONING PATH activates:
   ├─ Uses GLM-4.6v (128K window, safe)
   └─ Can handle 100+ facts comfortably
```

---

## 7. Memory Saturation Progression

```
┌──────────────────────────────────────────────────────────────────┐
│              FACTSTORE SATURATION OVER TIME                         │
└──────────────────────────────────────────────────────────────────┘

TIMELINE:

[Fact 1-10]: Early phase
├─ Memory: Very clear signal
├─ Noise floor: Minimal
├─ Confidence: 0.98+ (high)
├─ Resonance search: O(10) fast
└─ Status: OPTIMAL

[Fact 11-50]: Middle phase
├─ Memory: Interference patterns forming
├─ Noise floor: Rising (~5%)
├─ Confidence: 0.85-0.95 (good)
├─ Resonance search: O(50) acceptable
└─ Status: HEALTHY

[Fact 51-80]: Degradation phase
├─ Memory: Significant interference
├─ Noise floor: Rising (~15%)
├─ Confidence: 0.70-0.85 (caution zone)
├─ Resonance search: O(80) slower
├─ Queries starting to fail: ~2% error rate
└─ Status: WATCH

[Fact 81-100]: Saturation phase
├─ Memory: High interference
├─ Noise floor: Critical (~25%)
├─ Confidence: 0.60-0.75 (risky)
├─ Resonance search: O(100) slow
├─ Queries failing: ~5% error rate
├─ System accuracy degrading
└─ Status: CRITICAL (No warning message!)

[Fact 101+]: Overflow phase
├─ Memory: COMPLETELY SATURATED
├─ Noise floor: Dominates signal
├─ Confidence: <0.60 (unusable)
├─ Queries: >10% error rate
├─ Suggestion: Use HierarchicalFactStore (FAISS)
└─ Status: FAILURE (No automatic fallback!)

USER EXPERIENCE:
├─ Phase 1-2 (Facts 1-50): "The system is working great!"
├─ Phase 3 (Facts 51-80): "Hmm, sometimes it forgets things"
├─ Phase 4 (Facts 81-100): "The answers are less accurate"
├─ Phase 5 (Facts 101+): "The system is broken"
│
└─ NO WARNINGS AT ANY STAGE (Problem!)

RECOMMENDED FIX:
├─ At 80% capacity: WARN "Approaching saturation"
├─ At 100% capacity: WARN "At capacity, using FAISS"
├─ At overflow: WARN "Consider pruning facts"
└─ Current: SILENT (Bad UX)

CAPACITY EXPANSION (HierarchicalFactStore):
├─ Hot layer: FactStore (~100 facts, O(N) resonance)
├─ Cold layer: FAISS index (unlimited, O(log N) search)
│
├─ Benefits:
│  ├─ Unlimited fact storage
│  ├─ Logarithmic search time
│  └─ Graceful transition (hot → cold)
│
└─ Cost:
   ├─ Cold facts require FAISS library
   ├─ Disk I/O overhead
   └─ Slightly higher latency
```

---

## 8. Circuit Breaker State Machine

```
┌──────────────────────────────────────────────────────────────────┐
│         CIRCUIT BREAKER: FAILURE DETECTION & RECOVERY               │
└──────────────────────────────────────────────────────────────────┘

CONFIGURATION:
├─ Failure threshold: 3 failures
├─ Window size: 10 attempts
├─ Cooldown: 60 seconds
└─ Purpose: Prevent cascading failures

STATE DIAGRAM:

                    [CLOSED - Normal]
                           │
                           │ (Attempt generation)
                           ▼
                ┌─────────────────────┐
                │   Success? (Y/N)    │
                └──────┬──────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
        ✓ YES                      │ ✗ NO
           │              (Record failure)
           │                       │
           ▼                       ▼
    [Still CLOSED]          [Check count]
                                   │
                        ┌──────────┴──────────┐
                        │                     │
                    3 failures         <3 failures
                    in 10 attempts?    in 10 attempts?
                        │                     │
                        ▼                     ▼
                 [OPEN - Cooldown]     [Stay CLOSED]
                    (60 seconds)           (retry)
                        │
                        │ (Wait 60 seconds)
                        ▼
                   [HALF_OPEN]
                    (test mode)
                        │
                   (Try generation)
                        │
              ┌─────────┴─────────┐
              │                   │
           Success             Failure
              │                   │
              ▼                   ▼
           [CLOSED]            [OPEN]
         (resume normal)    (another 60s wait)


BEHAVIOR AT EACH STATE:

[CLOSED]: Normal operation
├─ Always attempt generation
├─ Record success/failure
├─ If 3 failures: transition to OPEN
└─ Return: GeneratedResult or None

[OPEN]: Blocked (recovering)
├─ Skip generation attempts
├─ Return: None (force fallback)
├─ Log: "Generation disabled (cooldown)"
├─ Duration: 60 seconds
└─ Transition: After 60s → HALF_OPEN

[HALF_OPEN]: Recovery test
├─ Try one generation attempt
├─ If success: transition to CLOSED
├─ If failure: stay OPEN (reset timer)
└─ Purpose: Detect if service recovered

FAILURE SCENARIOS:

Scenario 1: Transient API error (network hiccup)
├─ [CLOSED] Fail #1 → retry
├─ [CLOSED] Fail #2 → retry
├─ [CLOSED] Success #3 → stay CLOSED
├─ Result: System recovers (GOOD)
└─ User impact: Minimal delay

Scenario 2: Cascading failures (API down)
├─ [CLOSED] Fail #1 → count=1/10
├─ [CLOSED] Fail #2 → count=2/10
├─ [CLOSED] Fail #3 → TRANSITION TO OPEN
├─ [OPEN] Attempts blocked (return None)
├─ [OPEN] Wait 60 seconds
├─ [HALF_OPEN] Try once
├─ [HALF_OPEN] Success → back to CLOSED
└─ User impact: 60s delay, then recovery

Scenario 3: Persistent failures (credentials invalid)
├─ [CLOSED] Fail → OPEN (after 3 attempts)
├─ [OPEN] Wait 60 seconds
├─ [HALF_OPEN] Fail → stay OPEN
├─ [OPEN] Wait 60 seconds again
├─ [HALF_OPEN] Fail → stay OPEN (loop)
└─ User impact: System falls back to templates permanently

FALLBACK CHAIN:
├─ Generation fails 3× → Circuit opens
├─ Return: None
├─ ResponseSelector.select() catches this
├─ Tries next option:
│  ├─ ResonantGenerator (HDC)
│  ├─ ResponseCorpus (learned responses)
│  ├─ ResponsePatternStore (templates)
│  └─ Fallback pattern ("I'm not sure...")
└─ User always gets SOME response

METRICS TRACKED:
├─ failure_count: Current failures in window
├─ window_size: 10 attempts
├─ last_failure_time: Timestamp of last failure
├─ state: CLOSED, OPEN, or HALF_OPEN
└─ cooldown_start: When cooldown began
```

---

## 9. Fact Encoding: String vs Vector

```
┌──────────────────────────────────────────────────────────────────┐
│        FACT ENCODING: REPRESENTATION AND TOKEN COST                 │
└──────────────────────────────────────────────────────────────────┘

FACT: "France --capital--> Paris"

INTERNAL REPRESENTATION (HDC):
├─ subject: encode("france") → 10,000D vector
├─ predicate: encode("capital") → 10,000D vector
├─ object: encode("paris") → 10,000D vector
├─ key: bind(subject_vec, predicate_vec) → 10,000D vector
├─ stored in: MemoryTrace (superposed with other facts)
│
└─ COST: 0 SLM tokens (internal, not serialized)

STRING REPRESENTATION (For SLM):
├─ Format: "France --capital--> Paris"
├─ Tokens: ["France", "--capital-->", "Paris"] ≈ 7 tokens
│
└─ COST: 7 SLM tokens when passed to API

METADATA REPRESENTATION (For FactStore):
├─ subject: "France" (string)
├─ predicate: "capital" (string)
├─ object: "Paris" (string)
├─ confidence: 1.0 (float)
├─ source: "user" (string)
├─ timestamp: 2025-12-12T10:30:00 (datetime)
├─ surprise_score: 0.85 (float)
│
└─ COST: ~200 bytes in Python object

ENCODING FLOW:

Add fact to FactStore:
├─ HDC encode: hash → seed → torch.randn(10K) [~40KB]
├─ Store in trace: bind + bundle operations [no new memory]
├─ Add to metadata: Fact object [200 bytes]
└─ Total memory: +40KB to trace, +200 bytes to metadata

Query fact from FactStore:
├─ Look up in exact_index [O(1)]
├─ Return string "Paris"
└─ Cost: 0 SLM tokens at this point

Serialize fact to SLM:
├─ Format as string: "France --capital--> Paris"
├─ Count tokens: 7 tokens
└─ Cost: 7 SLM tokens

MULTIPLE FACTS EXAMPLE:

Facts added:
1. "France --capital--> Paris"
2. "France --area--> 551000"
3. "Paris --population--> 2.2M"
4. "Paris --country--> France"
5. "England --capital--> London"

When querying: "Who is the capital of France?"
├─ FactStore returns: "Paris" [0 tokens]
│
When generating response:
├─ System: "You are helpful..." [20 tokens]
├─ User: "Question: Who is the capital of France?" [8 tokens]
├─ Fact: "Paris" [1 token]
├─ Total: 29 tokens

But if using reasoning chain:
├─ System: [100 tokens]
├─ Query: [8 tokens]
├─ Facts (serialized):
│  - "France --capital--> Paris" [7 tokens]
│  - "Paris --population--> 2.2M" [7 tokens]
│  - ... (all 5 facts) [35 tokens total]
├─ Total: 143 tokens (still safe!)

With 100 facts:
├─ Serialized facts: 100 × 7 tokens = 700 tokens
├─ Plus overhead: 150 tokens
├─ Total: 850 tokens (safe for 8K window, borderline)
├─ Plus generation: 256 tokens
├─ Grand total: 1,106 tokens (FINE)

WARNING: If facts are very detailed:
├─ "France --capital--> Paris is the largest city in France..."
├─ Per fact: 25 tokens (vs 7)
├─ 100 facts: 2,500 tokens
├─ Plus system + query: 150 tokens
├─ TOTAL: 2,650 tokens (still safe, but getting tight)
├─ Plus generation: 256 tokens
├─ GRAND TOTAL: 2,906 tokens (OK, but close to limit)

CACHE OPTIMIZATION:

First time encoding "France":
├─ Hash "france" → seed
├─ torch.randn(10K) → vector
├─ Store in Codebook._cache
└─ Cost: ~10,000 FLOPs

Second time encoding "France":
├─ Look up in Codebook._cache
└─ Cost: O(1) dictionary lookup

Typical conversation:
├─ Facts 1-5: First mention → encode once
├─ Facts 6-50: Mostly cache hits
├─ Encoding cost: Minimal after warm-up
└─ Overall: ~5-10% of total cost
```

---

## 10. API Token Usage Example

```
┌──────────────────────────────────────────────────────────────────┐
│            REAL CONVERSATION WITH TOKEN BREAKDOWN                   │
└──────────────────────────────────────────────────────────────────┘

USER CONVERSATION:
┌────────────────────────────────────────────────────────────────┐
│ Turn 1                                                           │
├────────────────────────────────────────────────────────────────┤
│ User: "What is the capital of France?"                          │
│                                                                  │
│ System prompt: "You are a helpful assistant..."  ← 20 tokens   │
│ User message: "What is the capital of France?"   ← 8 tokens    │
│ Fact: "Paris"                                    ← 1 token     │
│ ─────────────────────────────────────────────────────────────  │
│ Input total: 29 tokens                                          │
│                                                                  │
│ Response: "The capital of France is Paris."     ← 11 tokens    │
│ ─────────────────────────────────────────────────────────────  │
│ TURN 1 TOTAL: 40 tokens                                         │
│ Cumulative: 40 tokens / 8,000 = 0.5% of window                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Turn 2                                                           │
├────────────────────────────────────────────────────────────────┤
│ User: "What about Germany?"                                     │
│                                                                  │
│ System prompt: [RESTARTED - new turn, no history]  ← 20 tokens│
│ User message: "What about Germany?"               ← 5 tokens   │
│ Fact: "Berlin" [from FactStore query]             ← 1 token    │
│ ─────────────────────────────────────────────────────────────  │
│ Input total: 26 tokens                                          │
│                                                                  │
│ Response: "The capital of Germany is Berlin."    ← 11 tokens   │
│ ─────────────────────────────────────────────────────────────  │
│ TURN 2 TOTAL: 37 tokens                                         │
│ Cumulative: 77 tokens / 8,000 = 1.0% of window                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Turn 3                                                           │
├────────────────────────────────────────────────────────────────┤
│ User: "Tell me about the capitals of European countries"        │
│                                                                  │
│ System prompt: [NEW TURN]                         ← 20 tokens   │
│ User message: [Longer question]                   ← 15 tokens   │
│ Facts: [Multiple retrieved]                                      │
│   - "France --capital--> Paris"                  ← 7 tokens    │
│   - "Germany --capital--> Berlin"                ← 7 tokens    │
│   - "Italy --capital--> Rome"                    ← 7 tokens    │
│   - "Spain --capital--> Madrid"                  ← 7 tokens    │
│ ─────────────────────────────────────────────────────────────  │
│ Input total: 63 tokens                                          │
│                                                                  │
│ Response: [Longer response]                      ← 50 tokens   │
│ ─────────────────────────────────────────────────────────────  │
│ TURN 3 TOTAL: 113 tokens                                        │
│ Cumulative: 190 tokens / 8,000 = 2.4% of window               │
└────────────────────────────────────────────────────────────────┘

CUMULATIVE USAGE ACROSS 3 TURNS:
├─ Total API tokens: 190 tokens
├─ Window available: 8,000 tokens
├─ Remaining: 7,810 tokens
├─ Estimated turnsuntil limit: ~420 more turns!
└─ Status: VERY SAFE

KEY INSIGHT: EACH TURN IS STATELESS
├─ Turn 2 doesn't see Turn 1's history
├─ Only current question + facts matter
├─ This PREVENTS context explosion
├─ Therefore: Token overflow is UNLIKELY in practice

EXCEPTION: Reasoning chain path
├─ Uses GLM-4.6v (128K context)
├─ Can handle 100+ facts
├─ max_tokens=512
└─ Much safer!
```

---

## Summary: Token Management Architecture

| Layer | Token Type | Limit | Overflow Behavior |
|-------|-----------|-------|------------------|
| **HDC (FactStore)** | Vector (10K dims) | ~100 facts | Graceful degradation (noise rises) |
| **ResonantGenerator** | HDC tokens | 10 tokens max | Hard stop at 10 |
| **VentriloquistGenerator** | SLM tokens | 256 tokens max | API truncation or error |
| **SLM API (Novita)** | Natural language | 8,000 tokens (estimated) | Silent truncation or error |
| **Resonator** | Vector iterations | 100 iterations | Hard stop, use partial result |
| **Circuit Breaker** | Failure count | 3 failures in 10 | Cooldown 60 seconds |

**Critical Finding**: No explicit overflow handling exists. System relies on:
1. Fixed max_tokens parameters (hard stops)
2. API-level truncation (silent)
3. Graceful degradation (hope for the best)


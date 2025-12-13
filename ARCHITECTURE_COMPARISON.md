# Kent Hologram ARC-AGI-2: Architecture Comparison

**Visual and textual comparison of current vs. desired self-referential architecture**

---

## 1. Component Interaction Matrices

### Current State: Disconnected Components

```
                    Solver  Resonator  Verifier  Metacog  Iterator
                    ──────────────────────────────────────────────
Solver              [X]      →          →        ✗        →
Resonator           ←        [X]        ✗        ✗        ✗
Verifier            ←        ✗          [X]      ✗        ✗
MetacogLoop         ✗        ✗          ✗        [X]      ✗
IterativeSolver     ←        →          ✗        ✗        [X]

Legend:
  →   One-way call
  ←   One-way call (reverse)
  ✗   No connection
  [X] Self
```

**Key observation:** MetacogLoop is isolated (all ✗ in row/column except self).

### Desired State: Connected Components

```
                    Solver  Resonator  Verifier  Metacog  Iterator
                    ──────────────────────────────────────────────
Solver              [X]      ⇄          ⇄        ⇄        ⇄
Resonator           ⇄        [X]        ⇄        ⇄        ⇄
Verifier            ⇄        ✗          [X]      ←        ⇄
MetacogLoop         ⇄        ←          ←        [X]      ⇄
IterativeSolver     ⇄        ⇄          ⇄        ⇄        [X]

Legend:
  ⇄   Bidirectional (feedback loop)
  →   One-way call
  ←   One-way call (reverse)
  ✗   No connection needed
  [X] Self
```

**Key improvement:** MetacogLoop has bidirectional arrows (feedback).

---

## 2. Control Flow Diagrams

### Current Control Flow (Linear)

```
START
  │
  ├─ [Skill Memory] Hit? → Return cached (90% of case)
  │   Miss ↓
  │
  ├─ [Strategy Selection]
  │
  ├─ [Observation Phase]
  │   Input: training pairs
  │   Processing: detect objects, encode transformations
  │   Output: observation_bundle
  │
  ├─ [Proposal Phase]
  │   Input: observation_bundle (STATIC)
  │   Processing: resonator.resonate_topk()
  │   Output: List[TransformResult]
  │
  ├─ [Verification Phase]
  │   Input: candidates
  │   Processing: verify_candidates()
  │   Output: TransformResult OR None
  │
  └─ [Decision]
      │
      ├─ [IF verified]
      │   ├─ execute()
      │   ├─ store_skill()
      │   └─ return SolverResult(output=grid, confidence=1.0)
      │
      └─ [IF NOT verified]
          └─ return SolverResult(output=None, confidence=0.0)

END

Properties:
  • Single execution path
  • No loops or retries
  • No feedback
  • Deterministic
```

### Desired Control Flow (Feedback Loop)

```
START
  │
  ├─ [Skill Memory] Hit? → Return cached
  │   Miss ↓
  │
  ├─ [Initialize Metacognitive State]
  │   state.mood = NEUTRAL
  │   state.confidence_history = []
  │
  └─ [RETRY LOOP] (attempt = 0; attempt < max_retries + 1)
      │
      ├─ [Self-Observation Phase] ← NEW
      │   ├─ Read: state.mood
      │   ├─ Read: state.confidence_trend()
      │   └─ Decide: should_broaden_search?
      │
      ├─ [Observation Phase]
      │   Input: training pairs + self_state
      │   Processing: detect objects, encode transformations
      │   Output: observation_bundle (possibly modulated)
      │
      ├─ [Adaptation Phase] ← NEW
      │   IF mood == CONFUSED:
      │     modulated_obs = bundle(observation_bundle, self_vector)
      │     k, slot_k = 2x normal
      │   ELSE:
      │     modulated_obs = observation_bundle
      │     k, slot_k = normal
      │
      ├─ [Proposal Phase]
      │   Input: modulated_obs
      │   Processing: resonator.resonate_topk()
      │   Output: List[TransformResult]
      │
      ├─ [Verification Phase]
      │   Input: candidates
      │   Processing: verify_candidates()
      │   Output: TransformResult OR None
      │
      ├─ [Self-Observation Update] ← NEW
      │   IF verified:
      │     state.update_from_confidence(1.0)
      │     Break loop → [EXECUTE]
      │   ELSE:
      │     state.update_from_confidence(0.0)
      │     attempt++
      │     Continue loop ← FEEDBACK
      │
      └─ [Retry Decision] ← NEW
          IF attempt >= max_retries:
            Break loop → [REFUSE]
          ELSE:
            Continue loop (state has been updated)

  ├─ [EXECUTE]
  │   ├─ execute(verified_transform)
  │   ├─ store_skill()
  │   └─ return SolverResult(
  │       output=grid,
  │       confidence=1.0,
  │       message=f"Solved on attempt {attempt+1}"
  │     )
  │
  └─ [REFUSE]
      └─ return SolverResult(
          output=None,
          confidence=0.0,
          message=f"Failed after {max_retries+1} attempts"
        )

END

Properties:
  • Feedback loops (observation → action → observation)
  • Multiple attempts possible
  • Adaptation based on state
  • Self-aware
```

---

## 3. Data Structure Transformation

### Current: Deterministic Transformation

```
INPUT:
  ARCTask {
    training: List[TrainingPair],
    test_input: Grid,
  }

PROCESSING:
  observation_bundle = ENCODE(training_pairs)
    └─ Always same result for same input
    └─ No modulation possible
    └─ No feedback from verification

PROPOSAL:
  candidates = RESONATE_TOPK(observation_bundle)
    └─ Deterministic (same obs → same candidates)
    └─ Limited to observation information

VERIFICATION:
  verified = VERIFY(candidates, training_pairs)
    └─ Returns: TransformResult | None
    └─ No score information
    └─ No recovery information

OUTPUT:
  SolverResult {
    output: Grid | None,
    confidence: float,
    message: str,
  }
    └─ No indication of retries
    └─ No internal state visible
    └─ No adaptation visible
```

### Desired: Adaptive Transformation with Feedback

```
INPUT:
  ARCTask {
    training: List[TrainingPair],
    test_input: Grid,
  }

MetacognitiveState {
    mood: MetacognitiveMood,      ← Driven by feedback
    self_vector: Tensor,          ← Updated each attempt
    confidence_history: List[float],  ← Trend tracking
  }

PROCESSING (ATTEMPT 1):
  [Self-Observe]
    confidence = None (first attempt)
    mood = NEUTRAL
    should_broaden = False

  [Observation]
    observation = ENCODE(training_pairs)

  [Modulation]
    modulated_obs = observation  (no change for NEUTRAL)

  [Proposal]
    candidates = RESONATE_TOPK(modulated_obs, k=20, slot_k=5)

  [Verification]
    verified = VERIFY(candidates)
    └─ Result: None (no candidates passed)

  [Self-Update]
    state.update_from_confidence(0.0)
    └─ mood: NEUTRAL → CONFUSED
    └─ self_vector += curiosity_signal

PROCESSING (ATTEMPT 2):
  [Self-Observe]
    confidence_trend = decreasing
    mood = CONFUSED
    should_broaden = True

  [Observation]
    observation = ENCODE(training_pairs)

  [Modulation]
    modulated_obs = BUNDLE(observation, self_vector)  ← USES FEEDBACK

  [Proposal]
    candidates = RESONATE_TOPK(modulated_obs, k=40, slot_k=7)  ← BROADER

  [Verification]
    verified = VERIFY(candidates)
    └─ Result: TransformResult (1 passed!)

  [Self-Update]
    state.update_from_confidence(1.0)
    └─ mood: CONFUSED → CONFIDENT
    └─ Break loop

OUTPUT:
  SolverResult {
    output: Grid,           ← Successful
    confidence: 1.0,        ← Verified
    message: "Verified (attempt 2, mood=confident): translate(...)",
  }
    └─ Shows adaptation happened
    └─ Shows internal state
    └─ Explains retry behavior
```

---

## 4. State Machine: Metacognitive Mood Progression

### Current State: Isolated

```
MetacognitiveState exists, but:
  • Never created in solver
  • Never updated in solver
  • Never used to modulate observation
  • Mood transitions never visible to solver

Mood state diagram:
  (never instantiated in solver context)
```

### Desired State: Integrated

```
                         NEUTRAL (initial)
                            │
                    ┌───────┴───────┐
                    │               │
              confidence            confidence
              >= 0.8                ≤ 0.2
                    │               │
                    ▼               ▼
              CONFIDENT        CONFUSED
                  ▲                 │
                  │                 │
              attempt              inject
              succeeds             CURIOSITY
                  │                 │
                  └────┬────────────┘
                       │
                  next attempt
                  (modified search)

Transitions:
  NEUTRAL → CONFIDENT  (if first attempt succeeds)
  NEUTRAL → CONFUSED   (if first attempt fails)
  CONFUSED → CONFIDENT (if retry succeeds)
  CONFUSED → CONFUSED  (if retry fails again)

Each transition:
  • Updates mood enum
  • Updates self_vector
  • Injects signals (CONFIDENCE, CURIOSITY, ANXIETY)
  • Available for NEXT attempt to read
```

---

## 5. Resonator Input Modulation

### Current: Static Input

```
observation_bundle
    │
    ├─ [obs1: rotate object 1]
    ├─ [obs2: rotate object 2]
    └─ [obs3: rotate object 3]
       │
       ▼
    [Resonator ALS]
    ├─ Initialize: a=superpose(all_actions)
    ├─ Iterate: solve_for_slot() x 3
    ├─ Converge or oscillate
    └─ Output: best_action, best_target, best_modifier

Result: Single best factorization

Properties:
  • Input always same for same observations
  • No external influence possible
  • No way to bias search toward different solution
  • Dead-end if first solution is wrong
```

### Desired: Adaptive Input

```
observation_bundle = ENCODE(training_pairs)
    │
    ├─ [obs1: rotate object 1]
    ├─ [obs2: rotate object 2]
    └─ [obs3: rotate object 3]

IF state.mood == CONFUSED:
    modulated_bundle = BUNDLE(
        observation_bundle,
        state.self_vector,        ← Curiosity signal
    )
    └─ Adds new "signal" to the observation
    └─ Changes energy landscape for ALS
    └─ Forces different convergence region
ELSE:
    modulated_bundle = observation_bundle

    modulated_bundle
        │
        ├─ [obs1: rotate object 1]
        ├─ [obs2: rotate object 2]
        ├─ [obs3: rotate object 3]
        └─ [curious: try alternative factorizations]  ← NEW!
           │
           ▼
        [Resonator ALS]
        ├─ Initialize: a=superpose(all_actions)
        ├─ Iterate: solve_for_slot() x 3
        ├─ Converge or oscillate
        ├─ OUTPUT: different best factorization
        │   (because energy landscape changed)
        └─ Also generate top-5 per slot (not just 1)

Results: Different candidates, potentially including solution

Properties:
  • Input changes based on state
  • Curiosity signal biases toward exploration
  • Different attempts search different regions
  • Can escape local minima
```

---

## 6. Verification Feedback Richness

### Current: Boolean Return

```
verify_candidates(candidates: List[TransformResult], training_pairs: List[TrainingPair])
    → TransformResult | None

Result:
  IF found first passing candidate:
    return TransformResult {...}
  ELSE:
    return None

Problem:
  • None = "nothing worked"
  • No information about how close candidates were
  • No scores to guide retry strategy
  • Can't say "3 of 5 candidates passed 3/4 training pairs"
```

### Desired: Rich Feedback

```
verify_candidates_with_feedback(candidates, training_pairs)
    → VerificationFeedback

VerificationFeedback {
    passed: bool,                     # Any candidate passed all?
    first_passing_candidate: Result | None,  # First success
    closest_candidate: Result,        # Best near-miss
    closest_score: float,             # How close? (0.0-1.0)
    passed_count: int,                # How many candidates passed some pairs?
    scores: List[float],              # Score per candidate
}

Result:
  {
    passed: False,
    closest_candidate: TransformResult(action="rotate", ...),
    closest_score: 0.75,  # Passed 3 of 4 training pairs
    passed_count: 3,
    scores: [0.75, 0.50, 0.25, 0.0, ...],
  }

Benefit:
  • Can decide: "Close enough to retry with broader search?"
  • Can log: "Closest candidate passed 3/4 pairs"
  • Can guide: "Focus on the 4th pair that failed"
```

---

## 7. Message Evolution Example

### Current Messages

```
Scenario 1 (Success):
  "Solved: rotate(all_objects, 90_degrees)"

Scenario 2 (Failure):
  "None of 20 candidates passed verification"

Scenario 3 (Skill Cache):
  "Retrieved from skill memory (conf=0.92)"

Scenario 4 (No Observations):
  "No valid observations from training pairs"
```

### Desired Messages

```
Scenario 1 (Success on first try):
  "Solved (attempt 1, mood=confident): rotate(all_objects, 90_degrees)"

Scenario 2 (Success on retry):
  "Solved (attempt 2 of 2, mood=confident): translate(all_objects, up)"
  └─ Shows: took 2 attempts, mood improved

Scenario 3 (Failure after retries):
  "Failed after 2 attempts (final mood=confused). "
  "Closest: 0.75 score (passed 3/4 training pairs). "
  "Candidates: 20→40 (search broadened on retry)"
  └─ Shows: retry happened, why, how close we got

Scenario 4 (No Observations):
  "No valid observations from training pairs (tried 1 attempt)"

Scenario 5 (Skill Cache):
  "Retrieved from skill memory (conf=0.92, cached from 3 attempts ago)"
```

---

## 8. Performance Impact Analysis

### Memory

```
Current:
  • No persistent state between solve() calls (except skill cache)
  • Task metadata: minimal

Added:
  • MetacognitiveState per solver instance
    └─ self_vector: ~10KB (for 10K dimensions)
    └─ mood enum: 1 byte
    └─ confidence_history: 80 bytes (10 floats)
  • Additional log entries: ~1KB per failed solve

Total additional memory: ~10-15KB per solver instance (negligible)
```

### Compute

```
Current flow:
  1. Observe training pairs: ~10ms
  2. Resonate top-20: ~50ms
  3. Verify all: ~100ms
  Total: ~160ms

With retries (if needed):
  1. Observe training pairs: ~10ms
  2. [Attempt 1]
     a. Resonate top-20: ~50ms
     b. Verify all: ~100ms
  3. [Attempt 2]  (only if first failed)
     a. Resonate top-40: ~80ms
     b. Verify all: ~200ms
  Total if 2 attempts: ~440ms

Ratio: 2.75x slower IF retries needed

BUT:
  • Most tasks pass on first try (skill cache hits)
  • Retries only happen on hard tasks
  • Better solution worth the latency
```

### Network (if distributed)

```
No additional network overhead:
  • All computation stays local
  • No need to communicate state to other nodes
  • Inference remains single-node
```

---

## 9. Implementation Complexity

### Current Complexity: LOW

```
solve(task)
  └─ Linear pipeline
  └─ 5 main branches (skill cache, search_verify, iterative, resonator, fallback)
  └─ Cyclomatic complexity: 3
  └─ Dependencies: solver → detector, encoder, resonator, executor, verifier
  └─ No cyclic dependencies
  └─ Easy to reason about

LOC: ~300 (solver.py main logic)
```

### With Metacog Integration: MEDIUM

```
solve(task)
  └─ Initialize metacog_state
  └─ Choose strategy (same as before)
  │
  ├─ search_verify (original)
  │   └─ _solve_search_verify(): ~70 LOC
  │
  ├─ search_verify_with_metacog (NEW)
  │   ├─ RETRY LOOP: ~100 LOC
  │   ├─ _modulate_observation(): ~15 LOC
  │   └─ _update_mood(): ~5 LOC (delegates to state)
  │
  └─ iterative_with_metacog (NEW)
      ├─ Integration: ~30 LOC
      └─ Uses similar retry pattern

Additional LOC: ~150
Total LOC: ~450 (+50%)

Cyclomatic complexity: 5-6
Dependencies: solver → metacog_state (NEW), everything else same
No cyclic dependencies (metacog is unidirectional input)

Complexity added: MINIMAL
  • No new algorithm changes
  • Just wraps existing logic
  • Can be disabled with single flag
  • No breaking changes
```

---

## 10. Integration Effort Estimate

```
Activity                      Time    Risk   Value
─────────────────────────────────────────────────
1. Add MetacogState to Solver  30min  LOW    HIGH
2. Implement retry loop        1hr    LOW    HIGH
3. Implement modulation        45min  LOW    MEDIUM
4. Test retry behavior         1hr    MED    HIGH
5. Test modulation             1hr    MED    MEDIUM
6. IterativeSolver integration 1.5hr  MED    MEDIUM
7. Documentation               1hr    LOW    MEDIUM
8. Bug fixes / iteration       1-2hr  MED    MEDIUM
─────────────────────────────────────────────
TOTAL                          7-9hr  LOW    HIGH

Critical path: Steps 1-2 (1.5 hours) enables "hologram talks to itself"
Phase 2 (steps 3-5): 3 hours for modulation
Phase 3 (step 6): 1.5 hours for iterative solver
```

---

## Summary Table

| Aspect | Current | Desired | Gap |
|--------|---------|---------|-----|
| **Integration** | Disconnected | Connected | CRITICAL |
| **Feedback** | None | Retry loops | CRITICAL |
| **Adaptation** | None | Mood-driven | CRITICAL |
| **Modulation** | Static | Dynamic | MEDIUM |
| **Verification** | Boolean | Rich feedback | MEDIUM |
| **Logging** | Basic | Detailed | LOW |
| **Complexity** | LOW | MEDIUM | LOW |
| **Performance** | ~160ms | ~160-440ms | LOW |
| **Memory** | Minimal | +15KB | LOW |
| **Implementation** | Complete | 70% complete | Need 7-9 hrs |

---

**Note:** All diagrams can be rendered in Mermaid or drawn as ASCII flow charts. The key takeaway is that the current architecture is **component-based** (good structure) but needs **integration** (missing orchestration) to achieve self-referential reasoning.

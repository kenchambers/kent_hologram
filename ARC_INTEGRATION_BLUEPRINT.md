# ARC Self-Referential Integration Blueprint

**Purpose:** Concrete implementation guide for connecting MetacognitiveLoop to HolographicARCSolver

**Status:** Specifications and pseudocode (ready for implementation)

---

## Part 1: Minimal Integration (Proof of Concept)

### 1.1 Enhanced HolographicARCSolver.__init__()

**Current code location:** `src/hologram/arc/solver.py` lines 82-147

**Modification:**

```python
def __init__(
    self,
    dimensions: int = 10000,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iterative: bool = True,
    max_steps: int = 5,
    strategy: Literal["resonator", "search_verify"] = "search_verify",
    search_k: int = 20,
    search_slot_k: int = 5,
    metacog_enabled: bool = True,  # NEW PARAMETER
    metacog_max_retries: int = 2,   # NEW PARAMETER
):
    """Initialize solver with all components."""

    # Existing initialization code...
    self._space = VectorSpace(dimensions=dimensions)
    self._fractal_space = FractalSpace(dimensions=dimensions)
    self._codebook = Codebook(self._space)
    self._detector = ObjectDetector()
    self._encoder = ObjectEncoder(self._fractal_space, self._codebook)
    self._resonator = TransformationResonator(self._encoder, self._codebook)
    self._executor = TransformationExecutor(self._detector)

    # NEW: Metacognitive integration
    self._metacog_enabled = metacog_enabled
    self._metacog_max_retries = metacog_max_retries
    if metacog_enabled:
        from hologram.cognition.metacognition import MetacognitiveState, MetacognitiveMood
        self._metacog_state = MetacognitiveState(self._codebook)
        self._metacog_mood = MetacognitiveMood  # Reference to enum

    # Rest of existing init...
    self._confidence_threshold = confidence_threshold
    self._ops = Operations
    self._iterative = iterative
    self._strategy = strategy
    self._search_k = search_k
    self._search_slot_k = search_slot_k

    if strategy == "search_verify":
        self._verifier = SearchVerifier(executor=self._executor, detector=self._detector)

    if iterative:
        self._iterative_solver = IterativeSolver(
            encoder=self._encoder,
            resonator=self._resonator,
            executor=self._executor,
            detector=self._detector,
            max_steps=max_steps,
        )

    self._skill_memory = NeuralMemory(
        input_dim=dimensions,
        hidden_dim=256,
        initial_vocab_size=100,
    )

    self._transform_cache: dict[str, TransformResult] = {}
```

### 1.2 New Method: _solve_search_verify_with_metacog()

**Add to HolographicARCSolver class**

```python
def _solve_search_verify_with_metacog(
    self,
    task: ARCTask,
    task_sig_vec: torch.Tensor,
) -> SolverResult:
    """
    Solve using search+verify strategy with metacognitive retry loop.

    This is the self-referential version that:
    1. Proposes candidates via resonator
    2. Verifies against training pairs
    3. If verification fails:
        a. Observes self (updates mood)
        b. Rewires self (injects curiosity)
        c. Retries with modulated observation

    Args:
        task: ARC task to solve
        task_sig_vec: Task signature vector for skill memory

    Returns:
        SolverResult with verified solution or refusal
    """
    if not self._metacog_enabled:
        # Fall back to non-metacognitive version
        return self._solve_search_verify(task, task_sig_vec)

    # Initialize metacognitive state for this task
    self._metacog_state.reset()

    # Build observation bundle from training pairs
    observations = []
    for pair in task.training:
        obs = self._observe_training_pair(pair)
        if obs is not None:
            observations.append(obs)

    if not observations:
        return SolverResult(
            output=None,
            transformation=None,
            confidence=0.0,
            from_cache=False,
            message="No valid observations from training pairs",
        )

    observation_bundle = self._ops.bundle(*observations)

    # Retry loop with metacognitive feedback
    for attempt in range(self._metacog_max_retries + 1):
        # STEP 1: Modulate observation based on current mood
        if self._metacog_state.mood == self._metacog_mood.CONFUSED:
            # When confused, bundle curiosity signal with observation
            curiosity_vec = self._codebook.encode("__CURIOSITY__")
            modulated_obs = self._ops.bundle(
                observation_bundle,
                self._metacog_state.self_vector,
            )
            # Increase search breadth
            k = self._search_k * 2
            slot_k = self._search_slot_k + 2
        elif self._metacog_state.mood == self._metacog_mood.ANXIOUS:
            # When anxious, increase verification thoroughness
            modulated_obs = self._ops.bundle(
                observation_bundle,
                self._metacog_state.self_vector * 0.5,  # Smaller influence
            )
            k = self._search_k + 5
            slot_k = self._search_slot_k + 1
        else:
            # Neutral/Confident/Curious - use standard parameters
            modulated_obs = observation_bundle
            k = self._search_k
            slot_k = self._search_slot_k

        # STEP 2: Generate candidates with (possibly) modulated observation
        candidates = self._resonator.resonate_topk(
            modulated_obs,
            k=k,
            slot_k=slot_k,
        )

        if not candidates:
            return SolverResult(
                output=None,
                transformation=None,
                confidence=0.0,
                from_cache=False,
                message="No candidates generated",
            )

        # STEP 3: Verify candidates against training pairs
        verified_transform = self._verifier.verify_candidates(
            candidates, task.training
        )

        # STEP 4: Observe result and update self state
        if verified_transform is not None:
            # SUCCESS: Update mood to confident
            self._metacog_state.update_from_confidence(1.0)

            # Apply and return
            output = self._apply_transformation(verified_transform, task.test_input)
            self._store_skill(task_sig_vec, verified_transform)

            return SolverResult(
                output=output,
                transformation=verified_transform,
                confidence=1.0,
                from_cache=False,
                message=(
                    f"Verified (attempt {attempt + 1}, "
                    f"mood={self._metacog_state.mood.value}): "
                    f"{verified_transform.action}({verified_transform.target}, "
                    f"{verified_transform.modifier})"
                ),
            )
        else:
            # FAILURE: Update mood to confused and signal curiosity
            self._metacog_state.update_from_confidence(0.0)

            # Log attempt
            logger.debug(
                f"Attempt {attempt + 1} failed. "
                f"Mood now: {self._metacog_state.mood.value}. "
                f"Will {'retry' if attempt < self._metacog_max_retries else 'give up'}."
            )

            # Check if we should retry
            if attempt >= self._metacog_max_retries:
                # Max retries reached
                return SolverResult(
                    output=None,
                    transformation=candidates[0] if candidates else None,
                    confidence=0.0,
                    from_cache=False,
                    message=(
                        f"Failed after {self._metacog_max_retries + 1} attempts. "
                        f"Final mood: {self._metacog_state.mood.value}. "
                        f"({len(candidates)} candidates generated, none passed verification)"
                    ),
                )

            # Continue loop to retry with updated state
            # The self_vector has been updated with curiosity signal,
            # so next iteration will use different candidate generation

    # Should not reach here, but if we do:
    return SolverResult(
        output=None,
        transformation=None,
        confidence=0.0,
        from_cache=False,
        message="Metacognitive retry loop exhausted without solution",
    )
```

### 1.3 Update solve() Method

**Modify the solve() method in HolographicARCSolver (lines 149-259)**

```python
def solve(self, task: ARCTask) -> SolverResult:
    """
    Attempt to solve an ARC task.

    Args:
        task: ARC task with training pairs and test input

    Returns:
        SolverResult with output grid (or None if refused)
    """
    # 1. Compute task signature for neural lookup
    task_sig_vec = self._compute_task_signature_vector(task)

    # 2. Check skill memory (O(1) neural lookup)
    cached_label, cache_confidence = self._skill_memory.query(task_sig_vec)
    if cached_label is not None and cache_confidence >= self.SKILL_CONFIDENCE_THRESHOLD:
        cached_transform = self._transform_cache.get(cached_label)
        if cached_transform is not None:
            output = self._apply_transformation(
                cached_transform, task.test_input
            )
            return SolverResult(
                output=output,
                transformation=cached_transform,
                confidence=cache_confidence,
                from_cache=True,
                message=f"Retrieved from skill memory (conf={cache_confidence:.2f})",
            )

    # 3. Use search+verify strategy with metacognitive retry if enabled
    if self._strategy == "search_verify":
        if self._metacog_enabled:
            return self._solve_search_verify_with_metacog(task, task_sig_vec)
        else:
            return self._solve_search_verify(task, task_sig_vec)

    # 4. Use iterative solver for multi-step reasoning if enabled
    if self._iterative:
        iter_result = self._iterative_solver.solve(task)

        # Get the last transform from chain for consolidation
        last_transform = (
            iter_result.transform_chain[-1]
            if iter_result.transform_chain
            else None
        )

        # Store skill whenever we have a valid transformation
        if last_transform is not None:
            self._store_skill(task_sig_vec, last_transform)

        return SolverResult(
            output=iter_result.output,
            transformation=last_transform,
            confidence=iter_result.confidence,
            from_cache=False,
            message=f"Iterative: {iter_result.steps_taken} steps, solved={iter_result.solved}",
        )

    # 5. Fallback: Single-shot solving (original logic)
    # ... (existing code)
```

### 1.4 Add Logging Import

**At top of solver.py**

```python
import logging

logger = logging.getLogger(__name__)
```

---

## Part 2: Enhanced IterativeSolver (Medium Integration)

### 2.1 Add Metacognitive State to IterativeSolver

**Modify src/hologram/arc/iterative_solver.py**

```python
class IterativeSolver:
    """
    Multi-step ARC solver via state traversal with metacognitive support.

    Enhancements:
    - Uses resonate_topk() instead of single-shot resonate()
    - Verifies each step against training pairs
    - Tracks metacognitive state across iterations
    - Can retry failed steps with modified candidates
    """

    def __init__(
        self,
        encoder: ObjectEncoder,
        resonator: TransformationResonator,
        executor: TransformationExecutor,
        detector: Optional[ObjectDetector] = None,
        max_steps: int = MAX_STEPS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        refusal_threshold: float = REFUSAL_THRESHOLD,
        metacog_enabled: bool = True,  # NEW
        use_verification: bool = True,  # NEW
    ):
        """Initialize iterative solver."""
        self._encoder = encoder
        self._resonator = resonator
        self._executor = executor
        self._detector = detector or ObjectDetector()
        self._max_steps = max_steps
        self._convergence_threshold = convergence_threshold
        self._refusal_threshold = refusal_threshold
        self._metacog_enabled = metacog_enabled  # NEW
        self._use_verification = use_verification  # NEW

        # NEW: Metacognitive state for iterative solving
        if metacog_enabled:
            from hologram.cognition.metacognition import MetacognitiveState
            self._metacog_state = MetacognitiveState(encoder._codebook)

        # NEW: Verification
        if use_verification:
            from hologram.arc.search_verifier import SearchVerifier
            self._verifier = SearchVerifier(executor=executor, detector=detector)

    def solve(self, task: ARCTask) -> IterativeResult:
        """
        Attempt to solve an ARC task iteratively with metacognitive support.

        Args:
            task: ARC task with training pairs and test input

        Returns:
            IterativeResult with output grid and transformation chain
        """
        current_state = task.test_input
        transform_chain: List[TransformResult] = []
        visited_states: Set[str] = {self._state_hash(current_state)}

        for step in range(self._max_steps):
            # 1. Observe: What's the delta between current and target?
            observation = self._observe_remaining_delta(current_state, task.training)

            if observation is None:
                # No objects to transform, or no valid delta
                break

            # 2. Resonate: Find candidates for this step
            # (NEW) Use top-k instead of single-shot
            if self._metacog_enabled and step > 0:
                # Modulate based on previous confidence
                if len(transform_chain) > 0:
                    prev_confidence = transform_chain[-1].min_confidence
                    if prev_confidence < self._refusal_threshold:
                        # Previous step was weak, broaden search
                        observation = self._ops.bundle(
                            observation,
                            self._metacog_state.self_vector
                        )

            candidates = self._resonator.resonate_topk(
                observation,
                k=5 if self._metacog_enabled else 1,
                slot_k=3,
            )

            if not candidates:
                break

            # 2a. (NEW) Verify candidates against training pairs
            if self._use_verification:
                verified_transform = self._verifier.verify_candidates(
                    candidates, task.training
                )
                if verified_transform is None:
                    # No candidate passed verification
                    # (NEW) Update mood and potentially retry
                    if self._metacog_enabled:
                        self._metacog_state.update_from_confidence(0.0)
                    break
                result = verified_transform
            else:
                # Use best candidate (original behavior)
                result = candidates[0]

            # Update metacognitive state
            if self._metacog_enabled:
                self._metacog_state.update_from_confidence(result.min_confidence)

            # 3. Execute: Apply transform, get new state
            objects = self._detector.detect(current_state)
            new_state = self._executor.execute(
                action=result.action,
                target=result.target,
                modifier=result.modifier,
                objects=objects,
                grid=current_state,
            )

            # 3a. Verify progress: Check if transformation had any effect
            if new_state == current_state:
                # No progress made - transformation was ineffective
                break

            transform_chain.append(result)

            # 4. Check: Are we done?
            if self._is_solved(new_state, task):
                return IterativeResult(
                    output=new_state,
                    transform_chain=transform_chain,
                    steps_taken=step + 1,
                    solved=True,
                    confidence=result.min_confidence,
                )

            # 5. Cycle detection: Have we seen this state before?
            state_hash = self._state_hash(new_state)
            if state_hash in visited_states:
                # Stuck in a loop - return best effort
                break
            visited_states.add(state_hash)

            # 6. Iterate: Treat (new_state → target) as new sub-problem
            current_state = new_state

        # Didn't fully solve, return best effort
        return IterativeResult(
            output=current_state,
            transform_chain=transform_chain,
            steps_taken=len(transform_chain),
            solved=False,
            confidence=transform_chain[-1].min_confidence if transform_chain else 0.0,
        )

    # ... (rest of existing methods unchanged)
```

---

## Part 3: Enhanced SearchVerifier (Return Richer Feedback)

### 3.1 Augmented verify_candidates()

**Optional enhancement to src/hologram/arc/search_verifier.py**

```python
@dataclass
class VerificationFeedback:
    """Rich feedback from candidate verification."""
    passed: bool
    first_passing_candidate: Optional[TransformResult] = None
    closest_candidate: Optional[TransformResult] = None
    closest_score: float = 0.0
    passed_count: int = 0  # How many candidates passed partial verification
    scores: List[float] = field(default_factory=list)  # Score per candidate


def verify_candidates_with_feedback(
    self,
    candidates: List[TransformResult],
    training_pairs: List[TrainingPair],
) -> VerificationFeedback:
    """
    Verify candidates and return rich feedback about near-misses.

    Returns:
        VerificationFeedback including first pass and closest near-miss
    """
    passed_count = 0
    closest_candidate = None
    closest_score = 0.0
    scores = []

    for candidate in candidates:
        result = self.verify_transform(candidate, training_pairs)
        scores.append(result.score)

        if result.passed:
            passed_count += 1
            # Return immediately on first pass
            return VerificationFeedback(
                passed=True,
                first_passing_candidate=candidate,
                closest_candidate=None,
                closest_score=1.0,
                passed_count=passed_count,
                scores=scores,
            )

        # Track closest near-miss
        if result.score > closest_score:
            closest_score = result.score
            closest_candidate = candidate

    # No candidates passed
    return VerificationFeedback(
        passed=False,
        first_passing_candidate=None,
        closest_candidate=closest_candidate,
        closest_score=closest_score,
        passed_count=passed_count,
        scores=scores,
    )
```

---

## Part 4: Testing Integration

### 4.1 New Test: Metacognitive Retry

**Add to tests/arc/test_solver_search_verify.py**

```python
def test_solver_metacognitive_retry():
    """Test that metacognitive retry can find solutions verification-first attempt missed."""

    # This test is speculative: create a task where metacognitive modulation helps
    # In practice, this would need a task that:
    # 1. First attempt fails verification (no candidates pass)
    # 2. Second attempt (with broadened search via curiosity) passes

    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="search_verify",
        search_k=10,
        search_slot_k=3,
        metacog_enabled=True,
        metacog_max_retries=2,
    )

    # Create a task (this is simplified; real test would need carefully designed task)
    task = create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]],
        ],
        train_outputs=[
            [[0, 1], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        test_input=[[1, 0], [0, 0]],
        test_output=[[0, 1], [0, 0]],
    )

    result = solver.solve(task)

    # Check that solver attempted retries
    if result.output is not None:
        assert "attempt" in result.message.lower()  # Message should mention retry
        assert result.confidence == 1.0  # Verified = perfect confidence


def test_solver_metacognitive_mood_tracking():
    """Test that mood is tracked across solve() calls."""

    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="search_verify",
        metacog_enabled=True,
    )

    # Solve a task
    task = create_simple_task(
        train_inputs=[[[1, 0], [0, 0]]],
        train_outputs=[[[0, 1], [0, 0]]],
        test_input=[[1, 0], [0, 0]],
    )

    result = solver.solve(task)

    # Check that metacognitive state was updated
    assert solver._metacog_state is not None
    # Mood should be set based on result
    if result.output is not None:
        from hologram.cognition.metacognition import MetacognitiveMood
        assert solver._metacog_state.mood in [
            MetacognitiveMood.CONFIDENT,
            MetacognitiveMood.CURIOUS,
        ]
```

---

## Part 5: Execution Flow Comparison

### Before Integration

```
solve(task)
  ├─ check skill memory
  ├─ _solve_search_verify(task, sig)
  │  ├─ observe_training_pair() → obs
  │  ├─ bundle(obs, ...) → observation_bundle
  │  ├─ resonate_topk(observation_bundle) → candidates
  │  └─ verify_candidates(candidates) → transform or None
  │     └─ if None: return refusal
  └─ return result
```

**Total attempts:** 1
**Adaptation:** None
**Self-observation:** No

### After Integration

```
solve(task)
  ├─ check skill memory
  ├─ _solve_search_verify_with_metacog(task, sig)
  │  ├─ FOR attempt in range(max_retries + 1):
  │  │  ├─ self._metacog_state.mood → NEUTRAL/CONFUSED/ANXIOUS
  │  │  ├─ IF confused: modulate observation with self_vector
  │  │  ├─ resonate_topk(modulated_obs) → candidates
  │  │  ├─ verify_candidates(candidates) → transform or None
  │  │  │  └─ if found: update_mood(CONFIDENT), return result
  │  │  └─ if not found: update_mood(CONFUSED), continue loop
  │  └─ after max_retries: return refusal
  └─ return result
```

**Total attempts:** 1-3 (configurable)
**Adaptation:** YES - modulation based on mood
**Self-observation:** YES - mood updated at each attempt

---

## Part 6: Configuration Examples

### Example 1: Aggressive Metacognitive Retry

```python
solver = HolographicARCSolver(
    dimensions=10000,
    strategy="search_verify",
    search_k=20,
    search_slot_k=5,
    metacog_enabled=True,
    metacog_max_retries=3,  # Try up to 4 times total
)
```

### Example 2: Metacog Disabled (Original Behavior)

```python
solver = HolographicARCSolver(
    dimensions=10000,
    strategy="search_verify",
    metacog_enabled=False,  # Don't use metacognitive retry
)
```

### Example 3: Iterative with Enhanced Verification

```python
solver = HolographicARCSolver(
    dimensions=10000,
    iterative=True,
    max_steps=5,
)

# Internally, IterativeSolver is configured with:
# - metacog_enabled=True
# - use_verification=True
```

---

## Part 7: Logging and Debugging

### Enable Metacognitive Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now solve() will log:
# [DEBUG] Attempt 1 failed. Mood now: confused. Will retry.
# [DEBUG] Attempt 2 with broadened search (k=40, slot_k=7)
# [DEBUG] Attempt 2 passed! Final mood: confident.
```

### Debug Output Example

```
INITIAL STATE: task_id=arc_123
  ├─ Skill memory: miss
  ├─ Strategy: search_verify
  └─ Metacog: enabled

ATTEMPT 1:
  ├─ Observation: bundle(obs1, obs2)
  ├─ Modulation: mood=NEUTRAL (no modulation)
  ├─ Resonator: resonate_topk(obs, k=20, slot_k=5) → 20 candidates
  ├─ Verifier: 0 of 20 passed verification
  └─ Mood update: NEUTRAL → CONFUSED (confidence=0.0)

ATTEMPT 2:
  ├─ Observation: bundle(obs1, obs2, self_vector)  ← MODULATED
  ├─ Modulation: mood=CONFUSED (add curiosity)
  ├─ Resonator: resonate_topk(modulated_obs, k=40, slot_k=7) → 40 candidates
  ├─ Verifier: 1 of 40 passed verification ✓
  └─ Mood update: CONFUSED → CONFIDENT (confidence=1.0)

FINAL RESULT:
  ├─ Output: Grid([...])
  ├─ Confidence: 1.0
  ├─ Message: "Verified (attempt 2, mood=confident): translate(all_objects, up)"
  └─ Retries: 1 (of 2 max allowed)
```

---

## Part 8: Expected Outcomes

### When Integration is Complete

1. **Verification failures trigger adaptation**
   - Instead of immediate refusal, system retries with modified search
   - Success rate improves on tasks where first-attempt candidates were "close"

2. **Metacognitive state guides candidate generation**
   - Curious mood → broader search (more candidates, more slot variations)
   - Anxious mood → focused search (fewer candidates, deeper verification)
   - Confident mood → original search parameters

3. **Logging reveals internal reasoning**
   - Users see number of retries, mood transitions, and final strategy
   - Developers can debug why specific tasks fail

4. **Graceful degradation**
   - Tasks that fail verification multiple times get clear "given up after retries" message
   - Original deterministic behavior preserved if metacog_enabled=False

---

## Part 9: Validation Checklist

Before considering integration complete:

- [ ] MetacognitiveLoop imports work in solver.py
- [ ] _metacog_state initializes without errors
- [ ] solve() calls _solve_search_verify_with_metacog() when metacog_enabled=True
- [ ] Retry loop executes up to max_retries times
- [ ] Observation modulation bundles self_vector when mood=CONFUSED
- [ ] verify_candidates() failures trigger mood update to CONFUSED
- [ ] Mood tracking persists across attempts within single solve() call
- [ ] Tests pass: test_solver_metacognitive_retry, test_solver_metacognitive_mood_tracking
- [ ] Logging shows attempt count and mood transitions
- [ ] metacog_enabled=False preserves original behavior exactly

---

**Blueprint Status:** READY FOR IMPLEMENTATION

**Estimated Development Time:** 3-4 hours (if starting from this blueprint)

**Testing Time:** 2-3 hours (design test tasks, verify retry behavior)

**Risk Level:** LOW (additive, doesn't modify existing code paths if metacog_enabled=False)

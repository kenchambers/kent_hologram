# Proposal 1: Holographic Surprise Learning

## Technical Specification for Titans-Inspired Noise Reduction

**Version**: 1.0
**Date**: 2025-12-09
**Status**: Proposed
**Estimated Effort**: 2-3 days
**Risk Level**: Low (Non-breaking enhancement)
**Inspired By**: [Google Titans Architecture](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

---

## Executive Summary

This proposal introduces a **Surprise Metric** into Hologram's learning pipeline, directly inspired by Google's Titans architecture. The core insight: **Don't memorize what you already know.**

Currently, Hologram bundles every fact with equal weight, causing noise accumulation (Root Cause #4). By gating updates based on "surprise" (how novel the information is), we achieve:

- **No duplicate encoding** - Teaching the same fact 1000x has the same effect as teaching it once
- **Natural saturation** - Memory stops "growing" once it has learned a domain
- **Increased capacity** - Less noise = more effective storage for new facts

---

## Problem Statement

### Current Implementation (memory_trace.py:60-91)

```python
def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
    """
    Store a key-value pair in memory.
    """
    # Bind key to value
    fact = Operations.bind(key, value)

    # Bundle into existing trace
    if self._fact_count == 0:
        self._trace = fact
    else:
        self._trace = Operations.bundle(self._trace, fact)  # ❌ ALWAYS bundles

    self._fact_count += 1
```

**The Problem:**
Every call to `store()` bundles unconditionally. If the user says "The capital of France is Paris" 50 times, we bundle 50 nearly-identical vectors. This:

1. **Increases noise** - Each bundle operation adds interference
2. **Wastes capacity** - Redundant facts consume limited holographic bandwidth
3. **Skews retrieval** - Over-represented facts dominate similarity scores

### Mathematical Analysis

Given a memory trace $M$ with $n$ facts, the expected retrieval similarity for a known fact is:

$$\text{Similarity} \approx \frac{1}{\sqrt{n}}$$

If we store the same fact $k$ times among $n$ unique facts, the "duplicate" fact gets an unfair boost:

$$\text{Similarity}_{dup} \approx \frac{k}{\sqrt{n + k}}$$

This creates **retrieval bias** toward frequently-taught facts, even if they're irrelevant to the query.

---

## Proposed Solution: Surprise-Gated Learning

### Core Algorithm

Before bundling a new fact, measure how "surprising" it is relative to existing memory:

```python
def store_with_surprise(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    learning_rate: float = 0.5,
    surprise_threshold: float = 0.1
) -> float:
    """
    Store with Titans-inspired surprise gating.

    Args:
        key: Key hypervector (e.g., bind(subject, predicate))
        value: Value hypervector (e.g., object encoding)
        learning_rate: Base learning rate for updates
        surprise_threshold: Minimum surprise to trigger learning

    Returns:
        Surprise score (0.0 = known, 1.0 = completely novel)
    """
    self._space.validate_vector(key)
    self._space.validate_vector(value)

    # Create the fact vector
    fact = Operations.bind(key, value)

    # Step 1: Calculate Surprise (Titans insight)
    if self._fact_count == 0:
        surprise = 1.0  # First fact is always surprising
    else:
        # How similar is this fact to what we already know?
        similarity = Similarity.cosine(self._trace, fact)
        surprise = 1.0 - max(0.0, similarity)  # Clamp negative similarities

    # Step 2: Gate the update
    if surprise < surprise_threshold:
        # We already know this - skip update
        return surprise

    # Step 3: Weighted update (more surprise = stronger update)
    update_strength = surprise * learning_rate

    if self._fact_count == 0:
        self._trace = fact
    else:
        # Scale the fact by surprise before bundling
        weighted_fact = fact * update_strength
        self._trace = Operations.bundle(self._trace, weighted_fact)

        # Re-normalize to prevent drift
        norm = torch.norm(self._trace)
        if norm > 1e-6:
            self._trace = self._trace / norm

    self._fact_count += 1
    return surprise
```

### Integration Point: FactStore

Modify `fact_store.py` to use surprise-gated learning:

```python
# fact_store.py - Modified add_fact method

def add_fact(
    self,
    subject: str,
    predicate: str,
    obj: str,
    source: Optional[str] = None,
    confidence: float = 1.0
) -> Optional[Fact]:
    """
    Add a fact with surprise-gated learning.
    """
    # ... existing normalization and duplicate check ...

    # Encode components
    s_vec = self._codebook.encode(subject_norm)
    p_vec = self._codebook.encode(predicate_norm)
    o_vec = self._codebook.encode(obj)

    # Create key: bind(subject, predicate)
    key = Operations.bind(s_vec, p_vec)

    # NEW: Store with surprise gating
    surprise = self._memory.store_with_surprise(
        key,
        o_vec,
        learning_rate=confidence,  # Use fact confidence as learning rate
        surprise_threshold=0.1
    )

    # Track surprise in metadata for debugging
    fact = Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source=source,
    )
    fact.surprise_score = surprise  # NEW: Track for analysis

    # Only count as new fact if surprise was above threshold
    if surprise >= 0.1:
        self._facts.append(fact)
        self._value_vocab.add(obj)
        self._exact_index[exact_key] = fact
        self._value_vectors_cache[obj] = o_vec
        return fact

    return None  # Duplicate - no new learning occurred
```

---

## Implementation Details

### File Changes Required

| File                                  | Change Type | Description                        |
| ------------------------------------- | ----------- | ---------------------------------- |
| `src/hologram/memory/memory_trace.py` | **Modify**  | Add `store_with_surprise()` method |
| `src/hologram/memory/fact_store.py`   | **Modify**  | Use surprise-gated storage         |
| `src/hologram/core/similarity.py`     | **None**    | Already has `cosine()` method      |
| `src/hologram/config/constants.py`    | **Add**     | Add `SURPRISE_THRESHOLD` constant  |

### New Constants (constants.py)

```python
# Surprise Learning (Titans-inspired)
SURPRISE_THRESHOLD = 0.1
"""Minimum surprise to trigger learning (0.0-1.0).
Below this, facts are considered 'already known' and skipped."""

SURPRISE_LEARNING_RATE = 0.5
"""Base learning rate for surprise-weighted updates.
Higher = faster learning, but more noise from novel facts."""

SURPRISE_DECAY = 0.99
"""Optional: Decay factor for old memories.
Enables forgetting of rarely-accessed facts (Titans momentum)."""
```

### Full Implementation: memory_trace.py

```python
class MemoryTrace:
    """
    Holographic memory trace with surprise-gated learning.
    """

    def __init__(self, space: VectorSpace):
        self._space = space
        self._trace = space.empty_vector()
        self._fact_count = 0
        self._momentum = space.empty_vector()  # NEW: Titans momentum
        self._momentum_decay = 0.9  # How quickly momentum fades

    def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Legacy store method - delegates to surprise-gated version."""
        self.store_with_surprise(key, value)

    def store_with_surprise(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        learning_rate: float = 0.5,
        surprise_threshold: float = 0.1
    ) -> float:
        """
        Store with Titans-inspired surprise gating.

        Returns:
            Surprise score (0.0 = known, 1.0 = novel)
        """
        self._space.validate_vector(key)
        self._space.validate_vector(value)

        # Create fact vector
        fact = Operations.bind(key, value)

        # Calculate surprise
        if self._fact_count == 0:
            surprise = 1.0
            momentum_surprise = 1.0
        else:
            # Current surprise: how different from memory?
            memory_sim = Similarity.cosine(self._trace, fact)
            surprise = 1.0 - max(0.0, memory_sim)

            # Momentum surprise: how different from recent learning direction?
            if torch.norm(self._momentum) > 1e-6:
                momentum_sim = Similarity.cosine(self._momentum, fact)
                momentum_surprise = 1.0 - max(0.0, momentum_sim)
            else:
                momentum_surprise = surprise

        # Combined surprise (Titans uses both instant and momentum)
        combined_surprise = 0.7 * surprise + 0.3 * momentum_surprise

        # Gate update
        if combined_surprise < surprise_threshold:
            return combined_surprise

        # Update momentum (exponential moving average of recent facts)
        self._momentum = self._momentum_decay * self._momentum + (1 - self._momentum_decay) * fact

        # Weighted update
        update_strength = combined_surprise * learning_rate

        if self._fact_count == 0:
            self._trace = fact
        else:
            weighted_fact = fact * update_strength
            self._trace = Operations.bundle(self._trace, weighted_fact)

            # Normalize
            norm = torch.norm(self._trace)
            if norm > 1e-6:
                self._trace = self._trace / norm

        self._fact_count += 1
        return combined_surprise

    def forget(self, decay: float = 0.99) -> None:
        """
        Apply forgetting (weight decay) to enable bounded memory.

        Titans insight: Active forgetting prevents memory saturation.
        Call periodically (e.g., every N store operations).

        Args:
            decay: Retention factor (0.99 = 1% forgetting per call)
        """
        self._trace = self._trace * decay
        # Re-normalize after decay
        norm = torch.norm(self._trace)
        if norm > 1e-6:
            self._trace = self._trace / norm
```

---

## Pitfalls & Mitigations

### Pitfall 1: Cold Start Problem

**Risk:** First few facts have surprise=1.0 and dominate memory.

**Mitigation:** Use a warm-up period with reduced learning rate:

```python
def store_with_surprise(self, key, value, learning_rate=0.5, ...):
    # Warm-up: reduce learning rate for first N facts
    if self._fact_count < 10:
        learning_rate = learning_rate * (self._fact_count + 1) / 10
    # ... rest of method
```

### Pitfall 2: Semantic Blindness

**Risk:** Two semantically equivalent facts (e.g., "France capital Paris" and "Paris is the capital of France") both have high surprise because vectors are different.

**Mitigation:** Add semantic similarity check before surprise calculation:

```python
def _semantic_duplicate_check(self, key, value) -> bool:
    """Check if fact is semantically equivalent to existing fact."""
    if self._fact_store._exact_index:
        # Use exact index for O(1) lookup
        return key in self._fact_store._exact_index
    return False
```

### Pitfall 3: Forgetting Important Facts

**Risk:** With `forget()` enabled, rarely-queried facts may decay below retrieval threshold.

**Mitigation:** Implement "access boost" - strengthen facts when queried:

```python
def query_with_boost(self, key: torch.Tensor) -> torch.Tensor:
    """Query and boost accessed facts (Hebbian reinforcement)."""
    result = self.query(key)

    # If high-confidence retrieval, slightly boost this fact
    # by re-bundling with small weight
    similarity = Similarity.cosine_batch(result, candidates)
    if similarity.max() > 0.7:
        boost_factor = 0.1
        fact = Operations.bind(key, result)
        self._trace = Operations.bundle(self._trace, fact * boost_factor)
        # Re-normalize
        self._trace = self._trace / torch.norm(self._trace)

    return result
```

### Pitfall 4: Threshold Sensitivity

**Risk:** Wrong `SURPRISE_THRESHOLD` causes either no learning (too high) or no filtering (too low).

**Mitigation:** Implement adaptive threshold:

```python
class AdaptiveSurpriseThreshold:
    """Learns optimal threshold from experience."""

    def __init__(self, initial: float = 0.1):
        self.threshold = initial
        self.history = []  # (surprise, was_useful) pairs

    def record(self, surprise: float, was_useful: bool):
        """Record whether a learned fact was useful."""
        self.history.append((surprise, was_useful))
        if len(self.history) >= 50:
            self._retune()

    def _retune(self):
        """Adjust threshold based on history."""
        # Find threshold that maximizes useful fact retention
        useful = [s for s, u in self.history if u]
        if useful:
            # Set threshold to 90th percentile of useful facts' surprise
            self.threshold = sorted(useful)[int(len(useful) * 0.1)]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_surprise_learning.py

def test_duplicate_fact_not_learned():
    """Teaching same fact twice should only learn once."""
    trace = MemoryTrace(VectorSpace())
    key = torch.randn(10000)
    value = torch.randn(10000)

    surprise1 = trace.store_with_surprise(key, value)
    surprise2 = trace.store_with_surprise(key, value)

    assert surprise1 > 0.5  # First time: novel
    assert surprise2 < 0.1  # Second time: known
    assert trace.fact_count == 1  # Only one fact stored

def test_similar_fact_reduced_learning():
    """Similar (but not identical) facts should have reduced learning."""
    trace = MemoryTrace(VectorSpace())
    key = torch.randn(10000)
    value1 = torch.randn(10000)
    value2 = value1 + torch.randn(10000) * 0.1  # 10% noise

    surprise1 = trace.store_with_surprise(key, value1)
    surprise2 = trace.store_with_surprise(key, value2)

    assert surprise1 > 0.5
    assert 0.1 < surprise2 < 0.5  # Partially novel

def test_novel_fact_full_learning():
    """Completely novel facts should be fully learned."""
    trace = MemoryTrace(VectorSpace())
    key1 = torch.randn(10000)
    value1 = torch.randn(10000)
    key2 = torch.randn(10000)  # Orthogonal key
    value2 = torch.randn(10000)  # Orthogonal value

    surprise1 = trace.store_with_surprise(key1, value1)
    surprise2 = trace.store_with_surprise(key2, value2)

    assert surprise1 > 0.5
    assert surprise2 > 0.5  # Both novel

def test_momentum_captures_context():
    """Momentum should capture learning direction."""
    trace = MemoryTrace(VectorSpace())

    # Train on related facts (same domain)
    for i in range(5):
        key = torch.randn(10000)
        value = torch.randn(10000)
        trace.store_with_surprise(key, value)

    # Momentum should be non-zero
    assert torch.norm(trace._momentum) > 0.5
```

### Integration Tests

```python
def test_quiz_accuracy_maintained():
    """Surprise learning should not reduce quiz accuracy."""
    # Setup trainer with surprise learning enabled
    trainer = CrewTrainer(surprise_learning=True)

    # Teach facts
    facts = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Italy", "capital", "Rome"),
    ]
    for s, p, o in facts:
        trainer.teach(f"The {p} of {s} is {o}.")

    # Quiz
    correct = 0
    for s, p, o in facts:
        answer = trainer.ask(f"What is the {p} of {s}?")
        if o.lower() in answer.lower():
            correct += 1

    assert correct >= 2  # At least 66% accuracy (same as baseline)

def test_repeated_teaching_no_degradation():
    """Repeating facts should not degrade performance."""
    trainer = CrewTrainer(surprise_learning=True)

    # Teach same fact 100 times
    for _ in range(100):
        trainer.teach("The capital of France is Paris.")

    # Memory should not be saturated
    assert trainer.fact_store.saturation_estimate < 0.05  # ~5% saturation

    # Retrieval should still work
    answer = trainer.ask("What is the capital of France?")
    assert "paris" in answer.lower()
```

---

## Feature Complete Checklist

| Feature                   | Status | Acceptance Criteria                               |
| ------------------------- | ------ | ------------------------------------------------- |
| Surprise calculation      | 🔲     | `surprise = 1 - cosine(memory, fact)` implemented |
| Gated updates             | 🔲     | Facts with `surprise < threshold` are skipped     |
| Momentum tracking         | 🔲     | Exponential moving average of recent facts        |
| FactStore integration     | 🔲     | `add_fact()` uses surprise-gated storage          |
| Backward compatibility    | 🔲     | Legacy `store()` method still works               |
| Cold start handling       | 🔲     | Reduced learning rate for first 10 facts          |
| Unit tests passing        | 🔲     | All tests in `test_surprise_learning.py` pass     |
| Integration tests passing | 🔲     | Quiz accuracy >= 80% maintained                   |
| Documentation             | 🔲     | Docstrings and README updated                     |

---

## Rollout Plan

### Phase 1: Core Implementation (Day 1)

- [ ] Implement `store_with_surprise()` in `memory_trace.py`
- [ ] Add constants to `constants.py`
- [ ] Write unit tests

### Phase 2: Integration (Day 2)

- [ ] Modify `fact_store.py` to use surprise gating
- [ ] Add `surprise_score` tracking to `Fact` dataclass
- [ ] Write integration tests

### Phase 3: Validation (Day 3)

- [ ] Run full training/quiz cycle
- [ ] Compare before/after metrics
- [ ] Document findings

---

## Expected Impact

| Metric                        | Before  | After (Expected) | Improvement    |
| ----------------------------- | ------- | ---------------- | -------------- |
| Facts per training round      | 9       | 9                | Same           |
| Duplicate facts stored        | All     | 0                | 100% reduction |
| Memory saturation (100 facts) | 100%    | ~50%             | 50% reduction  |
| Quiz accuracy                 | 81%     | 81%+             | Maintained     |
| Noise floor                   | 0.2-0.3 | 0.1-0.2          | 30% reduction  |

---

## References

1. [Titans: Learning to Memorize at Test Time](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Google Research, 2025
2. [MIRAS: A Unified View of Sequence Modeling](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Google Research, 2025
3. `src/hologram/memory/memory_trace.py` - Current implementation
4. `src/hologram/memory/fact_store.py` - Integration point

---

**Document Control**

- **Author**: Engineering Team
- **Reviewers**: TBD
- **Approval**: TBD
- **Last Updated**: 2025-12-09

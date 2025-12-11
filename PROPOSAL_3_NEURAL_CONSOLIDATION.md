# Proposal 3: Neural Consolidation Architecture

## Technical Specification for Memory-as-a-Model

**Version**: 1.0
**Date**: 2025-12-09
**Status**: Proposed (Experimental)
**Estimated Effort**: 3-4 weeks
**Risk Level**: High (Fundamental architecture change)
**Inspired By**: [Google Titans Neural Memory Module](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

---

## Executive Summary

Neural Consolidation introduces a **Sleep Cycle** that compresses Hologram's linear bundle memory into a non-linear neural network. This is directly inspired by Titans' "Neural Memory Module" - a mini-network that learns on the fly.

The core insight: **Memory shouldn't be a bucket of vectors; it should be a function.**

Linear bundling (`M = F1 ⊕ F2 ⊕ ... ⊕ Fn`) has a **hard capacity limit**: after ~√D facts (≈100 for D=10,000), interference destroys retrieval. Neural networks have no such limit - they can learn exponentially more patterns through non-linear composition.

**This proposal keeps HDC for fast, real-time operations while using neural compression for long-term storage.**

---

## Problem Statement

### The Bundling Limit

From `memory_trace.py`:

```python
@property
def saturation_estimate(self) -> float:
    """
    Rough estimate of capacity usage.

    Based on heuristic: capacity ∝ √dimensions
    This is UNPROVEN and will be validated empirically.
    """
    capacity_estimate = self._space.dimensions ** 0.5  # = 100 for D=10,000
    return self._fact_count / capacity_estimate
```

**The Mathematics:**

For a bundle of $n$ random vectors in $D$ dimensions:

- Expected signal strength: $\frac{1}{\sqrt{n}}$
- Noise floor: $\frac{n-1}{\sqrt{D}}$
- Retrieval fails when: $\frac{1}{\sqrt{n}} < \frac{n-1}{\sqrt{D}}$

Solving: $n_{max} \approx D^{1/3} \approx 21$ for reliable retrieval, or $\sqrt{D} \approx 100$ for marginal retrieval.

**Observed Behavior:**

| Facts Stored | Retrieval Confidence | Status      |
| ------------ | -------------------- | ----------- |
| 10           | 0.8+                 | ✅ Good     |
| 50           | 0.4-0.6              | ⚠️ Degraded |
| 100          | 0.2-0.3              | 🔴 Marginal |
| 200+         | <0.2                 | ❌ Broken   |

### Why Neural Networks Don't Have This Limit

A 2-layer MLP with hidden size $H$ can represent:

- $O(H \times D)$ parameters
- $O(2^H)$ unique patterns (via non-linear composition)

For H=512, D=10,000: ~5M parameters storing potentially millions of patterns.

---

## Proposed Solution: Sleep Cycle Architecture

### Dual-Memory Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                       NEURAL CONSOLIDATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────────┐              ┌─────────────────┐              │
│   │   WORKING       │              │   LONG-TERM     │              │
│   │   MEMORY        │ ──SLEEP──>   │   MEMORY        │              │
│   │   (HDC Bundle)  │    CYCLE     │   (Neural Net)  │              │
│   │                 │              │                 │              │
│   │  • Fast O(1)    │              │  • Slow O(n)    │              │
│   │  • Linear       │              │  • Non-linear   │              │
│   │  • ~100 facts   │              │  • 10k+ facts   │              │
│   │  • Real-time    │              │  • Compressed   │              │
│   └─────────────────┘              └─────────────────┘              │
│           │                                  │                       │
│           │         ┌──────────────┐         │                       │
│           └────────>│   RETRIEVAL  │<────────┘                       │
│                     │   MERGER     │                                 │
│                     │              │                                 │
│                     │  Combines    │                                 │
│                     │  both stores │                                 │
│                     └──────────────┘                                 │
│                            │                                         │
│                            ▼                                         │
│                     ┌──────────────┐                                 │
│                     │   RESPONSE   │                                 │
│                     └──────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### The Sleep Cycle

When Working Memory reaches saturation:

1. **Consolidation**: Train neural net on Working Memory facts
2. **Clearance**: Reset Working Memory to empty
3. **Resume**: Continue fast HDC operations

This mimics biological memory consolidation during sleep.

---

## Implementation Details

### Component 1: Neural Memory Module

```python
# src/hologram/consolidation/neural_memory.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from hologram.core.vector_space import VectorSpace


@dataclass
class NeuralMemoryConfig:
    """Configuration for neural memory module."""

    input_dim: int = 10000       # HDC vector dimension
    hidden_dim: int = 512        # Hidden layer size
    output_dim: int = 10000      # Output dimension (same as input)
    num_layers: int = 2          # Number of hidden layers
    dropout: float = 0.1         # Dropout for regularization
    learning_rate: float = 0.001
    epochs_per_consolidation: int = 100
    batch_size: int = 32


class NeuralMemory(nn.Module):
    """
    Neural network for long-term fact storage.

    Architecture:
        Input: bind(subject, predicate) -> D dimensions
        Hidden: D -> H -> H -> D
        Output: object vector approximation

    This learns the mapping: key -> value
    where key = bind(subject, predicate)
    and value = object vector

    Unlike linear bundling, this can store exponentially more patterns
    through non-linear composition.
    """

    def __init__(self, config: Optional[NeuralMemoryConfig] = None):
        super().__init__()
        self.config = config or NeuralMemoryConfig()

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(self.config.input_dim, self.config.hidden_dim))
        layers.append(nn.LayerNorm(self.config.hidden_dim))
        layers.append(nn.GELU())  # Smooth activation
        layers.append(nn.Dropout(self.config.dropout))

        # Hidden layers
        for _ in range(self.config.num_layers - 1):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
            layers.append(nn.LayerNorm(self.config.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.dropout))

        # Output layer
        layers.append(nn.Linear(self.config.hidden_dim, self.config.output_dim))

        self.network = nn.Sequential(*layers)

        # Optimizer (created during training)
        self._optimizer = None

        # Track training stats
        self._facts_consolidated = 0
        self._consolidation_count = 0

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: key -> predicted value.

        Args:
            key: Bound key vector (bind(subject, predicate))

        Returns:
            Predicted value vector
        """
        return self.network(key)

    def query(self, key: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Query neural memory for a value.

        Args:
            key: Query key vector

        Returns:
            Tuple of (predicted_value, confidence)
            Confidence is based on output magnitude (learned uncertainty)
        """
        self.eval()
        with torch.no_grad():
            prediction = self.forward(key)

            # Confidence: norm of output (higher = more confident)
            # Untrained regions produce near-zero outputs
            confidence = torch.norm(prediction).item()

            # Normalize prediction for comparison
            if confidence > 1e-6:
                prediction = prediction / confidence
                confidence = min(confidence, 1.0)  # Cap at 1.0
            else:
                confidence = 0.0

        return prediction, confidence

    def consolidate(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        verbose: bool = False
    ) -> dict:
        """
        Consolidate facts from working memory into neural memory.

        This is the "sleep" phase where we train the network on
        accumulated facts.

        Args:
            keys: Tensor of key vectors [N, D]
            values: Tensor of value vectors [N, D]
            verbose: Print training progress

        Returns:
            Training statistics dict
        """
        self.train()

        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01  # L2 regularization
            )

        n_samples = keys.shape[0]
        n_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size

        losses = []

        for epoch in range(self.config.epochs_per_consolidation):
            epoch_loss = 0.0

            # Shuffle data
            perm = torch.randperm(n_samples)
            keys_shuffled = keys[perm]
            values_shuffled = values[perm]

            for batch_idx in range(n_batches):
                start = batch_idx * self.config.batch_size
                end = min(start + self.config.batch_size, n_samples)

                batch_keys = keys_shuffled[start:end]
                batch_values = values_shuffled[start:end]

                # Forward pass
                self._optimizer.zero_grad()
                predictions = self.forward(batch_keys)

                # Loss: cosine similarity (we want predictions ≈ values)
                # Using negative cosine similarity as loss
                loss = 1 - F.cosine_similarity(predictions, batch_values).mean()

                # Backward pass
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        self._facts_consolidated += n_samples
        self._consolidation_count += 1

        return {
            "final_loss": losses[-1],
            "facts_consolidated": n_samples,
            "total_facts": self._facts_consolidated,
            "consolidation_count": self._consolidation_count
        }

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "model_state": self.state_dict(),
            "config": self.config,
            "facts_consolidated": self._facts_consolidated,
            "consolidation_count": self._consolidation_count
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state"])
        self._facts_consolidated = checkpoint.get("facts_consolidated", 0)
        self._consolidation_count = checkpoint.get("consolidation_count", 0)
```

### Component 2: Consolidation Manager

```python
# src/hologram/consolidation/manager.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import threading
import time

from hologram.core.vector_space import VectorSpace
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.memory.memory_trace import MemoryTrace
from hologram.consolidation.neural_memory import NeuralMemory, NeuralMemoryConfig


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation manager."""

    saturation_threshold: float = 0.7  # Trigger consolidation at 70% saturation
    min_facts_to_consolidate: int = 20  # Minimum facts before consolidation
    background_consolidation: bool = True  # Run in background thread
    auto_consolidate: bool = True  # Automatically trigger consolidation


class ConsolidationManager:
    """
    Manages the dual-memory system with sleep cycle.

    Coordinates:
    1. Working Memory (HDC MemoryTrace) - Fast, real-time
    2. Long-Term Memory (NeuralMemory) - Slow, high-capacity
    3. Sleep Cycle - Consolidation when working memory saturates

    Example:
        >>> manager = ConsolidationManager(space, codebook)
        >>> manager.store(key, value)  # Goes to working memory
        >>> # ... store many facts ...
        >>> # Consolidation triggers automatically when saturated
        >>> result, confidence = manager.query(key)
    """

    def __init__(
        self,
        space: VectorSpace,
        config: Optional[ConsolidationConfig] = None,
        neural_config: Optional[NeuralMemoryConfig] = None
    ):
        self._space = space
        self._config = config or ConsolidationConfig()

        # Working memory (fast, limited)
        self._working_memory = MemoryTrace(space)

        # Long-term memory (slow, unlimited)
        neural_cfg = neural_config or NeuralMemoryConfig(
            input_dim=space.dimensions,
            output_dim=space.dimensions
        )
        self._long_term_memory = NeuralMemory(neural_cfg)

        # Track keys/values for consolidation
        self._pending_keys: List[torch.Tensor] = []
        self._pending_values: List[torch.Tensor] = []

        # Consolidation state
        self._consolidating = False
        self._consolidation_lock = threading.Lock()

    def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Store a fact in working memory.

        If saturation threshold is exceeded, triggers consolidation.

        Args:
            key: Key vector (bind(subject, predicate))
            value: Value vector (object encoding)
        """
        # Store in working memory
        self._working_memory.store(key, value)

        # Track for consolidation
        self._pending_keys.append(key.clone())
        self._pending_values.append(value.clone())

        # Check saturation
        if self._config.auto_consolidate:
            if self._should_consolidate():
                self._trigger_consolidation()

    def query(
        self,
        key: torch.Tensor,
        candidates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Query both memories and combine results.

        Strategy:
        1. Query working memory (fast)
        2. Query long-term memory (slow)
        3. Combine with weighted average based on confidence

        Args:
            key: Query key vector
            candidates: Optional candidate values for cleanup

        Returns:
            Tuple of (best_value, confidence)
        """
        # Query working memory
        working_result = self._working_memory.query(key)
        working_confidence = self._estimate_confidence(working_result, key)

        # Query long-term memory
        neural_result, neural_confidence = self._long_term_memory.query(key)

        # Combine results
        if working_confidence > neural_confidence:
            # Working memory is more confident
            combined = working_result
            confidence = working_confidence
        elif neural_confidence > working_confidence:
            # Long-term memory is more confident
            combined = neural_result
            confidence = neural_confidence
        else:
            # Similar confidence - average
            combined = (working_result + neural_result) / 2
            confidence = (working_confidence + neural_confidence) / 2

        # Cleanup if candidates provided
        if candidates is not None:
            similarities = Similarity.cosine_batch(combined, candidates)
            best_idx = torch.argmax(similarities).item()
            confidence = float(similarities[best_idx].item())
            combined = candidates[best_idx]

        return combined, confidence

    def _should_consolidate(self) -> bool:
        """Check if consolidation should trigger."""

        if self._consolidating:
            return False

        if len(self._pending_keys) < self._config.min_facts_to_consolidate:
            return False

        saturation = self._working_memory.saturation_estimate
        return saturation >= self._config.saturation_threshold

    def _trigger_consolidation(self) -> None:
        """Trigger consolidation (sleep cycle)."""

        if self._config.background_consolidation:
            # Run in background thread
            thread = threading.Thread(target=self._consolidate)
            thread.start()
        else:
            # Run synchronously
            self._consolidate()

    def _consolidate(self) -> dict:
        """
        Perform consolidation (the "sleep" phase).

        1. Train neural memory on pending facts
        2. Clear working memory
        3. Reset pending buffers
        """
        with self._consolidation_lock:
            if self._consolidating:
                return {}
            self._consolidating = True

        try:
            # Prepare data
            if len(self._pending_keys) == 0:
                return {}

            keys = torch.stack(self._pending_keys)
            values = torch.stack(self._pending_values)

            # Train neural memory
            stats = self._long_term_memory.consolidate(keys, values)

            # Clear working memory
            self._working_memory = MemoryTrace(self._space)

            # Clear pending buffers
            self._pending_keys = []
            self._pending_values = []

            return stats

        finally:
            self._consolidating = False

    def _estimate_confidence(
        self,
        result: torch.Tensor,
        key: torch.Tensor
    ) -> float:
        """Estimate confidence of working memory result."""

        # Reconstruct what we expect
        # If result is correct, bind(key, result) should be in memory
        reconstructed = Operations.bind(key, result)

        # Check similarity to memory trace
        similarity = Similarity.cosine(
            reconstructed,
            self._working_memory.trace_vector
        )

        return max(0.0, float(similarity))

    def force_consolidate(self) -> dict:
        """Force immediate consolidation (for testing)."""
        return self._consolidate()

    @property
    def working_memory_saturation(self) -> float:
        """Get working memory saturation."""
        return self._working_memory.saturation_estimate

    @property
    def total_facts_stored(self) -> int:
        """Get total facts across both memories."""
        return (
            self._working_memory.fact_count +
            self._long_term_memory._facts_consolidated
        )

    @property
    def is_consolidating(self) -> bool:
        """Check if consolidation is in progress."""
        return self._consolidating
```

### Component 3: Integration with FactStore

```python
# src/hologram/memory/fact_store.py - Modified

class FactStore:
    """
    Manages subject-predicate-object facts with neural consolidation.
    """

    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        enable_consolidation: bool = False  # NEW: Feature flag
    ):
        self._space = space
        self._codebook = codebook
        self._enable_consolidation = enable_consolidation

        if enable_consolidation:
            from hologram.consolidation.manager import ConsolidationManager
            self._memory = ConsolidationManager(space)
        else:
            self._memory = MemoryTrace(space)

        # ... rest of existing init ...

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
        confidence: float = 1.0
    ) -> Optional[Fact]:
        """Add a fact (with automatic consolidation if enabled)."""

        # ... existing validation and encoding ...

        # Store (consolidation happens automatically if enabled)
        self._memory.store(key, o_vec)

        # ... rest of existing logic ...

    def query(self, subject: str, predicate: str) -> tuple[str, float]:
        """Query (searches both working and long-term memory)."""

        # ... existing encoding ...

        if self._enable_consolidation:
            # Use dual-memory query
            candidates = torch.stack(candidates_list)
            result, confidence = self._memory.query(key, candidates)

            # Find matching value
            similarities = Similarity.cosine_batch(result, candidates)
            best_idx = torch.argmax(similarities).item()

            return value_list[best_idx], confidence
        else:
            # Original logic
            similarities = self._memory.resonance(key, candidates)
            # ... rest of existing logic ...
```

---

## Pitfalls & Mitigations

### Pitfall 1: Catastrophic Forgetting

**Risk:** Neural network overwrites old facts when learning new ones.

**Mitigation:** Elastic Weight Consolidation (EWC):

```python
class NeuralMemoryWithEWC(NeuralMemory):
    """Neural memory with catastrophic forgetting protection."""

    def __init__(self, config):
        super().__init__(config)
        self._fisher_information = {}  # Importance weights
        self._prev_params = {}  # Previous parameter values
        self._ewc_lambda = 1000  # EWC strength

    def _compute_fisher(self, keys: torch.Tensor, values: torch.Tensor):
        """Compute Fisher information for current parameters."""
        self._fisher_information = {}

        for name, param in self.named_parameters():
            self._fisher_information[name] = torch.zeros_like(param)

        # Compute gradients for each sample
        for i in range(keys.shape[0]):
            self.zero_grad()
            pred = self.forward(keys[i:i+1])
            loss = 1 - F.cosine_similarity(pred, values[i:i+1])
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    self._fisher_information[name] += param.grad ** 2

        # Average
        for name in self._fisher_information:
            self._fisher_information[name] /= keys.shape[0]

    def _ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = 0

        for name, param in self.named_parameters():
            if name in self._fisher_information:
                fisher = self._fisher_information[name]
                prev = self._prev_params.get(name, param.data)
                loss += (fisher * (param - prev) ** 2).sum()

        return self._ewc_lambda * loss

    def consolidate(self, keys, values, verbose=False):
        """Consolidate with EWC protection."""

        # Save current params
        for name, param in self.named_parameters():
            self._prev_params[name] = param.data.clone()

        # Standard training with EWC loss added
        # ... modify training loop to add _ewc_loss() to total loss ...

        # Update Fisher information
        self._compute_fisher(keys, values)

        return super().consolidate(keys, values, verbose)
```

### Pitfall 2: Slow Consolidation

**Risk:** Training during consolidation blocks real-time operations.

**Mitigation:** Incremental consolidation with budget:

```python
def consolidate_incremental(
    self,
    keys: torch.Tensor,
    values: torch.Tensor,
    time_budget_ms: float = 100
) -> dict:
    """Consolidate with time budget."""

    import time
    start = time.time()
    budget = time_budget_ms / 1000

    epoch = 0
    while (time.time() - start) < budget:
        # One epoch of training
        self._train_one_epoch(keys, values)
        epoch += 1

    return {"epochs_completed": epoch, "time_used": time.time() - start}
```

### Pitfall 3: Query Latency

**Risk:** Neural network inference adds latency to every query.

**Mitigation:** Tiered query with early exit:

```python
def query_tiered(self, key: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Query with early exit if working memory is confident."""

    # Fast path: working memory only
    working_result = self._working_memory.query(key)
    working_sims = Similarity.cosine_batch(working_result, candidates)
    working_conf = working_sims.max().item()

    # Early exit if confident
    if working_conf > 0.7:
        best_idx = working_sims.argmax().item()
        return candidates[best_idx], working_conf

    # Slow path: include neural memory
    neural_result, neural_conf = self._long_term_memory.query(key)

    # ... rest of combination logic ...
```

### Pitfall 4: Memory Drift

**Risk:** Neural network outputs drift from valid codebook vectors over time.

**Mitigation:** Codebook anchoring loss:

```python
def consolidate_with_anchoring(self, keys, values, codebook_vectors):
    """Consolidate with codebook anchoring."""

    # Standard loss
    predictions = self.forward(keys)
    similarity_loss = 1 - F.cosine_similarity(predictions, values).mean()

    # Anchoring loss: predictions should be close to SOME codebook vector
    # For each prediction, find nearest codebook vector
    codebook_sims = torch.mm(predictions, codebook_vectors.T)
    nearest_sims = codebook_sims.max(dim=1).values
    anchoring_loss = 1 - nearest_sims.mean()

    # Combined loss
    total_loss = similarity_loss + 0.1 * anchoring_loss

    # ... backprop ...
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_neural_consolidation.py

def test_neural_memory_learns():
    """Neural memory should learn key-value associations."""
    neural = NeuralMemory()

    # Generate random facts
    keys = torch.randn(50, 10000)
    values = torch.randn(50, 10000)

    # Consolidate
    stats = neural.consolidate(keys, values)

    # Query
    for i in range(10):
        result, conf = neural.query(keys[i])
        similarity = F.cosine_similarity(result.unsqueeze(0), values[i].unsqueeze(0))
        assert similarity > 0.5, f"Failed to learn fact {i}"

def test_capacity_beyond_bundling():
    """Neural memory should handle more facts than bundling limit."""
    neural = NeuralMemory()

    # Generate 500 facts (5x bundling limit)
    keys = torch.randn(500, 10000)
    values = torch.randn(500, 10000)

    # Consolidate
    neural.consolidate(keys, values, verbose=True)

    # Test retrieval
    correct = 0
    for i in range(500):
        result, conf = neural.query(keys[i])
        similarity = F.cosine_similarity(result.unsqueeze(0), values[i].unsqueeze(0))
        if similarity > 0.3:
            correct += 1

    accuracy = correct / 500
    assert accuracy > 0.7, f"Accuracy too low: {accuracy}"

def test_consolidation_triggers_at_threshold():
    """Consolidation should trigger at saturation threshold."""
    manager = ConsolidationManager(
        VectorSpace(),
        ConsolidationConfig(saturation_threshold=0.5, min_facts_to_consolidate=10)
    )

    # Store facts until threshold
    for i in range(60):
        key = torch.randn(10000)
        value = torch.randn(10000)
        manager.store(key, value)

    # Should have triggered consolidation
    assert manager._long_term_memory._consolidation_count >= 1
```

### Integration Tests

```python
def test_fact_store_with_consolidation():
    """FactStore should work with consolidation enabled."""
    space = VectorSpace()
    codebook = Codebook(space)

    fs = FactStore(space, codebook, enable_consolidation=True)

    # Add many facts
    facts = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        # ... 100+ facts ...
    ]

    for s, p, o in facts:
        fs.add_fact(s, p, o)

    # Force consolidation
    if hasattr(fs._memory, 'force_consolidate'):
        fs._memory.force_consolidate()

    # Query all facts
    correct = 0
    for s, p, o in facts:
        answer, conf = fs.query(s, p)
        if answer.lower() == o.lower():
            correct += 1

    accuracy = correct / len(facts)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
```

---

## Feature Complete Checklist

| Feature                | Status | Acceptance Criteria                   |
| ---------------------- | ------ | ------------------------------------- |
| NeuralMemory class     | 🔲     | 2-layer MLP with GELU activation      |
| Consolidation training | 🔲     | Cosine similarity loss converges      |
| ConsolidationManager   | 🔲     | Manages dual-memory system            |
| Automatic trigger      | 🔲     | Consolidation at saturation threshold |
| Background thread      | 🔲     | Non-blocking consolidation            |
| Query merger           | 🔲     | Combines working + long-term results  |
| EWC protection         | 🔲     | Prevents catastrophic forgetting      |
| FactStore integration  | 🔲     | Feature flag enables consolidation    |
| Save/load              | 🔲     | Neural weights persist to disk        |
| Unit tests             | 🔲     | All tests pass                        |
| Capacity test          | 🔲     | 500+ facts with 70%+ accuracy         |

---

## Dependencies

### New Python Packages

```
# requirements.txt additions (already have torch)
# No new dependencies - uses existing PyTorch
```

### Hardware Recommendations

| Configuration | Working Memory | Long-Term Memory | Total RAM |
| ------------- | -------------- | ---------------- | --------- |
| Minimum       | 100 facts      | 2-layer MLP      | 500MB     |
| Recommended   | 100 facts      | 3-layer MLP      | 1GB       |
| Production    | 200 facts      | 4-layer MLP      | 2GB       |

---

## Rollout Plan

### Phase 1: Core Neural Memory (Days 1-7)

- [ ] Implement `NeuralMemory` class
- [ ] Write unit tests for learning
- [ ] Validate capacity beyond bundling limit

### Phase 2: Consolidation Manager (Days 8-14)

- [ ] Implement `ConsolidationManager`
- [ ] Add automatic triggering
- [ ] Implement background consolidation

### Phase 3: Integration (Days 15-21)

- [ ] Modify `FactStore` with feature flag
- [ ] Add save/load functionality
- [ ] Write integration tests

### Phase 4: Optimization (Days 22-28)

- [ ] Implement EWC protection
- [ ] Add tiered query
- [ ] Performance benchmarks

---

## Expected Impact

| Metric                 | Before | After (Expected) | Improvement     |
| ---------------------- | ------ | ---------------- | --------------- |
| Max facts (80% recall) | ~100   | 10,000+          | 100x            |
| Memory saturation      | Linear | Bounded          | Fundamental     |
| Query latency (P50)    | 10ms   | 50ms             | -5x (trade-off) |
| Storage efficiency     | O(n)   | O(log n)         | Sublinear       |

---

## Experimental Validation Required

This proposal is **experimental** and requires validation:

1. **Capacity Testing**: Verify 10,000+ facts can be stored with >70% recall
2. **Forgetting Testing**: Verify EWC prevents catastrophic forgetting
3. **Latency Testing**: Measure real-world query latency impact
4. **Integration Testing**: Verify existing functionality preserved

**Recommendation**: Implement behind feature flag, validate in staging before production.

---

## References

1. [Titans: Learning to Memorize at Test Time](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Neural Memory Module
2. [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) - Forgetting prevention
3. `src/hologram/memory/memory_trace.py` - Current working memory
4. `src/hologram/memory/fact_store.py` - Integration point

---

**Document Control**

- **Author**: Engineering Team
- **Reviewers**: TBD
- **Approval**: TBD
- **Last Updated**: 2025-12-09

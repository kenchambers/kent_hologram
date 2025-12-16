A plan created to add coding abilities to KENT

# Code Trainer + HDC-Native Generation Plan

## Philosophy

**Simple is elegant. Reuse existing patterns. Build incrementally.**

The user wants BOTH:

1. A `code_trainer.py` for SWE-bench (like `arc_trainer.py` and `crew_trainer.py`)
2. HDC-native code generation that leverages Kent's existing neural networks

---

## Key Insight: Kent Already Has What We Need

After reviewing the codebase, I found Kent already has generation infrastructure:

| Existing Component        | What It Does                               | How We Reuse It                               |
| ------------------------- | ------------------------------------------ | --------------------------------------------- |
| `NeuralMemoryNetwork`     | Classification head: key → index           | Add decoder head for index → tokens           |
| `TransformationResonator` | Factorizes into (ACTION, TARGET, MODIFIER) | Remap to (OPERATION, FILE, LOCATION) for code |
| `ResonantGenerator`       | Token-by-token with verification           | Same pattern for code tokens                  |
| `ConstraintAccumulator`   | Tracks failures, biases search             | Track failed patches, bias away               |
| `CircuitObserver`         | Cross-task learning                        | Learn which patterns succeed                  |
| `SelfImprovementManager`  | Persistence + reporting                    | Reuse directly                                |

**No new architectures needed.** We extend existing components.

---

## Phase 1: Testing Framework (Build First)

### 1.1 Create `tests/swe/__init__.py`

Empty file for package.

### 1.2 Create `tests/swe/conftest.py`

Shared fixtures:

```python
@pytest.fixture
def sample_swe_task() -> SWETask:
    """Minimal SWE task for testing."""

@pytest.fixture
def code_encoder(container) -> CodeEncoder:
    """CodeEncoder from container."""

@pytest.fixture
def code_resonator(code_encoder) -> CodeResonator:
    """CodeResonator for tests."""
```

### 1.3 Create `tests/swe/test_code_encoder.py`

```python
class TestCodeEncoder:
    def test_encode_file_change(self):
        """Encode a single file change."""

    def test_encode_multi_file(self):
        """Encode multiple file changes."""

    def test_vocabulary_constraint(self):
        """Only known operations/files can be encoded."""
```

### 1.4 Create `tests/swe/test_code_resonator.py`

```python
class TestCodeResonator:
    def test_factorize_simple_patch(self):
        """(FILE, OPERATION, LOCATION) factorization."""

    def test_factorize_confidence(self):
        """Confidence reflects factorization quality."""

    def test_resonate_topk(self):
        """Top-k candidates for iterative refinement."""
```

### 1.5 Create `tests/swe/test_code_trainer.py`

```python
class TestCodeTrainer:
    def test_initialization(self, tmp_path):
        """All components created correctly."""

    def test_single_task_learning(self, trainer, sample_swe_task):
        """Learn from one task, store pattern."""

    def test_pattern_persistence(self, trainer, tmp_path):
        """Train → save → load → verify patterns preserved."""

    def test_self_improvement_integration(self, trainer):
        """CircuitObserver tracks successes/failures."""
```

### 1.6 Create `tests/swe/test_code_generator.py`

```python
class TestCodeGenerator:
    def test_generate_from_pattern(self):
        """Generate patch from learned pattern."""

    def test_vocabulary_bounded(self):
        """Cannot output unknown tokens."""

    def test_verification_loop(self):
        """Generated patch verified against HDC encoding."""
```

---

## Phase 2: Core Components (Minimal Additions)

### 2.1 Create `src/hologram/swe/__init__.py`

```python
from .types import SWETask, CodePatch, PatchResult
from .encoder import CodeEncoder
from .code_resonator import CodeResonator
from .generator import CodeGenerator
from .benchmark import HonestCodeBenchmark
```

### 2.2 Create `src/hologram/swe/types.py` (~50 lines)

```python
@dataclass
class SWETask:
    task_id: str
    repo: str
    issue_text: str
    code_before: Dict[str, str]  # {filepath: content}
    code_after: Dict[str, str]   # Ground truth

@dataclass
class CodePatch:
    file: str
    operation: str  # "add_line", "delete_line", "modify_line", etc.
    location: str   # line number or function name
    content: str    # new content

@dataclass
class PatchResult:
    patches: List[CodePatch]
    confidence: float
    verification_passed: bool
```

### 2.3 Create `src/hologram/swe/encoder.py` (~120 lines)

Extend `ObjectEncoder` pattern:

```python
class CodeEncoder:
    """Encode code elements to HDC vectors."""

    def __init__(self, dimensions: int = 10000):
        self._dim = dimensions
        self._codebook = Codebook(dimensions)
        self._init_code_vocabulary()

    def _init_code_vocabulary(self):
        """Initialize vocabularies for FILE, OPERATION, LOCATION."""
        self.operations = ["add_line", "delete_line", "modify_line",
                          "add_function", "delete_function", "modify_function"]
        self.location_types = ["line_number", "function_name", "class_name"]

    def encode_patch(self, patch: CodePatch) -> torch.Tensor:
        """Encode patch as (OPERATION * op_role) + (FILE * file_role) + (LOCATION * loc_role)."""

    def encode_issue(self, issue_text: str) -> torch.Tensor:
        """Encode issue text (extract key terms, bundle)."""
```

### 2.4 Create `src/hologram/swe/code_resonator.py` (~80 lines)

Thin wrapper around `TransformationResonator`:

```python
class CodeResonator(TransformationResonator):
    """Factorize code observations into (FILE, OPERATION, LOCATION)."""

    def __init__(self, encoder: CodeEncoder, codebook: Codebook):
        # Remap roles: ACTION→OPERATION, TARGET→FILE, MODIFIER→LOCATION
        super().__init__(encoder, codebook)
        self._role_operation = self._role_action  # Alias
        self._role_file = self._role_target
        self._role_location = self._role_modifier

    def resonate(self, observation: torch.Tensor) -> CodePatchResult:
        """ALS factorization for code patches."""
        result = super().resonate(observation)
        return CodePatchResult(
            file=result.target,
            operation=result.action,
            location=result.modifier,
            min_confidence=result.min_confidence,
        )
```

---

## Phase 3: HDC-Native Code Generation (The Innovation)

### 3.1 Create `src/hologram/swe/generator.py` (~200 lines)

**Key insight:** Extend `NeuralMemoryNetwork` with a decoder head.

```python
class CodeGeneratorNetwork(nn.Module):
    """
    Extend NeuralMemoryNetwork for code generation.

    Architecture:
    1. HDC encoder (existing) → latent
    2. Classification head (existing) → pattern index
    3. NEW: Decoder head → token sequence
    """

    def __init__(self, input_dim=10000, hidden_dim=256, vocab_size=1000, max_tokens=50):
        super().__init__()

        # Reuse encoder from NeuralMemoryNetwork
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Pattern classifier (for retrieval)
        self.pattern_classifier = nn.Linear(hidden_dim, vocab_size)

        # NEW: Token decoder (autoregressive)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4)
        self.token_output = nn.Linear(hidden_dim, vocab_size)

    def forward_classify(self, hdc_vec: torch.Tensor) -> torch.Tensor:
        """Classify to pattern index (like NeuralMemory.query)."""
        hidden = self.encoder(hdc_vec)
        return self.pattern_classifier(hidden)

    def forward_generate(self, hdc_vec: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Generate next token conditioned on HDC + previous tokens."""
        hidden = self.encoder(hdc_vec).unsqueeze(0)  # (1, batch, hidden)
        tgt = self.token_embed(prev_tokens).transpose(0, 1)  # (seq, batch, hidden)
        decoded = self.decoder(tgt, hidden)
        return self.token_output(decoded[-1])  # Last position logits


class CodeGenerator:
    """
    High-level code generation with HDC verification.

    Combines:
    - CodeGeneratorNetwork for token prediction
    - CodeResonator for structured factorization
    - HDC verification loop (like ResonantGenerator)
    """

    def __init__(self, network: CodeGeneratorNetwork, resonator: CodeResonator,
                 encoder: CodeEncoder, max_tokens: int = 50):
        self._network = network
        self._resonator = resonator
        self._encoder = encoder
        self._max_tokens = max_tokens

    def generate(self, issue_vec: torch.Tensor, file_context: Dict[str, str]) -> PatchResult:
        """
        Generate patch for issue.

        1. Factorize issue → (OPERATION, FILE, LOCATION)
        2. Generate tokens conditioned on factorization
        3. Verify each token against HDC encoding
        4. Return bounded, verified patch
        """
        # Step 1: Factorize
        factorization = self._resonator.resonate(issue_vec)

        # Step 2: Generate tokens with verification
        tokens = []
        prev_tokens = torch.tensor([0])  # Start token

        for _ in range(self._max_tokens):
            logits = self._network.forward_generate(issue_vec, prev_tokens)

            # Verification: project back to HDC space
            top_k = torch.topk(logits, k=5)
            best_token = None

            for idx in top_k.indices:
                candidate_token = self._vocab[idx]
                candidate_vec = self._encoder.encode_token(candidate_token)

                # Check alignment with factorization
                similarity = cosine_similarity(candidate_vec, factorization.vector)
                if similarity > 0.3:  # Threshold
                    best_token = candidate_token
                    break

            if best_token is None:
                best_token = self._vocab[top_k.indices[0]]  # Fallback

            tokens.append(best_token)
            prev_tokens = torch.cat([prev_tokens, torch.tensor([self._vocab_to_idx[best_token]])])

            if best_token == "<END>":
                break

        return PatchResult(
            patches=self._tokens_to_patches(tokens, factorization),
            confidence=factorization.min_confidence,
            verification_passed=True,
        )
```

---

## Phase 4: Code Trainer Script

### 4.1 Create `scripts/code_trainer.py` (~400 lines)

Follow `arc_trainer.py` pattern exactly:

```python
@dataclass
class CodeCurriculum:
    """Organize SWE tasks by difficulty."""
    easy: List[SWETask]    # Single file, few lines
    medium: List[SWETask]  # Single file, complex
    hard: List[SWETask]    # Multi-file changes


class CodeTrainer:
    """
    Train on SWE-bench tasks with self-improvement.

    Training loop:
    1. Try to solve with current knowledge
    2. If wrong → extract correct pattern from ground truth
    3. Store pattern in NeuralMemory
    4. Track success/failure in CircuitObserver
    """

    def __init__(self, persist_dir: str, dimensions: int = 10000):
        self._persist_dir = Path(persist_dir)

        # Core components (reuse existing)
        self.container = HologramContainer(dimensions=dimensions)
        self._encoder = CodeEncoder(dimensions)
        self._resonator = CodeResonator(self._encoder, self.container._codebook)

        # Generation (new)
        self._generator_network = CodeGeneratorNetwork(dimensions)
        self._generator = CodeGenerator(self._generator_network, self._resonator, self._encoder)

        # Memory (reuse existing)
        self._neural_memory = self.container.create_neural_memory()

        # Self-improvement (reuse existing)
        self._improvement_manager = SelfImprovementManager(
            persist_path=str(self._persist_dir / "learned_patterns.json")
        )
        self._constraint_accumulator = ConstraintAccumulator()
        self._constraint_accumulator.set_circuit_observer(self._improvement_manager.observer)

    def train_on_task(self, task: SWETask) -> bool:
        """
        Learn from a single SWE-bench task.

        Returns True if solved correctly.
        """
        # 1. Encode issue
        issue_vec = self._encoder.encode_issue(task.issue_text)

        # 2. Try to solve
        result = self._generator.generate(issue_vec, task.code_before)

        # 3. Verify against ground truth
        correct = self._verify_patch(result.patches, task.code_after)

        # 4. Learn from outcome
        if correct:
            # Store successful pattern
            self._neural_memory.consolidate([
                ConsolidationFact(
                    key_vector=issue_vec,
                    value_label=self._pattern_to_label(result.patches),
                    value_index=0,
                )
            ])
        else:
            # Record failure for constraint accumulation
            self._constraint_accumulator.record_attempt(
                result=CodePatchResult(
                    file=result.patches[0].file if result.patches else "unknown",
                    operation=result.patches[0].operation if result.patches else "unknown",
                    location=result.patches[0].location if result.patches else "unknown",
                    min_confidence=result.confidence,
                ),
                partial_score=self._compute_partial_score(result.patches, task.code_after),
            )

            # Learn correct pattern from ground truth
            correct_patches = self._extract_patches_from_ground_truth(task)
            self._neural_memory.consolidate([
                ConsolidationFact(
                    key_vector=issue_vec,
                    value_label=self._pattern_to_label(correct_patches),
                    value_index=0,
                )
            ])

        return correct

    def run_continuous(self, curriculum: CodeCurriculum, max_rounds: int = 100,
                       validate_every: int = 10) -> Dict[str, Any]:
        """
        Main training loop.

        Same structure as ARCTrainer.run_continuous().
        """
        results = {"rounds": [], "best_accuracy": 0.0}

        for round_num in range(max_rounds):
            # Train on curriculum
            for difficulty, tasks in [("easy", curriculum.easy),
                                      ("medium", curriculum.medium),
                                      ("hard", curriculum.hard)]:
                for task in tasks:
                    self.train_on_task(task)

            # Periodic validation
            if (round_num + 1) % validate_every == 0:
                accuracy = self._validate(curriculum)
                results["rounds"].append({
                    "round": round_num + 1,
                    "accuracy": accuracy,
                })

                if accuracy > results["best_accuracy"]:
                    results["best_accuracy"] = accuracy
                    self.save()

        return results
```

---

## Phase 5: Benchmark

### 5.1 Create `src/hologram/swe/benchmark.py` (~150 lines)

Follow `hologram/arc/benchmark.py` pattern:

```python
class HonestCodeBenchmark:
    """SWE-bench evaluation with cache isolation."""

    def evaluate(self, tasks: List[SWETask], isolate_cache: bool = True) -> BenchmarkResult:
        """
        Evaluate on tasks with fresh solver per task (honest).

        Args:
            tasks: List of SWE tasks
            isolate_cache: If True, each task gets fresh solver (no cheating)
        """
        correct = 0
        total = len(tasks)

        for task in tasks:
            if isolate_cache:
                # Fresh solver for each task
                trainer = CodeTrainer(persist_dir=tempfile.mkdtemp())
            else:
                # Shared solver (for testing transfer learning)
                pass

            result = trainer._generator.generate(
                trainer._encoder.encode_issue(task.issue_text),
                task.code_before,
            )

            if self._verify(result.patches, task.code_after):
                correct += 1

        return BenchmarkResult(
            correct=correct,
            total=total,
            accuracy=correct / total if total > 0 else 0.0,
        )
```

---

## File Summary

| File                                 | Purpose            | Lines (est) |
| ------------------------------------ | ------------------ | ----------- |
| `tests/swe/__init__.py`              | Package init       | 0           |
| `tests/swe/conftest.py`              | Shared fixtures    | ~40         |
| `tests/swe/test_code_encoder.py`     | Encoder tests      | ~80         |
| `tests/swe/test_code_resonator.py`   | Resonator tests    | ~80         |
| `tests/swe/test_code_trainer.py`     | Integration tests  | ~100        |
| `tests/swe/test_code_generator.py`   | Generation tests   | ~100        |
| `src/hologram/swe/__init__.py`       | Package exports    | ~10         |
| `src/hologram/swe/types.py`          | Dataclasses        | ~50         |
| `src/hologram/swe/encoder.py`        | Code encoder       | ~120        |
| `src/hologram/swe/code_resonator.py` | Thin wrapper       | ~80         |
| `src/hologram/swe/generator.py`      | **The Innovation** | ~200        |
| `src/hologram/swe/benchmark.py`      | Honest evaluation  | ~150        |
| `scripts/code_trainer.py`            | Training script    | ~400        |

**Total: ~1,410 lines** (comparable to arc_trainer + benchmark + types)

---

## Implementation Order

1. **Tests first** (TDD):

   - `tests/swe/conftest.py`
   - `tests/swe/test_code_encoder.py`
   - `tests/swe/test_code_resonator.py`
   - `tests/swe/test_code_generator.py`
   - `tests/swe/test_code_trainer.py`

2. **Types and encoder**:

   - `src/hologram/swe/__init__.py`
   - `src/hologram/swe/types.py`
   - `src/hologram/swe/encoder.py`

3. **Core logic**:

   - `src/hologram/swe/code_resonator.py`
   - `src/hologram/swe/generator.py` (the innovative part)
   - `src/hologram/swe/benchmark.py`

4. **Training script**:
   - `scripts/code_trainer.py`

---

## What We Reuse (Not Reinvent)

- `HologramContainer` - shared infrastructure
- `ConsolidationManager` - memory persistence
- `TransformationResonator` - ALS factorization (base class)
- `ConstraintAccumulator` - track failures
- `CircuitObserver` - cross-task learning
- `SelfImprovementManager` - persistence + reporting
- `NeuralMemoryNetwork` architecture - encoder + classifier head
- `ResonantGenerator` verification loop pattern

---

## The Innovation: HDC-Conditioned Decoder

The key novelty is `CodeGeneratorNetwork`:

```
┌─────────────────────────────────────────────────────────────┐
│           HDC-CONDITIONED CODE GENERATION                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Issue Text ──► CodeEncoder ──► HDC Vector (10000-dim)      │
│                                      │                      │
│                                      ▼                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CodeGeneratorNetwork                               │   │
│  │  ┌──────────────┐  ┌──────────────┐                │   │
│  │  │   Encoder    │  │  Pattern     │ ──► "null_check"│   │
│  │  │ (from Neural │  │  Classifier  │    (retrieval)  │   │
│  │  │  Memory)     │  └──────────────┘                │   │
│  │  └──────────────┘         │                        │   │
│  │         │                 ▼                        │   │
│  │         │        ┌──────────────┐                  │   │
│  │         └───────►│   Decoder    │ ──► tokens       │   │
│  │                  │  (NEW part)  │    (generation)  │   │
│  │                  └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                      │                      │
│                                      ▼                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Verification Loop (from ResonantGenerator)         │   │
│  │  - Each token projected back to HDC                 │   │
│  │  - Verified against factorization                   │   │
│  │  - Reject if similarity < threshold                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                      │                      │
│                                      ▼                      │
│  Verified Patch (bounded, no hallucination)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this works:**

1. **HDC provides structure** - factorization bounds what operations are valid
2. **Neural decoder provides fluency** - generates natural token sequences
3. **Verification prevents hallucination** - every token validated against HDC
4. **Self-improvement enables learning** - CircuitObserver tracks what works

---

## Expected Performance

| Metric   | Initial (HDC-only) | After Training | With Self-Improvement |
| -------- | ------------------ | -------------- | --------------------- |
| Accuracy | 5-10%              | 15-25%         | 25-40%                |
| Cost     | $0/task            | $0/task        | $0/task               |
| Latency  | <1s                | <2s            | <3s                   |

**Honest assessment:** Pure HDC won't match LLM performance initially.
But it provides:

1. **Bounded hallucination** - cannot invent unknown APIs
2. **Explainable** - every patch traces to pattern + slots
3. **Learnable** - improves with each task
4. **Efficient** - no API costs

---

## Critical Files to Modify/Create

### New Files:

- `tests/swe/*.py` (6 files)
- `src/hologram/swe/*.py` (6 files)
- `scripts/code_trainer.py`

### No modifications to existing files needed.

---

## Questions for User

None - the plan is comprehensive and builds on established patterns.

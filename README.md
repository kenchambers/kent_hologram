# Hologram - Bentov-Style Holographic Memory

A Hyperdimensional Computing (HDC) system that stores knowledge as interference patterns rather than probabilistic weights, providing **bounded hallucination** through algebraic resonance.

## What Makes This Different?

Unlike traditional LLMs that predict the next token, Hologram:

- **Stores facts as interference patterns** in 10,000-dimensional space
- **Cannot hallucinate outside vocabulary** - can only retrieve stored concepts
- **Refuses when uncertain** - says "I don't know" instead of guessing
- **Provides citations** - every answer traces back to a source
- **Uses algebraic retrieval** - deterministic, not probabilistic

## Features

### Core Capabilities

- ✅ **Deterministic Vector Generation**: Same concept always produces same hypervector
- ✅ **Holographic Storage**: Facts bundled into interference patterns (the "water surface")
- ✅ **Bounded Hallucination**: Outputs constrained to vocabulary (cannot invent facts)
- ✅ **Sequence Encoding**: Preserves word order ("Dog bites Man" ≠ "Man bites Dog")
- ✅ **Confidence Thresholds**: "I don't know" when resonance is low
- ✅ **Citation Tracking**: Every fact has a source
- ✅ **Persistence**: Save/load memory to disk with checksum validation

### Resonant Cavity Architecture (v0.2.0)

- ✅ **Resonator**: Decomposes thoughts into (subject, verb, object) via ALS
- ✅ **Constrained Generation**: Outputs verified against target constraints
- ✅ **Style Modulation**: Formal, casual, urgent, or neutral tones
- ✅ **Disfluency Injection**: Natural "um", "uh" on low confidence
- ✅ **Generation Metrics**: Acceptance rate, hallucination risk tracking

### Conversational Learning (NEW in v0.3.0)

- ✅ **HDC Intent Classification**: Learned from examples, not hardcoded keywords
- ✅ **Teaching Detection**: Learns facts from natural language ("the capital of France is Paris")
- ✅ **Fact Persistence**: ChromaDB stores facts across sessions
- ✅ **Style Adaptation**: Tracks user communication style over time
- ✅ **Conversation Memory**: Context-aware responses for follow-ups
- ✅ **Neural Consolidation**: Sleep-inspired background learning transfers facts to long-term neural memory (v0.4.0)

### Safety Mechanisms

- **RefusalPolicy**: Refuses to answer when confidence < threshold
- **CitationEnforcer**: Every claim must trace to a stored fact
- **ConfidenceScorer**: Quantifies certainty of answers

### Developer Tools (NEW)

- 🔍 **Semantic Code Search**: AI-powered codebase navigation via EmbeddixDB
  - Natural language queries: "how does neural consolidation work?"
  - Ranked results by relevance with code context
  - Integrated with Claude Code for faster development
  - See [SEMANTIC_SEARCH.md](./SEMANTIC_SEARCH.md) for details

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable dependency management.

### Step 1: Install UV

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone <repo-url>
cd kent_hologram

# Sync dependencies (creates .venv and installs everything)
uv sync

# Verify installation
uv run python -c "import hologram; print(hologram.__version__)"
```

## Quick Start

> **Understanding Holographic Confidence Scores**:
>
> Holographic memory stores facts as interference patterns. When multiple facts are bundled together, they interfere with each other (this is normal and expected!). This means:
>
> - **Confidence scores of 20-40% for stored facts are normal** (not low!)
> - The system correctly identifies answers despite interference
> - More facts = more interference = lower raw similarity scores
> - Thresholds are calibrated to holographic characteristics (0.20 response, 0.10 refusal)
> - Unknown queries produce confidence <10% and are correctly refused
>
> Think of it like listening to a specific conversation at a party - you can hear it clearly even though many other conversations create background noise. The signal is there, just not at 100% clarity.

### Running the Interactive Chat Interface

The fastest way to try Hologram is through the interactive chat interface:

```bash
# Start the chat interface (conversational mode with persistence)
uv run hologram

# Or run as module
uv run python -m hologram.chat
```

**Conversational Mode (default):**

```
> Hello!
Hello! How can I help you?

> the capital of Germany is Berlin
Got it! I'll remember that Germany capital berlin.

> What is the capital of Germany?
The capital of Germany is berlin.

> Thanks!
You're welcome! Is there anything else you'd like to know?
```

Facts persist across sessions - restart the chat and your facts are still there!

**Slash Commands:**

- `/teach <fact>` - Teach a fact (e.g., `/teach France capital Paris`)
- `/facts` - List all stored facts
- `/learned` - Show what the system has learned
- `/store <subject> <predicate> <object> [source]` - Store a fact directly
- `/query <subject> <predicate>` - Query with confidence scoring
- `/save <path>` - Save memory to disk
- `/load <path>` - Load memory from disk
- `/stats` - Show memory statistics
- `/help` - Show help
- `/exit` - Exit

**Example Teaching Session:**

```
> the capital of France is Paris
Got it! I'll remember that France capital paris.

> /facts
📚 Stored Facts:
  1. France capital paris

> What is the capital of France?
The capital of France is paris.
```

### Basic Usage in Python

```python
from hologram import HologramContainer

# Initialize the system
container = HologramContainer(dimensions=10000)
fact_store = container.create_fact_store()

# Store facts with sources
fact_store.add_fact("France", "capital", "Paris", source="Wikipedia")
fact_store.add_fact("Germany", "capital", "Berlin", source="Wikipedia")

# Query with confidence scoring
answer, confidence = fact_store.query("France", "capital")
print(f"Answer: {answer} (confidence: {confidence:.2%})")
# Output: Answer: Paris (confidence: 95%)

# Unknown query - demonstrates bounded hallucination
answer, confidence = fact_store.query("Mars", "color")
print(f"Confidence: {confidence:.2%}")  # Very low
# Output: Confidence: 5% (will refuse to answer)
```

### With Safety Mechanisms

```python
from hologram import (
    HologramContainer,
    ConfidenceScorer,
    RefusalPolicy,
    CitationEnforcer
)

container = HologramContainer()
fact_store = container.create_fact_store()
fact_store.add_fact("Earth", "shape", "round", source="NASA")

# Setup safety layers
scorer = ConfidenceScorer(response_threshold=0.6, refusal_threshold=0.3)
policy = RefusalPolicy(scorer)
citations = CitationEnforcer(fact_store)

# Query
answer, conf = fact_store.query("Earth", "shape")

# Check if should refuse
refusal = policy.evaluate(answer, conf)
if refusal.should_refuse:
    print(policy.format_refusal(refusal))
else:
    print(scorer.format_response(answer, conf))
    # Find citation
    fact = citations.find_supporting_fact("Earth", "shape", answer)
    print(citations.format_citation(fact))
# Output: round (confidence: 92%)
#         [NASA] Earth --shape--> round
```

### Persistence

```python
from hologram import HologramContainer, StateManager
from pathlib import Path

# Create and populate
container = HologramContainer()
fact_store = container.create_fact_store()
fact_store.add_fact("France", "capital", "Paris")

# Save to disk
manager = StateManager()
manager.save(fact_store, Path("./data/session1"))

# Load later
restored = manager.load(Path("./data/session1"))
answer, conf = restored.query("France", "capital")
print(answer)  # Paris
```

### Conversational Chatbot with Persistent Learning

```python
from hologram import HologramContainer

# Create chatbot with ChromaDB persistence
container = HologramContainer(dimensions=10000)
chatbot = container.create_persistent_chatbot("./data/my_facts")

# Start session
greeting = chatbot.start_session()
print(greeting)  # "Hello! How can I help you today?"

# Teach facts via natural language
chatbot.respond("the capital of Germany is Berlin")
# → "Got it! I'll remember that Germany capital berlin."

# Query facts
chatbot.respond("What is the capital of Germany?")
# → "The capital of Germany is berlin."

# Facts persist across sessions!
# Exit and restart - facts are still there
```

### Running Examples

```bash
# Run the complete demonstration
uv run python examples/complete_demo.py

# Run basic usage example
uv run python examples/basic_usage.py

# Run persistence demo
uv run python examples/persistence_demo.py
```

## Overnight Training

Train your Hologram chatbot while you sleep using the CrewAI trainer! Multiple LLMs (Gemini and Claude) have conversations with your Hologram, naturally teaching it facts through dialogue.

**Quick Start:**

```bash
# Run 100 rounds of training overnight
./scripts/train_overnight.sh

# Or manually configure
uv run python scripts/crew_trainer.py --max-rounds 100
```

**What happens:**

1. Gemini starts conversations or teaches facts
2. Hologram learns and responds
3. Claude continues the conversation naturally
4. Facts automatically persist to ChromaDB
5. All conversations logged for review

**Features:**

- ✅ Automatic error recovery with exponential backoff
- ✅ Status reports every 10 rounds
- ✅ Graceful shutdown (Ctrl+C)
- ✅ All facts persist across sessions
- ✅ Full conversation logs

**Current Training System Performance:**

- **Quiz Accuracy**: **81%** (improved from 66.7% after garbage response fixes, up from 0% baseline)
- **Facts Learned**: 7-9 per training round
- **Responses Learned**: 6 per training round
- **Vocabulary Growth**: Dynamic (starts at 60+ words, grows with learning)
- **Target (Option 3 Full)**: 90%+ quiz accuracy

**Architecture Status:**

- ✅ **Option 2 Implemented** (66.7% baseline): Dynamic vocabulary, confidence thresholds, explicit learning protocol
- ✅ **Garbage Response Fixes** (81% accuracy): Vocabulary cleanup, improved subject extraction, enhanced validation
- ⚠️ **Option 3 Partially Implemented** (32% complete): Role-based thought vectors (60%), output validation (50%)
- 🎯 **Option 3 Target**: Circuit breaker, semantic coherence, hallucination detection

See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for complete documentation including [Project Gutenberg ingestion](./TRAINING_GUIDE.md#training-on-project-gutenberg) (train on 75,570+ public domain books with `--max-books` option), and [docs/CONSCIOUS_HOLOGRAM_ARCH.md](./docs/CONSCIOUS_HOLOGRAM_ARCH.md) for architecture details.

## Examples

See the `examples/` directory:

- `basic_usage.py` - Basic fact storage and retrieval
- `persistence_demo.py` - Save/load demonstration
- `complete_demo.py` - Full system with all features

Run any example with: `uv run python examples/<filename>.py`

## Architecture

### The Complete "Conscious Hologram" Stack

The Hologram system is built as a **5-layer architecture** where each layer adds essential capabilities. The **Ventriloquist is an addition, not a replacement** - it sits on top of the complete HDC foundation.

| Layer             | Component                                       | Status          | Role                                                       |
| ----------------- | ----------------------------------------------- | --------------- | ---------------------------------------------------------- |
| **1. Substrate**  | `FractalSpace`                                  | **IMPLEMENTED** | Every vector is a holographic "DNA" - robust to corruption |
| **2. Memory**     | `FactStore` + `MemoryTrace`                     | **IMPLEMENTED** | HDC binding/bundling for fact storage with S-P-O structure |
| **3. Controller** | `MetacognitiveLoop`                             | **IMPLEMENTED** | Self-monitoring, confidence tracking, retry when confused  |
| **4. Retrieval**  | `FactStore` + `ResponseSelector`                | **IMPLEMENTED** | Multi-strategy lookup (exact match + HDC resonance)        |
| **5. Voice**      | `ResonantGenerator` or `VentriloquistGenerator` | **IMPLEMENTED** | Hybrid: HDC-native (factual) or SLM-based (conversational) |

### Data Flow (All Layers)

```
User: "What is the capital of France?"
    ↓
[1. FractalSpace] — Query encoded as fractal vector (robust to noise)
    ↓
[3. MetacognitiveLoop] — Observes confidence, ready to retry if stuck
    ↓
[4. FactStore + ResponseSelector] — Finds "France → capital → Paris" via multi-strategy retrieval
    ↓
[3. MetacognitiveLoop] — Confidence is HIGH (0.9), no retry needed
    ↓
[5. Generator Selection] — ResonantGenerator (factual) or VentriloquistGenerator (conversational)
    ↓
Output: "The capital of France is Paris!" or "The capital of France is Paris! It's a beautiful city."
```

### What Each Layer Contributes

- **Without Fractals (Layer 1)**: Memory corrupts, vectors degrade catastrophically
- **Without Memory (Layer 2)**: No facts can be stored or bound together
- **Without Metacognition (Layer 3)**: System gives up on hard queries, no self-correction
- **Without Retrieval (Layer 4)**: Facts stored but cannot be found efficiently
- **Without Voice (Layer 5)**: Output sounds robotic ("France capital Paris")

**With All Five Layers**: Robust, self-aware, fluent responses with bounded hallucination.

> **Key Principle**: The Ventriloquist doesn't think. It only speaks what the Conscious Hologram tells it to say.

### The Bentov Model (HDC Foundations)

Based on Itzhak Bentov's holographic consciousness theory:

1. **Concepts as "Pebbles"**: Each concept is a random vector (a disturbance)
2. **Memory as "Water"**: Facts create interference patterns when superimposed
3. **Query as "Vibration"**: Querying resonates the surface to extract patterns

### HDC Operations

1. **Binding** (`torchhd.bind`): Combines two concepts into unique pattern

   ```python
   association = bind(subject, predicate)  # Creates unique key
   ```

2. **Bundling** (`torchhd.bundle`): Superimposes patterns (holographic storage)

   ```python
   memory = bundle(fact1, fact2, fact3)  # All facts in one vector
   ```

3. **Unbinding** (`bind(memory, inverse(key))`): Extracts associated value
   ```python
   value = bind(memory, inverse(key))  # Resonance extraction
   ```

### File Structure

```
kent_hologram/
├── src/hologram/
│   ├── core/           # HDC primitives (VectorSpace, Codebook, Operations, Resonator)
│   ├── memory/         # Holographic storage (MemoryTrace, FactStore)
│   ├── persistence/    # Save/load (StateManager, ChromaDB adapter)
│   ├── retrieval/      # Query systems (ConfidenceScorer)
│   ├── safety/         # Hallucination prevention (Refusal, Citation)
│   ├── cavity/         # Resonant Cavity (TargetEncoder, ReEncoder, Divergence)
│   ├── modulation/     # Style layer (SesameModulator)
│   ├── generation/     # Constrained generation (ResonantGenerator)
│   ├── conversation/   # Conversational learning (Intent, Entity, Memory, Patterns)
│   └── chat/           # Interactive interface
├── docs/               # Technical documentation
├── examples/           # Usage examples
└── tests/              # Test suite
```

## Resonant Cavity Architecture

The Resonant Cavity enables **creative exploration** ("jazz") while maintaining bounded hallucination. It adds a closed-loop verification system that ensures generated output stays aligned with stored knowledge.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    HologramContainer                          │
│              (shared Codebook + VectorSpace)                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ┌─────────┐          ┌───────────┐          ┌───────────┐
   │Resonator│          │  Sesame   │          │FactStore  │
   │  (ALS)  │          │Modulator  │          │(existing) │
   └────┬────┘          └─────┬─────┘          └───────────┘
        │                     │
        │ ResonatorResult     │ StyleVector
        ▼                     ▼
   ┌────────────────────────────────────────────────────────┐
   │                    TargetEncoder                        │
   └──────────────────────────┬─────────────────────────────┘
                              │ TargetPackage
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │                  ResonantGenerator                      │
   │                 (token-by-token loop)                   │
   └──────────────────────────┬─────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      ┌─────────┐      ┌────────────┐      ┌─────────┐
      │ReEncoder│─────▶│Divergence  │      │ Sesame  │
      │         │      │Calculator  │      │(fillers)│
      └─────────┘      └────────────┘      └─────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │GenerationResult │
                    └─────────────────┘
```

### Components

| Component                | Purpose                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------- |
| **Resonator**            | Decomposes thought vectors into (subject, verb, object) using Alternating Least Squares |
| **TargetEncoder**        | Packages Resonator output + style into constraint tensor                                |
| **ReEncoder**            | Projects generated tokens back to HDC space for verification                            |
| **DivergenceCalculator** | Measures drift between target and generated output (ACCEPT/REJECT)                      |
| **SesameModulator**      | Style vectors (formal/casual/urgent) + disfluency injection                             |
| **ResonantGenerator**    | Main orchestration loop combining all components                                        |

### Constrained Generation Example

```python
from hologram import HologramContainer, StyleType

# Setup
container = HologramContainer(dimensions=10000)
codebook = container.codebook

# Define vocabulary
vocabulary = {
    "nouns": ["cat", "dog", "fish", "mouse"],
    "verbs": ["eats", "chases", "catches"]
}

# Create generator
generator = container.create_resonant_generator(vocabulary)

# Create a thought vector: "cat eats fish"
from hologram.core.operations import Operations

cat = codebook.encode("cat")
eats = codebook.encode("eats")
fish = codebook.encode("fish")

thought = Operations.bundle(
    Operations.bind(cat, codebook.get_role("SUBJECT")),
    Operations.bind(eats, codebook.get_role("VERB")),
    Operations.bind(fish, codebook.get_role("OBJECT")),
)

# Generate with style
result = generator.generate(thought, style=StyleType.NEUTRAL)

print(f"Generated: {result.text}")
print(f"Tokens: {result.tokens}")
print(f"Acceptance rate: {result.metrics.acceptance_rate:.1%}")
print(f"Hallucination risk: {result.metrics.hallucination_risk:.1%}")
```

### Generation Metrics

The `GenerationResult` includes detailed metrics for auditing:

```python
@dataclass
class GenerationMetrics:
    total_tokens: int           # Total tokens generated
    accepted_first_try: int     # Passed verification immediately
    accepted_with_correction: int  # Accepted after correction signal
    rejected_tokens: int        # Failed verification, resampled
    fillers_injected: int       # "um", "uh" added for low confidence
    average_similarity: float   # Mean similarity to target
    acceptance_rate: float      # accepted / total
    hallucination_risk: float   # rejected / total
```

**Interpreting Metrics:**

- **acceptance_rate > 80%**: Generation is well-aligned with thought
- **hallucination_risk < 20%**: Low drift from constraints
- **fillers_injected > 0**: System expressed uncertainty naturally

### Style Types

```python
from hologram import StyleType

# Available styles
StyleType.FORMAL   # "therefore", "subsequently", "moreover"
StyleType.CASUAL   # "cool", "yeah", "like", "okay"
StyleType.URGENT   # "now", "immediately", "critical"
StyleType.NEUTRAL  # No style bias
```

### Disfluency Injection

When confidence drops below threshold (0.35), the system injects natural disfluencies:

```python
from hologram import FillerType

FillerType.UM      # "um" - mild uncertainty
FillerType.UH      # "uh" - moderate uncertainty
FillerType.PAUSE   # "..." - significant uncertainty
```

This creates more natural output that signals uncertainty to users.

## Tech Stack

- **Python 3.11+**
- **torchhd** - Hyperdimensional computing primitives
- **PyTorch** - Tensor operations
- **Pydantic** - Configuration management

## Key Concepts

### Bounded Hallucination

Unlike LLMs that can fabricate plausible-sounding nonsense, Hologram:

- Can only retrieve concepts from stored vocabulary
- Returns low confidence for unknown queries
- Refuses to answer when confidence < threshold
- **Cannot invent facts** - bounded by what was stored

### Deterministic Retrieval

- Same query always returns same result
- Cosine similarity is deterministic (no randomness)
- Confidence scores are reproducible
- Mathematical proof of whether fact is stored

### Graceful Degradation

- Noise in memory lowers confidence, doesn't change answer
- Holographic distribution allows partial corruption
- System degrades gracefully rather than catastrophically

## Development

Built following the original plan with corrections from validation agents (gemini-code and cursor-code):

**Implementation Phases:**

- ✅ **Phase 1**: Core HDC primitives with correct torchhd API
- ✅ **Phase 2**: Memory systems with holographic bundling
- ✅ **Phase 3**: Persistence (torch.save + JSON, not Faiss)
- ✅ **Phase 4**: Retrieval with confidence scoring
- ✅ **Phase 5**: Safety mechanisms (refusal, citation, bounded hallucination)
- ✅ **Phase 6**: Basic generation (minimal)
- ✅ **Phase 7**: Chat interface
- ✅ **Phase 8**: Resonant Cavity Architecture (v0.2.0)

**Critical Corrections Applied:**

- ✅ **torchhd API**: Fixed `unbind()` to use `bind(composite, inverse(key))`
- ✅ **Bundling**: Fixed to use pairwise bundling (torchhd limitation)
- ✅ **Confidence Thresholds**: Calibrated to holographic characteristics (0.20/0.10 instead of 0.60/0.30)
- ✅ **Package Structure**: Fixed hatch configuration for src-layout
- ✅ **Persistence**: Verified complete save/load cycle works perfectly

**v0.2.0 - Resonant Cavity Architecture:**

- ✅ **Resonator**: ALS solver for thought factorization
- ✅ **Cavity Components**: TargetEncoder, ReEncoder, DivergenceCalculator
- ✅ **Modulation**: SesameModulator with style vectors and disfluency
- ✅ **Generation**: ResonantGenerator with metrics and tracing
- ✅ **Container Integration**: Factory methods for all new components

**v0.3.0 - Conversational Learning:**

- ✅ **Intent Classification**: HDC-based, learned from examples (no keywords)
- ✅ **Teaching Detection**: TEACHING intent detects fact statements
- ✅ **Fact Extraction**: Token analysis (no regex) for subject/predicate/object
- ✅ **ChromaDB Persistence**: Facts persist across sessions with cosine distance
- ✅ **Style Tracking**: User style inference from message history
- ✅ **Conversation Memory**: Context-aware follow-up handling

## License

MIT

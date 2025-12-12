# Ventriloquist Architecture

The Ventriloquist Architecture implements a **hybrid generation system** that combines HDC-native generation (ResonantGenerator) with Small Language Model (SLM) generation (VentriloquistGenerator) for optimal performance.

## Overview

The architecture uses the right tool for the right job:

- **Factual Questions**: `ResonantGenerator` (HDC-native, bounded hallucination)
- **Conversational/Fluency**: `VentriloquistGenerator` (SLM for natural language)
- **Complex Reasoning**: `VentriloquistGenerator` (Reasoning Mode via GLM-4.6v)
- **Long Output**: `VentriloquistGenerator` (Multi-pass outline/expand/verify)

This hybrid approach ensures:

- **Accuracy**: HDC generator prevents hallucination for factual questions
- **Fluency**: SLM generator provides natural, conversational responses
- **Backward Compatibility**: Both generators coexist via shared protocol
- **Cost/Latency Scaling**: Token budgeting ensures requests stay within limits

## Architecture Components

### 1. GenerationContext (Interface Contract)

The `GenerationContext` dataclass provides a unified interface that carries BOTH text and vector information:

```python
@dataclass
class GenerationContext:
    query_text: str                         # For SLM (Ventriloquist)
    thought_vector: Optional[torch.Tensor]  # For HDC (ResonantGenerator)
    intent: IntentType
    fact_answer: Optional[str]
    entities: List[str]
    style: StyleType
    expected_subject: Optional[str] = None
    episodes: List[str] = field(default_factory=list) # Episodic memory context
```

This solves the critical interface mismatch: `ResponseSelector` passes `thought_vector`, but SLMs need text. The context carries both, allowing either generator type to work.

### 2. Generator Protocol

Both generators implement the same `Generator` protocol:

```python
class Generator(Protocol):
    def generate_with_validation(
        self,
        context: GenerationContext,
        max_tokens: int = 10,
    ) -> Optional[GenerationResult]:
        ...
```

This ensures type safety and allows hybrid routing without breaking changes.

### 3. ResonantGenerator (HDC-Native)

The `ResonantGenerator` uses HDC operations for constrained generation:

- **Input**: `thought_vector` from `GenerationContext`
- **Process**: Resonator factorization → Token-by-token generation with verification
- **Output**: Validated `GenerationResult` with trace and metrics
- **Use Case**: Factual questions where accuracy is critical

### 4. VentriloquistGenerator (SLM-Based)

The `VentriloquistGenerator` uses Novita API (Kimi/GLM models) for fluent generation. It includes several advanced pipelines:

#### A. Standard Fluency Mode (Kimi K2)
- **Input**: `query_text` and `fact_answer`
- **Process**: Construct prompt → Call SLM API → Validate fact incorporation
- **Output**: Fluent natural language response
- **Budgeting**: Uses `_budget_tokens()` to safely cap prompt length

#### B. Reasoning Mode (GLM-4.6v)
- **Method**: `generate_reasoning_chain()`
- **Process**: Generates step-by-step deduction chain in JSON
- **Verification**: Each step is checked against the FactStore for grounding
- **Use Case**: Complex multi-hop queries

#### C. Long-Output Pipeline (Multi-Pass)
- **Method**: `generate_long_form()`
- **Process**:
    1. **Outline**: Generate JSON section plan (Reasoning Model)
    2. **Expand**: Generate text for each section independently (Fluency Model)
    3. **Verify**: Check each section for fact grounding
- **Use Case**: Reports, summaries, long explanations

#### D. Code Generation (Dual-Retrieval)
- **Method**: `generate_code_with_context()`
- **Process**: Retrieves from both `ConceptStore` (patterns) and `ProjectFactStore` (APIs)
- **Verification**: `verify_code()` checks imports, function calls, and classes against index
- **Use Case**: Accurate, project-aware coding assistance

## Hybrid Routing Logic

The `ResponseSelector` implements intelligent routing:

```python
if is_factual_question and fact_answer:
    use ResonantGenerator()  # HDC ensures answer is grounded
else:
    use VentriloquistGenerator()  # SLM for fluent conversation
```

**Routing Criteria:**

- **Factual Questions** (with high confidence fact answer): Use HDC generator
- **Conversational** (or low confidence): Use SLM generator
- **Voice Mode**: Can be toggled ("off", "rewrite-only", "full")

## Configuration

### Environment Variables

Add to `.env`:

```bash
NOVITA_API_KEY=your_novita_api_key_here
```

Get your API key at: https://novita.ai/

### Container Setup

```python
from hologram.container import HologramContainer

container = HologramContainer(dimensions=10000)

# Create chatbot with hybrid generation
chatbot = container.create_persistent_chatbot(
    enable_generation=True,      # Enable ResonantGenerator
    enable_ventriloquist=True,    # Enable VentriloquistGenerator
    vocabulary=vocab_dict,
)
```

### Crew Trainer Configuration

The `crew_trainer.py` script automatically configures hybrid mode:

```python
self.chatbot = self.container.create_persistent_chatbot(
    enable_generation=True,
    enable_ventriloquist=True,
    ventriloquist_model="moonshotai/kimi-k2-thinking",
)
```

## Implementation Details

### GenerationContext Flow

1. **ResponseSelector** builds `GenerationContext` with:

   - `query_text`: Original user input
   - `thought_vector`: HDC thought vector (if facts available)
   - `fact_answer`: Retrieved fact (if available)
   - `entities`, `intent`, `style`: Context information
   - `episodes`: Retrieved episodic memories

2. **Hybrid Routing** selects generator based on:

   - Question type (factual vs conversational)
   - Fact availability and confidence
   - Generator availability

3. **Generator** processes context:
   - **ResonantGenerator**: Uses `thought_vector` for HDC generation
   - **VentriloquistGenerator**: Uses `query_text` + `fact_answer` for SLM generation

### Validation

Both generators validate output:

- **ResonantGenerator**: Validates against `expected_subject` and `fact_answer`
- **VentriloquistGenerator**: Validates that `fact_answer` appears in generated text
- **Code Verification**: Checks AST against known symbols

If validation fails, generators return `None`, causing fallback to template responses.

## Benefits

1. **Zero Breaking Changes**: Both generators coexist via shared protocol
2. **Explicit Contracts**: Type errors caught at compile time
3. **Backward Compatible**: Existing ResonantGenerator still works
4. **Testable**: Each generator can be tested independently
5. **Optimal Performance**: Right tool for right job

## Files Modified/Created

1. ✅ `src/hologram/generation/base.py` - Interface contracts (`GenerationContext`, `Generator` protocol)
2. ✅ `src/hologram/generation/ventriloquist.py` - SLM wrapper using Novita API
3. ✅ `src/hologram/generation/resonant_generator.py` - Implements protocol
4. ✅ `src/hologram/conversation/selector.py` - Uses `GenerationContext` for hybrid routing
5. ✅ `src/hologram/container.py` - Wiring (`create_ventriloquist_generator`)
6. ✅ `scripts/crew_trainer.py` - Hybrid mode configuration
7. ✅ `.env.example` - Added `NOVITA_API_KEY` placeholder
8. ✅ `docs/VENTRILOQUIST_ARCHITECTURE.md` - This documentation

## Risk Mitigation

- **Zero Breaking Changes**: Both generators coexist via shared protocol
- **Explicit Contracts**: No silent failures (type errors caught at compile time)
- **Backward Compatible**: Existing ResonantGenerator still works
- **Testable**: Each generator can be tested independently

## Future Enhancements

Potential improvements:

1. **Dynamic Routing**: Learn which generator performs better for different question types
2. **Confidence Thresholds**: Adjust routing thresholds based on performance metrics
3. **Fallback Chain**: Try HDC first, then SLM, then templates
4. **Cost Optimization**: Route based on API cost vs quality tradeoffs

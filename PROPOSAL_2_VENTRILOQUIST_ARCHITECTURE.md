# Proposal 2: Ventriloquist Architecture

## Technical Specification for HDC + SLM Hybrid Generation

**Version**: 1.0
**Date**: 2025-12-09
**Status**: Proposed
**Estimated Effort**: 1-2 weeks
**Risk Level**: Medium (New dependency, architecture change)
**Inspired By**: [Sesame AI Conversational Speech](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)

---

## Executive Summary

The Ventriloquist Architecture separates **Logic** (HDC) from **Fluency** (SLM). HDC remains the "Brain" - responsible for fact retrieval and semantic constraints. A small local language model (SLM) becomes the "Voice Box" - responsible for natural language generation.

This solves Hologram's fundamental weakness: **rigid, unnatural text output** that fails LLM benchmarks for fluency and perplexity.

**Key Insight from Sesame AI:**

> "Voice is our most intimate medium... Without unlocking the full power of voice, they cannot hope to effectively collaborate with us."

The same principle applies to text. HDC produces correct but sterile output ("The capital of France is Paris."). Users expect fluid, contextual responses ("Paris! Known for the Eiffel Tower, it's been France's capital since...").

---

## Problem Statement

### Current Output Quality (resonant_generator.py)

The `ResonantGenerator` produces S-V-O triplets:

```python
# Current output examples
"France capital Paris"           # ❌ Not a sentence
"The capital is Paris"           # ❌ No subject
"Australia capital is Canberra"  # ❌ Broken grammar
```

### Why HDC Fails at Fluency

1. **Fixed Vocabulary**: Generator can only output words in the codebook
2. **No Morphology**: Can't conjugate verbs or pluralize nouns
3. **No Function Words**: Missing "the", "a", "is", "of" unless explicitly encoded
4. **No Sentence Structure**: Resonator outputs fillers for slots, not sentences

### Benchmark Performance (Current)

| Benchmark           | Hologram (Current)    | GPT-4 | Target         |
| ------------------- | --------------------- | ----- | -------------- |
| MMLU (Knowledge)    | ~0% (empty KB)        | 86%   | 50%+ (with KB) |
| HellaSwag (Fluency) | ~10% (random)         | 95%   | 70%+           |
| Perplexity          | ∞ (not probabilistic) | ~3.5  | <10            |

---

## Proposed Solution: The Ventriloquist

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VENTRILOQUIST PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   User       │     │    HDC       │     │        SLM           │ │
│  │   Query      │ --> │   BRAIN      │ --> │      VOICE BOX       │ │
│  │              │     │              │     │                      │ │
│  │ "What is the │     │ Retrieves:   │     │ Generates:           │ │
│  │  capital of  │     │ - Subject:   │     │ "Paris is the        │ │
│  │  France?"    │     │   France     │     │  capital of France,  │ │
│  │              │     │ - Predicate: │     │  a beautiful city    │ │
│  │              │     │   capital    │     │  known for..."       │ │
│  │              │     │ - Object:    │     │                      │ │
│  │              │     │   Paris      │     │ [Constrained by HDC] │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│                              │                        ▲              │
│                              │    SEMANTIC ANCHORS    │              │
│                              └────────────────────────┘              │
│                                                                       │
│                       ┌──────────────────────┐                       │
│                       │    VALIDATOR         │                       │
│                       │ Ensures SLM output   │                       │
│                       │ contains all anchors │                       │
│                       └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### The Contract

1. **HDC Brain** guarantees: "The answer contains these exact entities"
2. **SLM Voice** guarantees: "The output is grammatically correct and fluent"
3. **Validator** guarantees: "The final output satisfies both constraints"

---

## Implementation Details

### Component 1: Semantic Anchor Extractor

Extracts constrained anchors from HDC:

```python
# src/hologram/ventriloquist/anchor_extractor.py

from dataclasses import dataclass
from typing import List, Optional
import torch

from hologram.memory.fact_store import FactStore
from hologram.core.resonator import Resonator, ResonatorResult


@dataclass
class SemanticAnchor:
    """A semantic constraint for the SLM."""

    text: str           # The word/phrase that MUST appear
    role: str           # SUBJECT, PREDICATE, OBJECT
    confidence: float   # HDC confidence (0.0-1.0)
    required: bool      # If True, output MUST contain this

    def __str__(self) -> str:
        req = "REQUIRED" if self.required else "optional"
        return f"[{self.role}:{self.text}]({req}, conf={self.confidence:.2f})"


@dataclass
class AnchorPackage:
    """Complete anchor set for one generation."""

    anchors: List[SemanticAnchor]
    question_type: str  # "what", "who", "where", "when", "how", "why"
    style_hint: str     # "formal", "casual", "educational"
    max_length: int     # Maximum output tokens

    def get_required_anchors(self) -> List[str]:
        """Get all required anchor texts."""
        return [a.text for a in self.anchors if a.required]

    def to_constraint_string(self) -> str:
        """Format for SLM prompt."""
        required = [a.text for a in self.anchors if a.required]
        optional = [a.text for a in self.anchors if not a.required]

        parts = []
        if required:
            parts.append(f"MUST INCLUDE: {', '.join(required)}")
        if optional:
            parts.append(f"MAY INCLUDE: {', '.join(optional)}")
        return " | ".join(parts)


class AnchorExtractor:
    """
    Extracts semantic anchors from HDC for SLM generation.

    This is the bridge between the HDC world (vectors) and the
    SLM world (text constraints).
    """

    def __init__(
        self,
        fact_store: FactStore,
        resonator: Resonator,
        confidence_threshold: float = 0.5
    ):
        self._fact_store = fact_store
        self._resonator = resonator
        self._confidence_threshold = confidence_threshold

    def extract(
        self,
        query_text: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> AnchorPackage:
        """
        Extract anchors for a query.

        Args:
            query_text: Original user query
            subject: Extracted subject entity (if known)
            predicate: Extracted predicate (if known)

        Returns:
            AnchorPackage with semantic constraints
        """
        anchors = []

        # Query fact store
        if subject and predicate:
            answer, confidence = self._fact_store.query(subject, predicate)

            if confidence >= self._confidence_threshold:
                # Subject anchor (from query)
                anchors.append(SemanticAnchor(
                    text=subject,
                    role="SUBJECT",
                    confidence=1.0,  # User provided
                    required=True
                ))

                # Predicate anchor (from query)
                anchors.append(SemanticAnchor(
                    text=predicate,
                    role="PREDICATE",
                    confidence=1.0,
                    required=False  # Can be paraphrased
                ))

                # Object anchor (from fact store)
                anchors.append(SemanticAnchor(
                    text=answer,
                    role="OBJECT",
                    confidence=confidence,
                    required=True  # THE ANSWER - must appear
                ))

        # Detect question type
        question_type = self._detect_question_type(query_text)

        # Infer style from query
        style_hint = self._infer_style(query_text)

        return AnchorPackage(
            anchors=anchors,
            question_type=question_type,
            style_hint=style_hint,
            max_length=50
        )

    def _detect_question_type(self, text: str) -> str:
        """Detect question type for appropriate response structure."""
        text_lower = text.lower()

        if text_lower.startswith("what"):
            return "what"
        elif text_lower.startswith("who"):
            return "who"
        elif text_lower.startswith("where"):
            return "where"
        elif text_lower.startswith("when"):
            return "when"
        elif text_lower.startswith("how"):
            return "how"
        elif text_lower.startswith("why"):
            return "why"
        else:
            return "what"  # Default

    def _infer_style(self, text: str) -> str:
        """Infer desired response style."""
        # Simple heuristic - could be enhanced
        if "please" in text.lower() or "could you" in text.lower():
            return "formal"
        elif "?" in text and len(text) < 30:
            return "concise"
        else:
            return "educational"
```

### Component 2: SLM Voice Box

The language model wrapper with constrained generation:

```python
# src/hologram/ventriloquist/voice_box.py

from dataclasses import dataclass
from typing import List, Optional, Callable
import torch

from hologram.ventriloquist.anchor_extractor import AnchorPackage, SemanticAnchor


@dataclass
class VoiceBoxConfig:
    """Configuration for the SLM."""

    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    max_new_tokens: int = 100
    temperature: float = 0.7
    device: str = "cpu"  # or "cuda" if available

    # Constraint enforcement
    retry_on_missing_anchor: bool = True
    max_retries: int = 3


class VoiceBox:
    """
    Small Language Model for natural language generation.

    The "Voice Box" - takes semantic anchors from HDC and produces
    fluent natural language that MUST contain those anchors.

    Uses constrained generation to ensure HDC facts appear in output.
    """

    def __init__(self, config: Optional[VoiceBoxConfig] = None):
        self._config = config or VoiceBoxConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Lazy-load the model (memory expensive)."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name
            ).to(self._config.device)
            self._loaded = True

        except ImportError:
            raise ImportError(
                "VoiceBox requires transformers. Install with: "
                "pip install transformers torch"
            )

    def speak(
        self,
        anchors: AnchorPackage,
        question: str,
    ) -> str:
        """
        Generate natural language containing all anchors.

        Args:
            anchors: Semantic anchors from HDC
            question: Original user question

        Returns:
            Fluent text containing all required anchors
        """
        self.load()

        # Build constrained prompt
        prompt = self._build_prompt(anchors, question)

        # Generate with retries
        for attempt in range(self._config.max_retries):
            output = self._generate(prompt)

            # Validate anchors present
            if self._validate_anchors(output, anchors):
                return output

            # Retry with stronger constraint language
            if attempt < self._config.max_retries - 1:
                prompt = self._strengthen_prompt(prompt, anchors, output)

        # Fallback: Force anchors into template
        return self._fallback_generation(anchors, question)

    def _build_prompt(self, anchors: AnchorPackage, question: str) -> str:
        """Build prompt for SLM with constraints."""

        required = anchors.get_required_anchors()
        constraints = anchors.to_constraint_string()

        # Prompt template optimized for small models
        prompt = f"""You are a helpful assistant. Answer the question naturally.

QUESTION: {question}

CONSTRAINTS: Your answer {constraints}

STYLE: {anchors.style_hint}

Answer in one clear sentence:"""

        return prompt

    def _strengthen_prompt(
        self,
        original_prompt: str,
        anchors: AnchorPackage,
        failed_output: str
    ) -> str:
        """Strengthen prompt after failed validation."""

        missing = self._get_missing_anchors(failed_output, anchors)

        return f"""{original_prompt}

IMPORTANT: Your previous answer was missing: {', '.join(missing)}
You MUST include these exact words. Try again:"""

    def _generate(self, prompt: str) -> str:
        """Generate text from prompt."""

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self._config.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode and extract answer only
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        answer = full_text[len(prompt):].strip()

        # Clean up
        answer = answer.split("\n")[0]  # Take first line only

        return answer

    def _validate_anchors(self, output: str, anchors: AnchorPackage) -> bool:
        """Check if all required anchors appear in output."""
        output_lower = output.lower()

        for anchor in anchors.anchors:
            if anchor.required:
                if anchor.text.lower() not in output_lower:
                    return False

        return True

    def _get_missing_anchors(
        self,
        output: str,
        anchors: AnchorPackage
    ) -> List[str]:
        """Get list of missing required anchors."""
        output_lower = output.lower()
        missing = []

        for anchor in anchors.anchors:
            if anchor.required:
                if anchor.text.lower() not in output_lower:
                    missing.append(anchor.text)

        return missing

    def _fallback_generation(
        self,
        anchors: AnchorPackage,
        question: str
    ) -> str:
        """
        Fallback: Template-based generation if SLM fails.

        This ensures we ALWAYS return a valid answer.
        """
        # Extract key anchors
        subject = None
        predicate = None
        obj = None

        for anchor in anchors.anchors:
            if anchor.role == "SUBJECT":
                subject = anchor.text
            elif anchor.role == "PREDICATE":
                predicate = anchor.text
            elif anchor.role == "OBJECT":
                obj = anchor.text

        # Template based on question type
        templates = {
            "what": f"The {predicate} of {subject} is {obj}.",
            "who": f"{obj} is the {predicate} of {subject}.",
            "where": f"{subject} is located in {obj}.",
            "when": f"{subject} {predicate} in {obj}.",
        }

        template = templates.get(anchors.question_type, templates["what"])

        return template
```

### Component 3: Ventriloquist Orchestrator

The main coordination layer:

```python
# src/hologram/ventriloquist/orchestrator.py

from dataclasses import dataclass
from typing import Optional

from hologram.ventriloquist.anchor_extractor import AnchorExtractor, AnchorPackage
from hologram.ventriloquist.voice_box import VoiceBox, VoiceBoxConfig
from hologram.memory.fact_store import FactStore
from hologram.core.resonator import Resonator
from hologram.conversation.entity import Entity


@dataclass
class VentriloquistResult:
    """Result of ventriloquist generation."""

    text: str                    # Final output
    anchors: AnchorPackage       # Semantic constraints used
    source: str                  # "slm" or "fallback"
    hdc_confidence: float        # HDC fact confidence
    validated: bool              # Whether anchors were validated

    def __str__(self) -> str:
        return f'VentriloquistResult("{self.text}", source={self.source})'


class Ventriloquist:
    """
    Main orchestrator for HDC + SLM hybrid generation.

    Coordinates:
    1. AnchorExtractor - Gets semantic constraints from HDC
    2. VoiceBox - Generates fluent text via SLM
    3. Validation - Ensures HDC constraints are satisfied

    Example:
        >>> vent = Ventriloquist(fact_store, resonator)
        >>> result = vent.respond("What is the capital of France?",
        ...                       subject="France", predicate="capital")
        >>> print(result.text)
        "Paris is the beautiful capital city of France."
    """

    def __init__(
        self,
        fact_store: FactStore,
        resonator: Resonator,
        voice_config: Optional[VoiceBoxConfig] = None,
        confidence_threshold: float = 0.5
    ):
        self._fact_store = fact_store
        self._resonator = resonator

        # Initialize components
        self._extractor = AnchorExtractor(
            fact_store=fact_store,
            resonator=resonator,
            confidence_threshold=confidence_threshold
        )
        self._voice = VoiceBox(config=voice_config)
        self._confidence_threshold = confidence_threshold

    def respond(
        self,
        question: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        entities: Optional[list[Entity]] = None,
    ) -> VentriloquistResult:
        """
        Generate a fluent response to a question.

        Args:
            question: User's question
            subject: Extracted subject (or inferred from entities)
            predicate: Extracted predicate (e.g., "capital", "population")
            entities: Extracted entities from question

        Returns:
            VentriloquistResult with fluent text
        """
        # Extract subject from entities if not provided
        if subject is None and entities:
            subject = entities[0].canonical_form if entities else None

        # Extract anchors from HDC
        anchors = self._extractor.extract(
            query_text=question,
            subject=subject,
            predicate=predicate
        )

        # Check if HDC has an answer
        hdc_confidence = 0.0
        for anchor in anchors.anchors:
            if anchor.role == "OBJECT":
                hdc_confidence = anchor.confidence
                break

        if hdc_confidence < self._confidence_threshold:
            # HDC doesn't know - return refusal
            return VentriloquistResult(
                text="I don't have information about that.",
                anchors=anchors,
                source="refusal",
                hdc_confidence=hdc_confidence,
                validated=True
            )

        # Generate via SLM
        try:
            text = self._voice.speak(anchors, question)

            # Validate
            validated = self._voice._validate_anchors(text, anchors)

            return VentriloquistResult(
                text=text,
                anchors=anchors,
                source="slm",
                hdc_confidence=hdc_confidence,
                validated=validated
            )

        except Exception as e:
            # SLM failed - use fallback
            fallback_text = self._voice._fallback_generation(anchors, question)

            return VentriloquistResult(
                text=fallback_text,
                anchors=anchors,
                source="fallback",
                hdc_confidence=hdc_confidence,
                validated=True  # Fallback always includes anchors
            )

    def is_slm_loaded(self) -> bool:
        """Check if SLM is loaded (for resource management)."""
        return self._voice._loaded

    def preload_slm(self) -> None:
        """Pre-load SLM for faster first response."""
        self._voice.load()
```

### Integration with ResponseSelector

Modify `selector.py` to use Ventriloquist:

```python
# src/hologram/conversation/selector.py - Modified

class ResponseSelector:
    def __init__(
        self,
        pattern_store: ResponsePatternStore,
        conversation_memory: ConversationMemory,
        fact_store: Optional[FactStore],
        codebook: Codebook,
        response_corpus: Optional[ResponseCorpus] = None,
        resonant_generator: Optional[ResonantGenerator] = None,
        ventriloquist: Optional[Ventriloquist] = None,  # NEW
    ):
        # ... existing init ...
        self._ventriloquist = ventriloquist

    def select(
        self,
        intent: IntentResult,
        entities: List[Entity],
        text: str,
        style: Optional[StyleType] = None,
    ) -> ResponseCandidate:
        # ... existing logic ...

        # NEW: Try Ventriloquist for questions if available
        if is_question and self._ventriloquist:
            # Extract predicate from question
            predicate = self._extract_predicate(text)
            subject = entity_names[0] if entity_names else None

            result = self._ventriloquist.respond(
                question=text,
                subject=subject,
                predicate=predicate,
                entities=entities
            )

            if result.source != "refusal":
                return ResponseCandidate(
                    pattern=None,  # No pattern - generated
                    filled_response=result.text,
                    thought_vector=context_vec,
                    confidence=result.hdc_confidence,
                    fact_answer=result.anchors.get_required_anchors()[-1] if result.anchors.anchors else None
                )

        # ... fall through to existing logic ...
```

---

## Pitfalls & Mitigations

### Pitfall 1: SLM Ignores Constraints

**Risk:** Small language models may ignore prompt constraints and hallucinate.

**Mitigation:** Implement constrained decoding with logit bias:

```python
def _generate_constrained(self, prompt: str, required_tokens: List[str]) -> str:
    """Generate with hard token constraints."""

    # Get token IDs for required words
    required_ids = []
    for word in required_tokens:
        ids = self._tokenizer.encode(word, add_special_tokens=False)
        required_ids.extend(ids)

    # Create logit bias: boost required tokens
    logit_bias = {id: 5.0 for id in required_ids}  # +5 logit boost

    # Generate with bias
    outputs = self._model.generate(
        **inputs,
        bad_words_ids=None,
        force_words_ids=[required_ids],  # Force these tokens
        max_new_tokens=self._config.max_new_tokens,
    )

    return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Pitfall 2: Latency

**Risk:** Loading and running SLM adds 200-500ms latency.

**Mitigation:**

1. **Lazy loading**: Only load SLM on first use
2. **Pre-loading**: Load during startup in background thread
3. **Caching**: Cache SLM outputs for repeated queries
4. **Fallback fast path**: Use template if latency budget exceeded

```python
class VoiceBox:
    def speak_with_timeout(
        self,
        anchors: AnchorPackage,
        question: str,
        timeout_ms: float = 200
    ) -> str:
        """Generate with latency constraint."""
        import concurrent.futures
        import time

        start = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.speak, anchors, question)

            try:
                result = future.result(timeout=timeout_ms / 1000)
                return result
            except concurrent.futures.TimeoutError:
                # Fallback to template
                return self._fallback_generation(anchors, question)
```

### Pitfall 3: Memory Footprint

**Risk:** Even small LMs use 500MB+ RAM.

**Mitigation:**

1. **Use quantized models**: 4-bit quantization reduces memory by 75%
2. **Offload when idle**: Unload model after 60s of inactivity
3. **CPU-only option**: Avoid GPU memory requirements

```python
class VoiceBoxConfig:
    # ... existing ...
    quantize: bool = True  # Use 4-bit quantization
    offload_timeout: float = 60.0  # Unload after 60s idle

class VoiceBox:
    def load(self) -> None:
        if self._loaded:
            return

        if self._config.quantize:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name,
                quantization_config=quantization_config
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name
            )
```

### Pitfall 4: Prompt Injection

**Risk:** User query may contain instructions that override constraints.

**Mitigation:** Sanitize user input before prompt construction:

```python
def _sanitize_input(self, text: str) -> str:
    """Remove potential prompt injection attacks."""

    # Remove instruction-like patterns
    dangerous_patterns = [
        r"ignore.*instructions",
        r"forget.*above",
        r"new instructions:",
        r"system:",
        r"assistant:",
    ]

    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)

    return sanitized
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_ventriloquist.py

def test_anchors_appear_in_output():
    """SLM output must contain all required anchors."""
    voice = VoiceBox()

    anchors = AnchorPackage(
        anchors=[
            SemanticAnchor("France", "SUBJECT", 1.0, True),
            SemanticAnchor("capital", "PREDICATE", 1.0, False),
            SemanticAnchor("Paris", "OBJECT", 0.9, True),
        ],
        question_type="what",
        style_hint="educational",
        max_length=50
    )

    output = voice.speak(anchors, "What is the capital of France?")

    assert "france" in output.lower()
    assert "paris" in output.lower()

def test_fallback_on_slm_failure():
    """Fallback should be used if SLM fails to include anchors."""
    # ... test with mocked failing SLM ...

def test_low_confidence_returns_refusal():
    """Low HDC confidence should return refusal."""
    # ... test with confidence < threshold ...
```

### Integration Tests

```python
def test_full_pipeline():
    """Test end-to-end ventriloquist pipeline."""
    # Setup
    fact_store = FactStore(VectorSpace(), Codebook(VectorSpace()))
    fact_store.add_fact("France", "capital", "Paris")

    vent = Ventriloquist(fact_store, resonator)

    # Test
    result = vent.respond(
        "What is the capital of France?",
        subject="France",
        predicate="capital"
    )

    # Assertions
    assert "paris" in result.text.lower()
    assert result.source == "slm"
    assert result.hdc_confidence > 0.5
    assert result.validated

def test_unknown_fact_refusal():
    """Unknown facts should return refusal."""
    fact_store = FactStore(VectorSpace(), Codebook(VectorSpace()))
    # No facts added

    vent = Ventriloquist(fact_store, resonator)

    result = vent.respond(
        "What is the capital of Narnia?",
        subject="Narnia",
        predicate="capital"
    )

    assert result.source == "refusal"
    assert "don't have information" in result.text.lower()
```

### Benchmark Tests

```python
def test_fluency_improvement():
    """Ventriloquist should improve fluency metrics."""

    # Generate 100 responses with current system
    current_outputs = [generate_current(q) for q in test_questions]

    # Generate 100 responses with Ventriloquist
    vent_outputs = [vent.respond(q).text for q in test_questions]

    # Measure fluency (perplexity with reference LLM)
    current_perplexity = measure_perplexity(current_outputs)
    vent_perplexity = measure_perplexity(vent_outputs)

    assert vent_perplexity < current_perplexity * 0.5  # 50% improvement
```

---

## Feature Complete Checklist

| Feature                      | Status | Acceptance Criteria                    |
| ---------------------------- | ------ | -------------------------------------- |
| AnchorExtractor              | 🔲     | Extracts S-P-O anchors from fact query |
| VoiceBox                     | 🔲     | Generates fluent text via SLM          |
| Constraint validation        | 🔲     | All required anchors appear in output  |
| Retry mechanism              | 🔲     | 3 retries with strengthened prompt     |
| Fallback generation          | 🔲     | Template-based fallback always works   |
| ResponseSelector integration | 🔲     | Ventriloquist used for questions       |
| Lazy loading                 | 🔲     | SLM only loaded on first use           |
| Quantization support         | 🔲     | 4-bit quantization option              |
| Timeout handling             | 🔲     | Fallback if generation exceeds 200ms   |
| Unit tests                   | 🔲     | All tests pass                         |
| Integration tests            | 🔲     | Full pipeline works                    |
| Benchmark improvement        | 🔲     | Perplexity reduced by 50%+             |

---

## Dependencies

### New Python Packages

```
# requirements.txt additions
transformers>=4.36.0
torch>=2.0.0
accelerate>=0.25.0  # For model loading
bitsandbytes>=0.41.0  # For quantization (optional)
```

### Recommended Models

| Model                | Size | Speed  | Quality | Use Case         |
| -------------------- | ---- | ------ | ------- | ---------------- |
| SmolLM-135M-Instruct | 135M | Fast   | Good    | Default          |
| TinyLlama-1.1B       | 1.1B | Medium | Better  | Quality priority |
| Phi-2                | 2.7B | Slow   | Best    | Benchmarks       |

---

## Rollout Plan

### Phase 1: Core Components (Days 1-3)

- [ ] Implement `AnchorExtractor`
- [ ] Implement `VoiceBox`
- [ ] Write unit tests

### Phase 2: Integration (Days 4-7)

- [ ] Implement `Ventriloquist` orchestrator
- [ ] Modify `ResponseSelector`
- [ ] Add to `container.py` DI

### Phase 3: Optimization (Days 8-10)

- [ ] Add quantization support
- [ ] Implement caching
- [ ] Add timeout handling

### Phase 4: Validation (Days 11-14)

- [ ] Run benchmark suite
- [ ] Measure latency impact
- [ ] Document findings

---

## Expected Impact

| Metric               | Before | After (Expected) | Improvement       |
| -------------------- | ------ | ---------------- | ----------------- |
| HellaSwag (Fluency)  | ~10%   | 70%+             | 7x                |
| Response naturalness | Rigid  | Fluid            | Qualitative       |
| User satisfaction    | Low    | High             | Qualitative       |
| Latency (P50)        | 50ms   | 250ms            | -5x (trade-off)   |
| Memory usage         | 200MB  | 700MB            | -3.5x (trade-off) |

---

## References

1. [Sesame AI - Crossing the Uncanny Valley](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
2. [SmolLM - Small Language Models](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
3. `src/hologram/generation/resonant_generator.py` - Current generation
4. `src/hologram/conversation/selector.py` - Integration point

---

**Document Control**

- **Author**: Engineering Team
- **Reviewers**: TBD
- **Approval**: TBD
- **Last Updated**: 2025-12-09

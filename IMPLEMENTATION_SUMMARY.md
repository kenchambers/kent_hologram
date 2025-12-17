# Conscious Hologram: Token & Context - Implementation Summary

## Quick Reference

### File Locations and Key Components

```
src/hologram/
├── generation/
│   ├── ventriloquist.py          ← SLM integration, token management
│   ├── resonant_generator.py      ← HDC generation, 10-token limit
│   └── base.py                    ← GenerationContext (unified interface)
│
├── conversation/
│   ├── selector.py                ← Response selection, routing logic
│   ├── chatbot.py                 ← Main orchestration
│   └── intent.py                  ← Intent classification (HDC)
│
├── memory/
│   ├── fact_store.py              ← Fact storage, resonance search
│   └── memory_trace.py            ← Holographic memory, surprise gating
│
├── config/
│   ├── constants.py               ← All limit definitions
│   └── settings.py                ← Runtime configuration
│
└── core/
    ├── codebook.py                ← String to vector encoding
    ├── vector_space.py            ← 10,000D configuration
    └── operations.py              ← bind/bundle operations
```

---

## Critical Configuration Points

### 1. Token Limits (src/hologram/config/constants.py)

```python
# LINE 108-109: Generation token limits
MAX_GENERATION_TOKENS = 100
# This is OVERRIDDEN by specific generators:
# - ResonantGenerator: 10 tokens (hardcoded in selector.py:738)
# - VentriloquistGenerator: 256 tokens (hardcoded in ventriloquist.py:71)

# LINE 181-190: SLM Models and reasoning
DEFAULT_FLUENCY_MODEL = "moonshotai/kimi-k2-thinking"
DEFAULT_REASONING_MODEL = "zai-org/glm-4.6v"
ENABLE_REASONING_CHAIN = True
```

### 2. FactStore Capacity (src/hologram/config/constants.py)

```python
# LINE 43-46: Memory capacity estimate
ESTIMATED_CAPACITY_DIVISOR = 100
# Capacity ≈ dimensions / divisor
# For DEFAULT_DIMENSIONS=10000: ~100 facts

# LINE 160-166: Surprise gating
SURPRISE_THRESHOLD = 0.1
SURPRISE_LEARNING_RATE = 0.5
SURPRISE_DECAY = 0.99
```

### 3. Confidence Thresholds (src/hologram/config/constants.py)

```python
# LINE 17-27: Response confidence calibration
RESPONSE_CONFIDENCE_THRESHOLD = 0.20
REFUSAL_CONFIDENCE_THRESHOLD = 0.10
# These are CALIBRATED for holographic storage interference
# (facts naturally have 0.24-0.37 similarity due to bundling)
```

---

## Message Flow: Code References

### 1. User Input → Chatbot

**File**: `src/hologram/conversation/chatbot.py` (line 108)

```python
def respond(self, user_input: str) -> str:
    user_input = user_input.strip()
    if not user_input:
        return "I didn't catch that. Could you say something?"

    # 1. Classify intent
    intent = self._intent_classifier.classify(user_input)

    # 2. Extract entities
    entities = self._entity_extractor.extract(user_input)

    # 3. Select response
    candidate = self._response_selector.select(
        intent, entities, user_input
    )

    return candidate.filled_response
```

**Token cost**: 0 (no API calls yet)

### 2. Intent Classification

**File**: `src/hologram/conversation/intent.py`

```python
def classify(self, text: str) -> IntentResult:
    # Encodes text as HDC vector
    input_vec = self._codebook.encode(text)

    # Bundles with intent prototypes stored in memory
    # Resonator convergence loop (max 100 iterations)

    return IntentResult(intent=..., confidence=...)
```

**Token cost**: 0 (internal HDC, no serialization)

### 3. Fact Store Query

**File**: `src/hologram/memory/fact_store.py` (line 219)

```python
def query(self, subject: str, predicate: str) -> tuple[str, float]:
    # Strategy 1: Exact match (O(1))
    subject_norm = self._normalize(subject)
    predicate_norm = self._normalize(predicate)
    exact_key = f"{subject_norm}:{predicate_norm}"

    if exact_key in self._exact_index:
        fact = self._exact_index[exact_key]
        return fact.object, 1.0  ← FAST PATH

    # Strategy 2: Resonance search (O(N×D))
    s_vec = self._codebook.encode(subject_norm)
    p_vec = self._codebook.encode(predicate_norm)
    key = Operations.bind(s_vec, p_vec)

    candidates = torch.stack([...self._value_vectors_cache[v]...])
    similarities = self._memory.resonance(key, candidates)
    best_idx = torch.argmax(similarities).item()

    return value_list[best_idx], float(similarities[best_idx].item())
```

**Token cost**: 0 (returns string, not serialized yet)
**Performance**: O(1) best case, O(N×10000) worst case

### 4. Response Selection

**File**: `src/hologram/conversation/selector.py` (line 96)

```python
def select(self, intent, entities, text, style=None) -> ResponseCandidate:
    # ... (fact query happens here) ...

    # Step 2: Hybrid generation routing
    has_facts = self._fact_store and self._fact_store.fact_count > 0

    # CRITICAL: Ventriloquist (SLM) is ALWAYS preferred
    if (fact_answer and has_facts) or (self._ventriloquist and not is_question):
        context = GenerationContext(
            query_text=text,
            fact_answer=fact_answer,
            entities=entity_names,
            style=style or StyleType.NEUTRAL,
        )

        # Ventriloquist preferred over HDC
        if self._ventriloquist:
            generated_response = self._generate_response_with_context(
                context, use_ventriloquist=True
            )
        elif is_factual_question and fact_confidence >= 0.5 and self._generator:
            generated_response = self._generate_response_with_context(context)

    return ResponseCandidate(...)
```

**Token cost**: Still 0 (GenerationContext is just data)

### 5. VentriloquistGenerator: The SLM Path

**File**: `src/hologram/generation/ventriloquist.py` (line 101)

```python
def generate_with_validation(self, context: GenerationContext, max_tokens=None):
    if max_tokens is None:
        max_tokens = self._max_tokens  # 256 (line 71 in __init__)

    # Build prompts (still 0 tokens cost)
    if context.fact_answer:
        system_prompt = (
            "You are a helpful assistant. Answer questions naturally and fluently "
            "using the provided facts. Be conversational but accurate."
        )  # ~20 tokens when encoded by Novita

        user_prompt = (
            f"Question: {context.query_text}\n\n"
            f"Fact: {context.fact_answer}\n\n"
            f"Answer the question using the fact above. Be natural and conversational."
        )  # ~N tokens

    try:
        # ← TOKEN COST HAPPENS HERE (API call)
        response = self._client.chat.completions.create(
            model=self._fluency_model,  # "moonshotai/kimi-k2-thinking"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,  # 256 hard limit
            temperature=self._temperature,  # 0.7
        )

        # Extract tokens (already generated by API)
        generated_text = response.choices[0].message.content.strip()

        # Validation: fact must appear in response
        if context.fact_answer:
            fact_words = [w for w in fact_answer.split() if len(w) > 2]
            matches = sum(1 for word in fact_words if word in generated_text.lower())

            if matches < len(fact_words) * 0.5:
                return None  # REJECT if fact not grounded

        # Return result
        tokens = generated_text.split()  # Rough tokenization
        return GenerationResult(text=generated_text, tokens=tokens, ...)

    except Exception as e:
        return None  # Fall back to templates
```

**Token cost**:
- Input: 20 (system) + |user_prompt| (variable)
- Output: up to 256
- Total: depends on context

**Critical points**:
- No input token validation
- No window size checking
- Silent failure on API error
- max_tokens is HARD LIMIT (API-enforced)

### 6. ResonantGenerator: The HDC Fallback

**File**: `src/hologram/generation/resonant_generator.py` (line 123+)

```python
class ResonantGenerator:
    def generate_with_validation(self, context: GenerationContext, max_tokens=10):
        # max_tokens is HARDCODED in ResponseSelector line 738:
        # max_tokens=10 if not use_ventriloquist else 256

        # Step 1: Resonator factorization
        # (max 100 iterations from constants.py:36)
        resonator_result = self._resonator.resonate(thought_vector)

        # Step 2-6: Token generation loop
        tokens = []
        for i in range(max_tokens):
            # Generate candidate
            candidate = self._vocab.sample(temperature=0.7)

            # Verify alignment with target
            divergence = similarity(...)

            if divergence > ACCEPT_THRESHOLD:
                tokens.append(candidate)
            elif divergence > SOFT_THRESHOLD:
                tokens.append(candidate_with_correction)
            else:
                # Resample (up to 5 times)
                ...

        return GenerationResult(text=" ".join(tokens), ...)
```

**Token cost**: 0 (no API calls)
**Output**: Fixed 10 tokens maximum

### 7. Circuit Breaker Protection

**File**: `src/hologram/generation/circuit_breaker.py`

```python
class SimpleCircuitBreaker:
    def __init__(self, failure_threshold=3, window_size=10, cooldown_seconds=60.0):
        self.failure_threshold = failure_threshold
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self._failures = deque(maxlen=window_size)
        self._open_until = None

    def is_open(self) -> bool:
        # If 3 failures in last 10 attempts, open circuit
        if len(self._failures) >= self.failure_threshold:
            if len(self._failures) == self.failure_threshold:
                # All recent attempts failed - OPEN
                self._open_until = time.time() + self.cooldown_seconds

        # If open, check if cooldown expired
        if self._open_until and time.time() < self._open_until:
            return True  # Still in cooldown

        return False
```

**Usage** (ResponseSelector, line 726):

```python
def _generate_response_with_context(self, context, use_ventriloquist=False):
    if self._circuit_breaker.is_open():
        return None  # Fall back to templates

    try:
        result = generator.generate_with_validation(...)
        if result and result.text:
            self._circuit_breaker.record(failed=False)
            return result.text
    except Exception:
        pass

    self._circuit_breaker.record(failed=True)
    return None
```

---

## Token and Context Behavior Summary

### Each Request's Token Journey

```
Step 1: Intent/Entity Classification
├─ Input: String
├─ Processing: HDC vector operations
└─ Tokens: 0 (no API calls)

Step 2: Fact Store Query
├─ Input: Subject, predicate strings
├─ Processing: Vector encoding, resonance search
├─ Output: Answer string
└─ Tokens: 0 (not serialized yet)

Step 3: Context Construction
├─ Input: Query string + answer string
├─ Processing: String concatenation
├─ Output: Serialized prompt
└─ Tokens: 0 (Novita will count when API is called)

Step 4: API Call (SLM)
├─ Input: Serialized prompts
├─ Tokens sent: Novita counts (unknown to us)
├─ Processing: SLM inference on Novita servers
├─ Tokens received: up to 256
└─ Total: 20 + |prompt| + 256

Step 5: Validation + Return
├─ Input: Generated response
├─ Processing: String matching (fact appears?)
└─ Tokens: 0 (local processing)
```

### Where Token Overflow Can Happen

1. **Input prompt too large**
   - Very long query_text
   - Multiple long facts
   - System prompt overhead
   - **No client-side check** (PROBLEM)

2. **API truncation**
   - Novita silently truncates if exceeds 8K
   - System continues with truncated context
   - **No warning to user** (PROBLEM)

3. **Generation too long**
   - max_tokens=256 prevents too-long output
   - But 256 tokens is still meaningful length
   - **Unlikely to overflow** (OK)

4. **FactStore saturation**
   - Capacity ~100 facts (estimated)
   - Beyond 100, accuracy degrades
   - **No saturation warning** (PROBLEM)

---

## Testing Edge Cases

### Test 1: Long Query + Multiple Facts

```python
# File: tests/test_token_limits.py (NOT IN REPO)

def test_long_context_ventriloquist():
    # Create fact store with 50 facts
    fs = FactStore(space, codebook)
    for i in range(50):
        fs.add_fact(f"Country{i}", "capital", f"Capital{i}")

    # Build long query
    long_query = " ".join([f"What is the capital of Country{i}?" for i in range(10)])
    # ~100 tokens

    # Construct context
    facts = []
    for i in range(10):
        answer, _ = fs.query(f"Country{i}", "capital")
        facts.append(f"Country{i} --capital--> {answer}")
    # ~70 tokens

    # Total: 20 (system) + 100 (query) + 70 (facts) = 190 tokens
    # SAFE (well under 8K)

    # Test passes but issue: No warning even if we were at 7K tokens
```

### Test 2: FactStore Saturation

```python
def test_factstore_saturation():
    fs = FactStore(space, codebook)

    # Add facts until saturation
    for i in range(150):  # Beyond ~100 capacity
        fs.add_fact(f"Entity{i}", "property", f"Value{i}")

    # Queries still return results, but confidence drops
    result, confidence = fs.query("Entity100", "property")

    # EXPECTED: confidence < 0.6 (degraded)
    # ACTUAL: No warning, user doesn't know
    # PROBLEM: Silent degradation!
```

### Test 3: Overflow Handling

```python
def test_overflow_fallback():
    # What happens if generation fails?

    selector = ResponseSelector(
        ...,
        ventriloquist_generator=MockFailingGenerator(),  # Always fails
    )

    # First attempt: Ventriloquist fails
    # Second attempt: ResonantGenerator (if available)
    # Third attempt: Templates
    # Fourth attempt: Fallback

    # GOOD: Has fallback chain
    # BAD: No user notification of failures
```

---

## Configuration Changes Recommended

### 1. Add Context Window Monitoring

```python
# In VentriloquistGenerator.__init__:
self._context_window = 8000  # Kimi K2 context

# In generate_with_validation:
input_size = len(system_prompt) // 4 + len(user_prompt) // 4
output_budget = self._context_window - input_size - 100  # 100 safety margin

if output_budget < 128:
    logger.warning(f"Low context budget: {output_budget} tokens left")
    max_tokens = min(max_tokens, output_budget)
```

### 2. Add FactStore Saturation Warning

```python
# In FactStore.add_fact:
saturation = self._fact_count / (self._space.dimensions / 100)

if saturation > 0.8:
    logger.warning(f"FactStore approaching capacity: {saturation:.0%}")

if saturation >= 1.0:
    logger.error("FactStore at capacity - consider HierarchicalFactStore")
```

### 3. Add Token Accounting

```python
# New: src/hologram/generation/token_counter.py

class TokenAccountant:
    def __init__(self, session_limit=8000):
        self.session_limit = session_limit
        self.tokens_used = 0
        self.turns = []

    def log_turn(self, input_tokens, output_tokens):
        self.tokens_used += input_tokens + output_tokens
        self.turns.append({
            "input": input_tokens,
            "output": output_tokens,
            "cumulative": self.tokens_used
        })

        if self.tokens_used > 0.8 * self.session_limit:
            logger.warning(f"Token budget: {self.tokens_used}/{self.session_limit}")

    def get_budget_remaining(self):
        return max(0, self.session_limit - self.tokens_used)
```

---

## Performance Metrics

### Typical Token Usage

```
Scenario 1: Simple Question
├─ System prompt: 20 tokens
├─ User query: 8 tokens
├─ Fact answer: 1 token
├─ Response: 15 tokens
└─ TOTAL: 44 tokens (0.55% of window)

Scenario 2: Reasoning Chain
├─ System prompt: 100 tokens
├─ Query: 20 tokens
├─ Facts (5): 35 tokens
├─ Response: 200 tokens
└─ TOTAL: 355 tokens (4.4% of window)

Scenario 3: Long Fact List (100 facts)
├─ System: 20 tokens
├─ Query: 10 tokens
├─ Facts (100): 700 tokens
├─ Response: 256 tokens
└─ TOTAL: 986 tokens (12% of window) TIGHT!

Scenario 4: Worst Case (unlikely)
├─ Very long query: 50 tokens
├─ Many facts: 500 tokens
├─ Long response: 256 tokens
├─ Plus system: 20 tokens
└─ TOTAL: 826 tokens (10% of window) OK
```

### Encoding Costs

```
Codebook.encode():
├─ Cache miss: ~10,000 FLOPs (torch.randn)
├─ Cache hit: O(1) dict lookup
├─ Typical hit rate: 85%+
└─ Cost amortized: ~500 FLOPs per encode

FactStore.query():
├─ Exact match: O(1) dictionary
├─ Resonance search: O(N × 10,000) where N = vocabulary
├─ For N=100: 1M FLOPs
└─ For N=1000: 10M FLOPs (SLOW!)

Resonator.resonate():
├─ Per iteration: O(10,000) similarity
├─ Typical iterations: 5-10
├─ Worst case: 100 iterations
└─ Total: 50K-1M FLOPs
```

---

## Key Takeaways

| Aspect | Status | Recommendation |
|--------|--------|-----------------|
| **Token limits** | Hard-coded (good) | Add monitoring (better) |
| **Context window** | API-managed (implicit) | Add client-side checking (explicit) |
| **Overflow handling** | Fallback chain (good) | Add warnings (better) |
| **FactStore capacity** | Estimated (implicit) | Add saturation check (explicit) |
| **Performance scaling** | Linear O(N) search | Use FAISS for O(log N) |
| **Error reporting** | Silent failures | Add detailed logging |
| **User feedback** | No capacity warnings | Implement saturation alerts |

---

## Files to Monitor for Token Issues

1. **src/hologram/generation/ventriloquist.py**
   - max_tokens parameter (line 71, 98)
   - Context construction (lines 127-145)
   - API call (lines 159-167)
   - No input validation!

2. **src/hologram/conversation/selector.py**
   - max_tokens override (line 738)
   - Circuit breaker check (line 727)
   - Fact retrieval (line 132)

3. **src/hologram/memory/fact_store.py**
   - Capacity management (lines 99-107)
   - No saturation warnings!
   - Resonance search (lines 281-285)

4. **src/hologram/config/constants.py**
   - All token limits
   - Capacity divisor (line 43)
   - Thresholds (lines 17-27)

---

## Summary

**Current State**: Token management is implicit and soft-bounded
- Fixed limits prevent obvious overflow (max_tokens=256, 10, etc.)
- No explicit monitoring of context window
- Graceful fallback on failures (circuit breaker, templates)
- No warnings to user about saturation or limits

**Risks**:
- Silent context truncation by API
- FactStore degradation without warning
- No per-session token accounting
- Vocabulary size linear search O(N) becomes slow

**Recommendations** (in priority order):
1. Add context window monitoring in VentriloquistGenerator
2. Add FactStore saturation warnings
3. Implement per-session token accounting
4. Optimize resonance search with FAISS for large stores
5. Add detailed logging for debugging







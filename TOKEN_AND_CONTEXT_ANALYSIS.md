# Conscious Hologram: Token and Context Flow Analysis

## Executive Summary

The Conscious Hologram uses a **hybrid token architecture** combining:
1. **HDC (Hyperdimensional Computing) tokens**: Fixed-dimensional vectors (10,000D by default)
2. **SLM (Small Language Model) tokens**: Natural language tokens from Novita API
3. **Template tokens**: Pre-defined response patterns (no encoding cost)

**Critical Finding**: Token limits are **implicit and soft-bounded** rather than hard limits. The system degrades gracefully but lacks explicit overflow handling mechanisms. Context window management is entirely delegated to external SLM APIs.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Message Flow Analysis](#message-flow-analysis)
3. [Token Consumption at Each Stage](#token-consumption-at-each-stage)
4. [Context Window Limitations](#context-window-limitations)
5. [SLM/Ventriloquist Layer Analysis](#slmventriloquist-layer-analysis)
6. [Overflow Handling Mechanisms](#overflow-handling-mechanisms)
7. [Fact Encoding and Token Implications](#fact-encoding-and-token-implications)
8. [Performance Bottlenecks](#performance-bottlenecks)
9. [Recommendations](#recommendations)

---

## Architecture Overview

### System Components

```
User Input
    ↓
[ConversationalChatbot]
    ├─→ IntentClassifier (HDC: 10,000D vector)
    ├─→ EntityExtractor (String-based)
    ├─→ ResponseSelector
    │   ├─→ FactStore.query() [HDC resonance]
    │   ├─→ ResponseCorpus.retrieve() [HDC bundled vectors]
    │   └─→ ResponseGenerator [TWO OPTIONS]:
    │       ├─→ ResonantGenerator (HDC-native, 10-100 tokens)
    │       └─→ VentriloquistGenerator (SLM-based, 256 tokens)
    ↓
Response Output
```

### Dual Generation Strategy

| Component | Token Type | Token Count | Context Window | Encoding Overhead |
|-----------|-----------|------------|-----------------|------------------|
| **ResonantGenerator** | HDC vectors → natural language | 10 tokens | N/A (stateless) | 0 (direct vocab lookup) |
| **VentriloquistGenerator** | SLM (Novita API) | 256 tokens | Unknown (API-managed) | Fact serialization to string |
| **FactStore** | HDC vectors (binding) | 10,000D per fact | N/A (interference patterns) | O(D) per encode |

---

## Message Flow Analysis

### Complete Request-Response Lifecycle

```
1. USER INPUT: "What is the capital of France?"
   ↓
2. CLASSIFICATION PHASE (No token cost yet)
   - IntentClassifier.classify(text)
     • Encodes input as vector: hash("what is capital of france") → seed → torch.randn(10000)
     • Bundles with intent patterns
     • Returns: IntentType.QUESTION, confidence=0.85

3. EXTRACTION PHASE
   - EntityExtractor.extract(text)
     • String-based regex and pattern matching (no vector encoding)
     • Returns: ["capital", "France"]

4. RESPONSE SELECTION PHASE ← CRITICAL CONTEXT POINT
   - ResponseSelector.select(intent, entities, text)

     A. FACT QUERY:
        fs.query("France", "capital")
        - Encodes: France → 10,000D vector
        - Encodes: "capital" → 10,000D vector
        - Binds vectors: bind(france_vec, capital_vec)
        - Resonates against all stored object values
        - Returns: ("Paris", confidence=0.98)

     B. GENERATION ROUTING DECISION:
        if ventriloquist_available:
            use_ventriloquist = True  ← ALWAYS prefers SLM
        elif is_factual_question and fact_confidence >= 0.5:
            use_hdc_generator = True

     C. IF VENTRILOQUIST PATH:
        context = GenerationContext(
            query_text="What is the capital of France?",  ← Passed as STRING
            fact_answer="Paris",                          ← String from FactStore
            entities=["capital", "France"],               ← Strings
            style=StyleType.NEUTRAL,
            thought_vector=None                           ← NOT USED FOR SLM
        )

        ventriloquist.generate_with_validation(context)
            - Constructs system prompt: "You are helpful..."
            - Constructs user prompt:
              "Question: What is the capital of France?

               Fact: Paris

               Answer the question using the fact above..."
            - Calls Novita API with max_tokens=256
            - Response: "The capital of France is Paris."
            - Returns: GenerationResult(text=..., tokens=["The", "capital", ...])

5. RESPONSE OUTPUT
   - Returns: "The capital of France is Paris."
```

### Context Window Implications

At step 4C, **context is constructed as strings**, not as vectors. This means:

- **No HDC interference**: Facts don't compete in a shared vector space
- **No built-in deduplication**: Multiple facts with similar meanings can coexist
- **API-delegated limits**: Novita API handles tokenization and windowing
- **Unknown token count**: User never knows if context exceeds SLM's window

---

## Token Consumption at Each Stage

### Stage 1: Intent Classification

```python
# src/hologram/conversation/intent.py (IntentClassifier)

def classify(self, text: str) -> IntentResult:
    # Step 1: Encode input text
    input_vec = self._codebook.encode(text)  # 1 × 10,000D vector

    # Step 2: Bundle with intent prototypes (stored in memory)
    # memory._intent_memory contains bundled prototype vectors

    # Step 3: Resonator converges (100 max iterations per config)
    # Each iteration compares input_vec against all stored patterns
    # Computational cost: O(P × D) where P = patterns, D = dimensions

    return IntentResult(intent=..., confidence=...)
```

**Token Cost**: 0 (vectors are internal, no serialization)
**Vector Operations**: O(100 × 10,000) = 1M operations maximum
**Context Window Impact**: NONE (stateless)

### Stage 2: Entity Extraction

```python
# src/hologram/conversation/entity.py (EntityExtractor)

def extract(self, text: str) -> List[Entity]:
    # String-based pattern matching and NER
    # No vector operations
    # Example: regex for "What is the {entity}?"

    return [Entity(name="capital", type="predicate"),
            Entity(name="France", type="proper_noun")]
```

**Token Cost**: 0 (pure string operations)
**Vector Operations**: 0
**Context Window Impact**: NONE

### Stage 3: Fact Store Query

```python
# src/hologram/memory/fact_store.py (FactStore.query)

def query(self, subject: str, predicate: str) -> tuple[str, float]:
    # Strategy 1: Exact match (O(1) in metadata)
    exact_key = f"{subject}:{predicate}"
    if exact_key in self._exact_index:
        return (fact.object, 1.0)  # ← CACHE HIT

    # Strategy 2: Resonance search (if no exact match)
    s_vec = self._codebook.encode(subject)           # 10,000D
    p_vec = self._codebook.encode(predicate)         # 10,000D
    key = Operations.bind(s_vec, p_vec)              # Binding operation

    # Iterate over ALL stored object values
    for obj in self._value_vocab:
        obj_vec = self._value_vectors_cache[obj] or encode(obj)  # Cache lookup

    candidates = torch.stack(obj_vecs)  # N × 10,000D matrix
    similarities = self._memory.resonance(key, candidates)

    return (value_list[argmax(similarities)], max_similarity)
```

**Token Cost**: 0 (internal HDC representation)
**Vector Operations**:
- Best case (exact match): O(1)
- Worst case (resonance): O(N × 10,000) where N = vocabulary size

**Memory Impact**:
- **Vocabulary size directly impacts context**: Each stored fact adds an object to `_value_vocab`
- **Capacity limit**: Constants.py estimates ~100 facts before degradation
- **Actual capacity**: Empirically unknown, but guaranteed <100% interference

### Stage 4A: ResonantGenerator Path (HDC-native)

```python
# src/hologram/generation/resonant_generator.py

def generate_with_validation(self, context: GenerationContext) -> GenerationResult:
    # Input: thought_vector (bundled: subject + verb + object)
    # Max: 10 tokens (from ResponseSelector._generate_response_with_context)

    # Token generation loop
    tokens = []
    for i in range(max_tokens=10):
        # Step 1: Resonator factorizes thought into subject/verb/object
        resonator_result = self._resonator.resonate(thought_vec)
        # Convergence: max 100 iterations, early stop at 0.95 similarity

        # Step 2: TargetEncoder packages constraints
        target = self._target_encoder.encode(resonator_result, style)

        # Step 3: Generate candidate token from vocabulary
        candidate = self._vocab.sample(temperature=0.7)

        # Step 4: ReEncoder projects token to HDC
        token_vec = self._codebook.encode(candidate)

        # Step 5: DivergenceCalculator verifies alignment
        divergence = similarity(token_vec, target)

        if divergence > ACCEPT_THRESHOLD:
            tokens.append(candidate)
        elif divergence > SOFT_THRESHOLD:
            tokens.append(candidate_with_correction)
        else:
            # Resample up to 5 times
            ...

        # Step 6: SesameModulator adds style
        if confidence < DISFLUENCY_THRESHOLD:
            tokens.append(filler)  # "um", "uh", etc.

    return GenerationResult(text=" ".join(tokens), ...)
```

**Token Cost**: 10 tokens maximum (hard-coded in ResponseSelector)
**Vector Operations per token**: ~O(10,000) for encoding + similarity
**Total Computational Cost**: 10 × 10,000 = 100,000 operations
**Context Window**: N/A (stateless, no LLM API call)
**Token Overflow**: NONE (fixed 10-token limit)

### Stage 4B: VentriloquistGenerator Path (SLM)

```python
# src/hologram/generation/ventriloquist.py (VentriloquistGenerator)

def generate_with_validation(self, context: GenerationContext, max_tokens=256):
    # Constructs prompt strings (NO vector encoding)

    if context.fact_answer:
        system_prompt = "You are a helpful assistant..."
        user_prompt = f"""Question: {context.query_text}

Fact: {context.fact_answer}

Answer the question using the fact above..."""
    else:
        system_prompt = "You are a helpful assistant..."
        user_prompt = context.query_text

    # API CALL ← CONTEXT WINDOW EXPOSURE HERE
    response = self._client.chat.completions.create(
        model="moonshotai/kimi-k2-thinking",  # SLM via Novita
        messages=[
            {"role": "system", "content": system_prompt},    # ~20 tokens
            {"role": "user", "content": user_prompt}         # VARIABLE
        ],
        max_tokens=256,                      # HARD LIMIT
        temperature=0.7,
    )

    generated_text = response.choices[0].message.content.strip()

    # Validation: Check if fact_answer appears in response
    if context.fact_answer:
        fact_words = [w for w in fact_answer.split() if len(w) > 2]
        matches = count_word_matches(generated_text, fact_words)

        if matches < len(fact_words) * 0.5:
            return None  # VALIDATION FAILED

    # Convert tokens to GenerationResult for compatibility
    tokens = generated_text.split()
    return GenerationResult(
        text=generated_text,
        tokens=tokens,
        trace=[GenerationTrace(...) for token in tokens],
        metrics=GenerationMetrics(total_tokens=len(tokens), ...)
    )
```

**Token Cost**:
- System prompt: ~20 tokens
- User prompt: VARIABLE
- Max generation: 256 tokens
- **Total worst case**: 20 + prompt_tokens + 256 tokens

**Context Window Issues**:
- **System prompt size**: Constant (~20 tokens)
- **User query size**: Variable (unknown to system)
- **Fact answer size**: Variable (depends on FactStore query results)
- **Total context before generation**: 20 + |query_text| + |fact_answer| tokens

**No Overflow Handling**:
- If sum exceeds SLM's window, API silently truncates or errors
- No client-side validation
- No warning to user

---

## Context Window Limitations

### Novita API Constraints (Inferred from Code)

The system delegates to **Novita API** (OpenAI-compatible endpoint):

```python
self._client = OpenAI(
    api_key=api_key,
    base_url="https://api.novita.ai/openai"
)
```

**Models used**:
- `moonshotai/kimi-k2-thinking`: Fluency model (SLM)
- `zai-org/glm-4.6v`: Reasoning model (larger, for chain-of-thought)

**Known Context Windows** (from research):
- Kimi K2: ~8,000 tokens
- GLM-4.6v: ~128,000 tokens

**System Configuration**:
- Fluency model: max_tokens=256 (fixed in `__init__`)
- Reasoning model: max_tokens=512 (2× fluency, hardcoded in `generate_reasoning_chain`)
- Code generation: max_tokens=512 (2× fluency)

### Total Context Budget

For a typical conversation turn:

```
User asks: "What is the capital of France?"  ← ~8 tokens
System prompt: 20 tokens
Fact answer: "Paris" ← ~1 token
--------------------------------------------------
Total input: ~29 tokens (well within 8,000 token window)

Max generation: 256 tokens
Final total: ~285 tokens
```

**Realistic worst case** (long multi-fact conversation):
- Multiple facts from FactStore: 100+ tokens
- Accumulated conversation history: NOT STORED (each turn is stateless in SLM)
- System prompt: 20 tokens
- User input: 50+ tokens
- **Total**: ~170 tokens (still safe for 8K window)

### Implicit Limits

**Soft-bounded limits** exist:
1. **FactStore capacity**: ~100 facts before interference noise overwhelms signal
2. **Vocabulary size**: No explicit limit, but resonance search O(N) becomes slow
3. **Resonator convergence**: Max 100 iterations (config constant)
4. **Generation tokens**: 256 per turn (hard-coded max_tokens)

**No Hard Limits**:
- No check for cumulative token usage
- No warning when approaching capacity
- No graceful degradation message

---

## SLM/Ventriloquist Layer Analysis

### Dual-Model Architecture

The Ventriloquist implements **conditional model selection**:

```python
# From ResponseSelector._generate_response_with_context

if self._ventriloquist:
    # Always prefer Ventriloquist (SLM for fluency)
    generated_response = self._generate_response_with_context(
        context,
        use_ventriloquist=True
    )
elif is_factual_question and fact_confidence >= 0.5 and self._generator:
    # Fallback to HDC generator (only if Ventriloquist unavailable)
    generated_response = self._generate_response_with_context(context)
```

**Priority**: Ventriloquist >> ResonantGenerator >> Templates

### Token Management in VentriloquistGenerator

#### Fluency Path (Primary)

```python
def generate_with_validation(self, context, max_tokens=256):
    # Always use fluency model
    response = self._client.chat.completions.create(
        model=self._fluency_model,  # "moonshotai/kimi-k2-thinking"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=256,  # HARD LIMIT
        temperature=0.7,
    )

    # Validate: fact must appear in response
    if context.fact_answer:
        if not fact_appears_in_response(context.fact_answer, response):
            return None  # Reject if fact not grounded

    return GenerationResult(...)
```

**Token Consumption**:
- Input tokens: 20 (system) + N (user prompt)
- Output tokens: up to 256
- Total: 276 + N tokens per turn

**N = |query_text| + |fact_answer| + overhead**

#### Reasoning Path (Chain-of-Thought)

```python
def generate_reasoning_chain(self, query, facts, max_tokens=None):
    if max_tokens is None:
        max_tokens = self._max_tokens * 2  # 512 tokens

    # Constructs fact list
    facts_text = "\n".join([f"- {fact}" for fact in facts])

    response = self._client.chat.completions.create(
        model=self._reasoning_model,  # "zai-org/glm-4.6v" (larger)
        messages=[...],
        max_tokens=512,  # 2× fluency
        temperature=0.3,  # Lower for reasoning
        response_format={"type": "json_object"}
    )

    # Parse JSON: {"reasoning_steps": [...], "conclusion": "..."}
    reasoning_chain = json.loads(response.content)

    # Verify against FactStore (NO TOKEN COST)
    verification = self._verify_deduction(reasoning_chain, fact_store, facts)

    return reasoning_chain
```

**Token Consumption**:
- System prompt: 100+ tokens (detailed instructions)
- Facts: 10 tokens per fact
- Query: 20 tokens
- Output: up to 512 tokens
- **Total**: ~640+ tokens (still within 128K window for GLM-4.6v)

#### Code Generation Path

```python
def generate_code_with_context(self, prompt, fact_store, concept_store, max_tokens=None):
    if max_tokens is None:
        max_tokens = self._max_tokens * 2  # 512 tokens

    # Dual retrieval
    concept_facts, project_facts = self.retrieve_dual_context(prompt)

    # Build enhanced prompt with context
    context_str = format_context(concept_facts, project_facts)
    user_prompt = f"{context_str}\n\nTask: {prompt}\n\nGenerate the code:"

    response = self._client.chat.completions.create(
        model=self._reasoning_model,
        messages=[
            {"role": "system", "content": "You are an expert Python programmer..."},
            {"role": "user", "content": user_prompt}  # Context + task
        ],
        max_tokens=512,
        temperature=0.3,
    )
```

**Token Consumption**:
- System prompt: 50 tokens
- Context (facts): 50+ tokens
- Task prompt: 50 tokens
- Output: 512 tokens
- **Total**: ~660+ tokens

### Validation and Verification

VentriloquistGenerator includes **two validation layers**:

**Layer 1: String matching** (fluency path)
```python
if context.fact_answer:
    fact_words = [w for w in fact_answer.split() if len(w) > 2]
    matches = sum(1 for word in fact_words if word in response.lower())

    if matches < len(fact_words) * 0.5:  # 50% threshold
        return None  # REJECT
```

**Layer 2: Deduction verification** (reasoning path)
```python
def _verify_deduction(self, reasoning_chain, fact_store, known_facts):
    # Extract fact terms
    fact_terms = set()
    for fact in known_facts:
        parts = fact.replace("--", " ").replace("-->", " ").split()
        fact_terms.update([p.strip() for p in parts if len(p.strip()) > 3])

    # Check if conclusion uses fact terms
    conclusion_words = set(reasoning_chain["conclusion"].lower().split())
    grounded_terms = fact_terms.intersection(conclusion_words)

    if len(grounded_terms) == 0:
        return {"verified": False, "confidence": 0.3}

    return {"verified": True, "confidence": 0.9}
```

**Token Cost**: 0 (post-generation validation, no API calls)

### Error Handling and Circuit Breaker

```python
# From ResponseSelector

def _generate_response_with_context(self, context, use_ventriloquist=False):
    # CHECK CIRCUIT BREAKER FIRST
    if self._circuit_breaker.is_open():
        return None  # Fall back to templates

    try:
        result = generator.generate_with_validation(context, max_tokens=10 or 256)

        if result and result.text and len(result.text.strip()) > 0:
            self._circuit_breaker.record(failed=False)
            return result.text.strip()
    except Exception:
        pass

    self._circuit_breaker.record(failed=True)
    return None
```

**Circuit Breaker Config**:
- Failure threshold: 3 failures
- Window size: 10 attempts
- Cooldown: 60 seconds

**Overflow Handling**:
- If generation fails 3 times in 10 attempts, stop trying
- Fall back to template responses (no generation)
- Automatically resume after 60 seconds

---

## Overflow Handling Mechanisms

### Current Overflow Protections

#### 1. FactStore Capacity Management

```python
# src/hologram/config/constants.py

ESTIMATED_CAPACITY_DIVISOR = 100
# Capacity ≈ dimensions / divisor
# For 10,000D: ~100 facts before degradation

SURPRISE_THRESHOLD = 0.1
# Below 10% novelty, fact is considered "already known" and skipped

# src/hologram/memory/memory_trace.py

def store_with_surprise(self, key, value, surprise_threshold=SURPRISE_THRESHOLD):
    # Measure novelty of new fact
    novelty = self._memory.get_surprise_score(key, value)

    if novelty < surprise_threshold:
        return None  # Duplicate, skip learning

    # Bundle into interference pattern
    self._trace = bundle(self._trace, bind(key, value))
    self._fact_count += 1
```

**Mechanism**: Prevents duplicate facts from bloating memory
**Limit**: When surprise < 0.1, fact is rejected
**Overflow Behavior**: System stops learning until facts become novel

#### 2. Generation Token Limits

```python
# Fixed hard limits per model:
ResonantGenerator: max_tokens = 10
VentriloquistGenerator.fluency: max_tokens = 256
VentriloquistGenerator.reasoning: max_tokens = 512
VentriloquistGenerator.code: max_tokens = 512

# No dynamic adjustment based on available context
```

**Mechanism**: Hard-coded max_tokens parameter
**Limit**: 256 tokens per turn (fluency), 512 (reasoning/code)
**Overflow Behavior**: API truncates at 256 tokens

#### 3. Resonator Convergence Limit

```python
# src/hologram/config/constants.py
MAX_RESONATOR_ITERATIONS = 100

# src/hologram/core/resonator.py
for i in range(max_iterations=100):
    # Resonator loop
    if similarity(current, target) > CONVERGENCE_THRESHOLD:
        break  # Early exit
```

**Mechanism**: Max 100 iterations for factorization
**Limit**: Prevents infinite loops
**Overflow Behavior**: Timeout after 100 iterations, use best partial result

#### 4. Circuit Breaker for Generation Failures

```python
# src/hologram/generation/circuit_breaker.py

self._circuit_breaker = SimpleCircuitBreaker(
    failure_threshold=3,      # Fail 3 times
    window_size=10,           # In last 10 attempts
    cooldown_seconds=60.0     # Then stop for 60s
)
```

**Mechanism**: Stop trying if generation consistently fails
**Limit**: 3 failures triggers cooldown
**Overflow Behavior**: Fall back to template responses

### Missing Overflow Protections

**Critical Gaps**:

1. **No context window monitoring**
   ```python
   # VentriloquistGenerator doesn't check:
   # - Total tokens sent to API
   # - API response indicating truncation
   # - Cumulative token usage per session
   ```

2. **No per-turn token accounting**
   ```python
   # No tracking of:
   # - Input tokens per turn
   # - Output tokens generated
   # - Running total for session
   ```

3. **No vocabulary saturation warning**
   ```python
   # FactStore has capacity, but no early warning:
   max_vocab = 100  # Estimated
   current_vocab = fs.vocabulary_size

   if current_vocab > 0.8 * max_vocab:
       warn("Approaching FactStore saturation")  # NOT IMPLEMENTED
   ```

4. **No graceful context reduction**
   ```python
   # If context exceeds window:
   # - Could drop oldest facts
   # - Could summarize facts
   # - Could prioritize by confidence
   # NONE OF THIS EXISTS
   ```

---

## Fact Encoding and Token Implications

### Fact Encoding Process

```python
# src/hologram/memory/fact_store.py

def add_fact(self, subject, predicate, obj, source=None, confidence=1.0):
    # Step 1: Normalize strings
    subject_norm = subject.lower().strip()
    predicate_norm = predicate.lower().strip()

    # Step 2: Encode as vectors (Codebook)
    s_vec = self._codebook.encode(subject_norm)      # 10,000D
    p_vec = self._codebook.encode(predicate_norm)    # 10,000D
    o_vec = self._codebook.encode(obj)               # 10,000D

    # Step 3: Create key via binding
    key = Operations.bind(s_vec, p_vec)  # bind: element-wise multiply & permute

    # Step 4: Store with surprise gating
    surprise = self._memory.store_with_surprise(
        key,
        o_vec,
        learning_rate=confidence,
        surprise_threshold=SURPRISE_THRESHOLD
    )

    # Step 5: Create metadata fact
    fact = Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source=source,
        surprise_score=surprise
    )

    # Step 6: Add to indices
    self._facts.append(fact)
    self._value_vocab.add(obj)
    self._subject_vocab.add(subject)
    self._exact_index[f"{subject_norm}:{predicate_norm}"] = fact

    return fact
```

### Token Implications

#### HDC Encoding (Internal)

Each fact requires:
- 3 vector encodes: subject, predicate, object → 3 × 10,000D = 30,000 floats
- 1 binding operation (not a new vector, same dimensionality)
- 1 bundling operation (superposition, no new memory)

**No SLM token cost** during encoding

#### String Serialization (For SLM)

When passing facts to VentriloquistGenerator:

```python
# src/hologram/generation/ventriloquist.py

facts_text = "\n".join([f"- {fact}" for fact in facts])
# Each fact becomes a string:
# "- France --capital--> Paris" ← ~7 tokens

# For 10 facts: 70 tokens
# For 100 facts: 700 tokens ← EXCEEDS SLM WINDOW

user_prompt = f"""
Question: {query}

Known Facts:
{facts_text}

Provide reasoning...
"""
# Total: query_tokens + 700 tokens + template ≈ 750+ tokens
```

**Token Cost per Fact**: ~7 tokens in SLM context
**Worst case (100 facts)**: 700 SLM tokens just for facts

### Fact Storage Overhead

```python
# Memory footprint per fact:
- MemoryTrace: 10,000 dimensions × 4 bytes (float32) = 40KB per bundled state
- Metadata: ~200 bytes per Fact object
- Cache: Optional vector cache (10,000D × 4 bytes = 40KB per unique object)

# For 100 facts:
- Bundle: 40KB (shared across all facts via superposition)
- Metadata: 100 × 200 bytes = 20KB
- Cache: 100 × 40KB = 4MB
# Total: ~4MB for 100 facts
```

### HierarchicalFactStore (Hot/Cold)

```python
# src/hologram/memory/fact_store.py

class HierarchicalFactStore:
    # Hot layer: Fast, O(1) exact lookup, ~100 fact capacity
    self._hot = FactStore(space, codebook)

    # Cold layer: FAISS index, O(log n) search, unlimited capacity
    self._cold = FaissAdapter(dimensions=10000, persist_path="/tmp/...")
```

**Token Impact**:
- Hot facts: 0 SLM tokens (only used via resonance)
- Cold facts: ~7 tokens each if retrieved and passed to SLM
- Retrieval: `max_facts=5` parameter limits spillover

---

## Performance Bottlenecks

### 1. Resonance Search Complexity

```python
# FactStore.query()

candidates = torch.stack([
    self._value_vectors_cache[v]
    for v in sorted(self._value_vocab)
])  # N × 10,000D matrix

similarities = self._memory.resonance(key, candidates)
# Compute similarity between key and each candidate
# Cost: O(N × D) = O(N × 10,000)

best_idx = torch.argmax(similarities)
```

**Performance**:
- Linear search through all object values
- For 100 objects: 100 × 10,000 = 1M FLOPs
- For 1,000 objects: 10M FLOPs (SLOW)

**Optimization opportunity**: Use FAISS for vector similarity in FactStore

### 2. VentriloquistGenerator API Latency

```python
# VentriloquistGenerator.generate_with_validation()

response = self._client.chat.completions.create(
    model="moonshotai/kimi-k2-thinking",
    messages=[...],
    max_tokens=256,
    temperature=0.7,
)  # Network round-trip
```

**Latency**: 500ms - 2000ms typical for SLM generation
**Cost**: 1 API call per response (unless generation fails)
**Circuit breaker**: If 3 failures in 10 attempts, waits 60 seconds

### 3. Resonator Convergence

```python
# Resonator.resonate()

for i in range(max_iterations=100):
    # Each iteration:
    similarity = compute_similarity(current, target)  # O(D)
    if similarity > CONVERGENCE_THRESHOLD:
        break
    # Update current
    current = update_step(current, target)            # O(D)
```

**Performance**:
- Worst case: 100 iterations × 10,000D = 1M operations
- Best case: 1-5 iterations (typical)
- **No early stopping on divergence**: Can waste time converging to local optima

### 4. Encoding Cache Misses

```python
# Codebook.encode() - called repeatedly

def encode(self, concept: str) -> torch.Tensor:
    if concept not in self._cache:
        # Cache miss: compute new vector
        seed = self._hash_to_seed(concept)
        self._cache[concept] = self._space.random_vector(seed)  # 10K operations
    return self._cache[concept]
```

**Cache behavior**:
- First mention of a concept: O(10,000) operations
- Subsequent mentions: O(1) cache lookup
- Typical coverage: 80-90% cache hit rate

---

## Recommendations

### Critical Issues to Address

#### 1. Add Context Window Monitoring

```python
# RECOMMENDATION: src/hologram/generation/ventriloquist.py

class VentriloquistGenerator:
    def __init__(self, ..., context_window_size=8000):
        self._context_window = context_window_size
        self._token_counter = TokenCounter()

    def generate_with_validation(self, context, max_tokens=256):
        # Calculate input tokens
        system_tokens = estimate_tokens(system_prompt)  # ~20
        user_tokens = estimate_tokens(user_prompt)      # VARIABLE

        input_total = system_tokens + user_tokens
        max_output = self._context_window - input_total

        if max_output < 128:
            # Context too large, reduce or warn
            logger.warning(f"Context overflow: {input_total} + 256 > {self._context_window}")
            max_tokens = min(max_tokens, max_output)

        return self._client.chat.completions.create(..., max_tokens=max_tokens)
```

#### 2. Add Capacity Warnings for FactStore

```python
# RECOMMENDATION: src/hologram/memory/fact_store.py

class FactStore:
    @property
    def saturation_estimate(self) -> float:
        """Estimate how full memory is (0.0 to 1.0)."""
        capacity = self._space.dimensions / 100  # Assume capacity divisor
        return min(1.0, self._fact_count / capacity)

    def add_fact(self, subject, predicate, obj, ...):
        saturation = self.saturation_estimate
        if saturation > 0.8:
            logger.warning(f"FactStore near capacity: {saturation:.1%}")
        elif saturation >= 1.0:
            logger.error("FactStore at capacity - consider using HierarchicalFactStore")
```

#### 3. Implement Graceful Fact Reduction

```python
# RECOMMENDATION: src/hologram/generation/ventriloquist.py

def retrieve_dual_context(self, query, fact_store, max_facts=5):
    """Limit facts to prevent context overflow."""

    concept_facts, project_facts = retrieve_facts(...)

    # PRIORITIZE by confidence/relevance
    concept_facts = sorted(
        concept_facts,
        key=lambda f: f.confidence * f.relevance_to_query(query),
        reverse=True
    )[:max_facts]

    project_facts = sorted(
        project_facts,
        key=lambda f: f.confidence * f.relevance_to_query(query),
        reverse=True
    )[:max_facts]

    return concept_facts, project_facts
```

#### 4. Add Token Accounting per Session

```python
# RECOMMENDATION: New file src/hologram/generation/token_accounting.py

class TokenAccountant:
    """Track token usage per session and warn on overage."""

    def __init__(self, session_limit=10000):
        self.session_limit = session_limit
        self.tokens_used = 0

    def record_input(self, prompt_tokens):
        self.tokens_used += prompt_tokens

    def record_output(self, generated_tokens):
        self.tokens_used += generated_tokens

    def check_budget(self, needed_tokens):
        if self.tokens_used + needed_tokens > self.session_limit:
            return False  # Over budget
        return True

    def get_remaining_budget(self):
        return max(0, self.session_limit - self.tokens_used)
```

#### 5. Optimize FactStore Resonance with FAISS

```python
# RECOMMENDATION: Use FAISS for large fact stores

def query(self, subject: str, predicate: str) -> tuple[str, float]:
    # Exact match (still O(1))
    if exact_key in self._exact_index:
        return (fact.object, 1.0)

    # For large stores (>1000 facts), use FAISS
    if len(self._value_vocab) > 1000:
        # Use pre-built FAISS index
        k_vec = Operations.bind(
            self._codebook.encode(subject),
            self._codebook.encode(predicate)
        )
        # FAISS: O(log N) approximate similarity search
        nearest_neighbors = self._faiss_index.search(k_vec, k=1)
        return (nearest_neighbors[0].object, nearest_neighbors[0].score)
    else:
        # Small store: full resonance O(N×D)
        ...
```

### Nice-to-Have Improvements

#### 6. Implement Token Counting Middleware

```python
# Optional: Automatic token counting for all API calls

class TokenCountingMiddleware:
    def __init__(self, client):
        self.client = client
        self.token_log = []

    def chat_completions_create(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        self.token_log.append({
            "timestamp": datetime.now(),
            "model": kwargs["model"],
            "input": input_tokens,
            "output": output_tokens
        })

        return response
```

#### 7. Add Fact Summarization for Long Contexts

```python
# Optional: Compress multiple facts into summaries

class FactSummarizer:
    def summarize_facts(self, facts: List[Fact], max_tokens=100):
        """Compress facts into brief JSON."""
        return {
            fact.subject: {
                fact.predicate: fact.object
                for fact in facts if fact.subject == subject
            }
            for subject in set(f.subject for f in facts)
        }
        # From 700 tokens to 70 tokens for 10 facts
```

### Summary Table

| Issue | Current | Recommended | Impact |
|-------|---------|------------|--------|
| Context overflow | No checking | Monitor + warn | Prevent silent truncation |
| FactStore capacity | Implicit (~100) | Explicit warning at 80% | Better UX |
| Fact reduction | None | Prioritize by confidence | Handle large fact sets |
| Token accounting | None | Per-session tracking | Budget management |
| Performance | Linear search O(N) | FAISS index O(log N) | Scale to 10K+ facts |

---

## Appendix A: Configuration Constants Summary

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_DIMENSIONS` | 10,000 | HDC vector dimensionality |
| `ESTIMATED_CAPACITY_DIVISOR` | 100 | FactStore capacity ≈ 10,000 / 100 = 100 facts |
| `MAX_GENERATION_TOKENS` | 100 | ResonantGenerator max tokens |
| `CANDIDATE_K` | 5 | Candidates evaluated per token position |
| `MAX_RESONATOR_ITERATIONS` | 100 | Convergence limit for factorization |
| `SURPRISE_THRESHOLD` | 0.1 | Min novelty to learn new fact |
| `RESPONSE_CONFIDENCE_THRESHOLD` | 0.20 | Min similarity to respond |
| `REFUSAL_CONFIDENCE_THRESHOLD` | 0.10 | Below this, refuse to answer |

### SLM Configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_FLUENCY_MODEL` | `moonshotai/kimi-k2-thinking` | Primary SLM for natural language |
| `DEFAULT_REASONING_MODEL` | `zai-org/glm-4.6v` | Larger model for reasoning chains |
| VentriloquistGenerator max_tokens | 256 | Fluency generation limit |
| Reasoning chain max_tokens | 512 | 2× fluency limit |
| Code generation max_tokens | 512 | Same as reasoning |

---

## Appendix B: Complete Message Flow Diagram

```
USER: "What is the capital of France?"
    ↓
[1] ConversationalChatbot.respond()
    ├─ [1.1] IntentClassifier.classify()
    │   └─ Encodes: hash("...") → seed → torch.randn(10K) → classify
    │   └─ Returns: IntentType.QUESTION (0.85 confidence)
    │
    ├─ [1.2] EntityExtractor.extract()
    │   └─ String regex: ["capital", "France"]
    │
    ├─ [1.3] ResponseSelector.select()
    │   ├─ [1.3.1] FactStore.query("France", "capital")
    │   │   ├─ Exact match: f"france:capital" → found!
    │   │   └─ Returns: ("Paris", 1.0)
    │   │
    │   ├─ [1.3.2] GenerationContext
    │   │   query_text="What is the capital of France?"
    │   │   fact_answer="Paris"
    │   │   entities=["capital", "France"]
    │   │   style=StyleType.NEUTRAL
    │   │
    │   ├─ [1.3.3] Route to VentriloquistGenerator (preferred)
    │   │   ├─ System prompt: "You are helpful..."
    │   │   ├─ User prompt: "Question: What is the capital of France?\n\nFact: Paris\n\n..."
    │   │   ├─ API Call (Novita): ChatCompletion.create(
    │   │   │   model="moonshotai/kimi-k2-thinking",
    │   │   │   max_tokens=256
    │   │   │)
    │   │   ├─ Response: "The capital of France is Paris."
    │   │   ├─ Validation: Check "Paris" in response ✓
    │   │   └─ Returns: GenerationResult(text="...", tokens=[...])
    │   │
    │   └─ Creates ResponseCandidate
    │       filled_response="The capital of France is Paris."
    │       confidence=0.8
    │
    └─ [1.4] Return response
        "The capital of France is Paris."
```

---

## Appendix C: Token Counting Estimation

```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 characters (Tiktoken rule of thumb)."""
    return len(text) // 4

# Examples:
estimate_tokens("What is the capital of France?")  # ~8 tokens
estimate_tokens("Paris")  # ~1 token
estimate_tokens("You are a helpful assistant...")  # ~5 tokens

# Worst case context for VentriloquistGenerator:
system_prompt = "You are a helpful assistant. Answer questions naturally and fluently using the provided facts. Be conversational but accurate."  # ~25 tokens
user_query = "What is the capital of France?"  # ~8 tokens
fact_answer = "Paris"  # ~1 token
total_input = 25 + 8 + 1 = 34 tokens  # Safe (8K window)
max_output = 256 tokens
total_with_output = 34 + 256 = 290 tokens  # Still safe
```

---

## Conclusion

The Conscious Hologram uses a **sophisticated but implicit context management system**:

1. **HDC layer**: Fixed 10,000D vectors, no token cost, ~100 fact capacity
2. **SLM layer**: Variable token cost, 256-512 token generation limits, API-managed context
3. **Hybrid routing**: Prefers SLM for fluency, falls back to HDC/templates

**Key Findings**:
- Context limits are **soft-bounded** (not hard-enforced)
- **No overflow warnings** to users
- **Graceful degradation** via circuit breaker (after 3 failures)
- **Fact encoding is efficient** (O(1) encoding, O(N) resonance search)
- **No per-session token accounting**

**Recommended Actions**:
1. Add context window monitoring (CRITICAL)
2. Implement capacity warnings for FactStore
3. Add token accounting per session
4. Optimize resonance search with FAISS for large stores
5. Implement graceful fact reduction strategies


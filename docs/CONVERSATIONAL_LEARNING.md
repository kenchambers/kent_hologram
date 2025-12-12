# Conversational Learning Architecture

Technical documentation for the HDC-based conversational learning system in Hologram.

## Overview

The conversational chatbot learns from interactions using **Hyperdimensional Computing (HDC)** rather than traditional ML or external LLMs. Learning occurs through algebraic operations on high-dimensional vectors:

- **Bundling**: Superposition of vectors (element-wise addition + normalization)
- **Binding**: Association of vectors (element-wise multiplication)
- **Similarity**: Cosine similarity for retrieval

No gradient descent. No backpropagation. No external API calls. Pure vector algebra.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IntentClassifier                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Intent Prototypes (learned from examples)               │    │
│  │  greeting_vec = bundle(hello_vec, hi_vec, hey_vec, ...)  │    │
│  │  question_vec = bundle(what_is_vec, who_is_vec, ...)     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  Classification: argmax(cosine(input_vec, prototype_vec))       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ EntityExtractor │ │  StyleTracker   │ │ConversationMem  │
│                 │ │                 │ │                 │
│ Resonance match │ │ Bundle user     │ │ Holographic     │
│ against vocab   │ │ messages into   │ │ trace + Episode │
│                 │ │ style trace     │ │ Retrieval       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ResponseSelector                              │
│  1. Match patterns via resonance                                 │
│  2. Query FactStore for knowledge                                │
│  3. Retrieve Episodic Memories                                   │
│  4. Return ResponseCandidate                                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Implicit Learning                             │
│  If conversation flows naturally → strengthen pattern            │
│  If user repeats/confused → weaken pattern                       │
│  (Hebbian: "neurons that fire together wire together")           │
└─────────────────────────────────────────────────────────────────┘
```

## Learning Mechanisms

### 1. Intent Classification (Example-Based Learning)

Intent classification does **not** use hardcoded keyword lists. Instead, it learns from example phrases.

#### How Prototypes Are Built

Each intent has a **prototype vector** built by bundling example phrases:

```python
# Pseudocode for prototype construction
greeting_prototype = encode("hello")
greeting_prototype = bundle(greeting_prototype, encode("hi"))
greeting_prototype = bundle(greeting_prototype, encode("hey"))
greeting_prototype = bundle(greeting_prototype, encode("good morning"))
# ... more examples
```

The bundle operation creates a **superposition** - a single vector that is similar to all its components.

#### Mathematical Foundation

Given vectors v₁, v₂, ..., vₙ for n example phrases:

```
prototype = normalize(v₁ + v₂ + ... + vₙ)
```

Due to the properties of high-dimensional spaces:

- `cosine(prototype, vᵢ) ≈ 1/√n` for any component vᵢ
- `cosine(prototype, unrelated) ≈ 0`

This means the prototype "remembers" all examples while remaining dissimilar to unrelated inputs.

#### Classification Algorithm

```python
def classify(text: str) -> IntentType:
    input_vec = encode(text)

    scores = {}
    for intent, prototype in intent_prototypes.items():
        scores[intent] = cosine(input_vec, prototype)

    best = argmax(scores)
    if scores[best] < threshold:
        return UNKNOWN
    return best
```

#### Learning New Patterns

```python
def learn(text: str, intent: IntentType):
    example_vec = encode(text)
    intent_prototypes[intent] = bundle(
        intent_prototypes[intent],
        example_vec
    )
```

Each new example strengthens the prototype's response to similar inputs.

**Key insight**: Unlike keyword matching, this generalizes. Teaching "yo what's up" as a greeting makes the system more likely to recognize "hey what's up" even without explicit training.

### 2. Response Pattern Learning (Hebbian)

Response patterns have a `strength` attribute that modulates their selection probability.

#### Pattern Structure

```python
@dataclass
class ResponsePattern:
    pattern_id: str
    intent: IntentType
    entity_pattern: List[str]
    response_template: str      # "The capital of {entity} is {answer}."
    style: StyleType
    strength: float = 1.0       # Learning weight
    pattern_vector: Tensor      # HDC representation
```

#### Pattern Matching

Patterns are matched via cosine similarity against a query vector constructed from:

- Intent vector
- Entity vectors
- Context vector (from conversation memory)

```python
def match(intent, entities, context_vec) -> List[Tuple[Pattern, float]]:
    # Build query vector
    query = bind(intent_vec, entity_vec)
    if context_vec is not None:
        query = bundle(query, context_vec)

    # Score all patterns
    scores = []
    for pattern in patterns:
        sim = cosine(query, pattern.pattern_vector)
        weighted_score = sim * pattern.strength  # Strength modulates score
        scores.append((pattern, weighted_score))

    return sorted(scores, key=lambda x: -x[1])
```

#### Hebbian Learning

After each response, the system observes the user's next input:

```python
def implicit_learning(current_intent, last_pattern):
    if user_seems_confused(current_intent):
        # Weaken the pattern that caused confusion
        last_pattern.strength *= 0.8
    else:
        # Conversation flowing - strengthen the pattern
        last_pattern.strength *= 1.5
```

**Confusion signals**:

- User intent is UNKNOWN
- User repeats same intent with low confidence
- User says "what?" or "huh?"

**Flow signals**:

- Different intent than previous turn
- Higher confidence classification

Over time, effective patterns become stronger and are selected more often.

#### Vector-Based Strengthening

For deeper learning, patterns can also be bundled with successful contexts:

```python
def strengthen_pattern(pattern_id: str):
    pattern = get_pattern(pattern_id)
    # Re-bundle with itself = amplify signal
    pattern.pattern_vector = bundle(
        pattern.pattern_vector,
        pattern.pattern_vector
    )
    pattern.strength *= STRENGTHEN_FACTOR
```

### 3. Style Tracking

The system adapts to user communication style over time.

#### Style Trace Construction

Every user message is encoded and bundled into a cumulative style trace:

```python
class UserStyleTracker:
    def observe(self, text: str):
        message_vec = encode(text)
        if self.style_trace is None:
            self.style_trace = message_vec
        else:
            self.style_trace = bundle(self.style_trace, message_vec)
        self.message_count += 1
```

#### Style Inference

After sufficient messages (default: 3), compare the style trace against style prototypes:

```python
def get_inferred_style(self) -> StyleType:
    if self.message_count < MIN_MESSAGES:
        return NEUTRAL

    best_style = NEUTRAL
    best_sim = -1

    for style in [FORMAL, CASUAL, URGENT]:
        style_vec = get_style_prototype(style)
        sim = cosine(self.style_trace, style_vec)
        if sim > best_sim:
            best_sim = sim
            best_style = style

    return best_style if best_sim > 0.1 else NEUTRAL
```

Style prototypes are built from characteristic words:

- FORMAL: "therefore", "subsequently", "moreover"
- CASUAL: "cool", "yeah", "like", "awesome"
- URGENT: "now", "immediately", "urgent", "asap"

#### Style Application

The inferred style influences pattern selection, preferring patterns tagged with matching style.

### 4. Conversation Memory (Dual-Layer)

The memory system now combines holographic short-term context with episodic retrieval for long-term context.

#### A. Holographic Short-Term Context

Recent turns are encoded as holographic traces to maintain immediate context.

**Turn Encoding**:

```python
turn_vector = bind(
    user_input_vector,
    response_vector,
    position_vector  # Temporal marker
)
```

**Context Vector**:

```python
def get_context_vector(lookback: int = 3) -> Tensor:
    recent = self.turns[-lookback:]
    if not recent:
        return empty_vector()

    context = recent[0].turn_vector
    for turn in recent[1:]:
        context = bundle(context, turn.turn_vector)
    return context
```

#### B. Episodic Long-Term Context (FAISS/Chroma)

Every turn is also stored as an **episode** in a vector database.

**Episode Storage**:

```python
episode_vec = bind(user_vec, response_vec)
episodic_store.add(episode_vec, metadata={
    "user": "What about Germany?",
    "response": "Berlin is the capital.",
    "turn": 42
})
```

**Episode Retrieval**:
When the user speaks, we retrieve the top-k most similar past episodes:

```python
episodes = episodic_store.query(current_query_vec, k=3)
# These episodes are injected into the generation context
```

This allows the system to recall conversations from hundreds of turns ago without keeping them all in the active prompt window.

### 5. Teaching Detection (TEACHING Intent)

The system detects when users are teaching facts via HDC intent classification.

#### Intent Prototype

The TEACHING intent has a prototype built from example phrases:

```python
SEED_EXAMPLES[IntentType.TEACHING] = [
    "the capital of france is paris",
    "the capital of germany is berlin",
    "paris is the capital of france",
    "france's capital is paris",
    "python was created by guido",
    "the sky is blue",
    "dogs are mammals",
    # ... more examples
]
```

#### Fact Extraction

When TEACHING intent is detected, facts are extracted using token analysis (not regex):

```python
def _extract_fact_structure(self, text: str) -> Optional[tuple]:
    tokens = tokenize(text)

    # Find relation word (is, are, was, were, has, have)
    relation_idx = find_relation_word(tokens)

    if relation_idx is None:
        return None

    # Pattern: "the Y of X is Z" or "X is Z"
    subject = extract_subject(tokens, relation_idx)
    predicate = extract_predicate(tokens, relation_idx)
    object = extract_object(tokens, relation_idx)

    return (subject, predicate, object)
```

#### Why No Regex?

Using regex for fact extraction would contradict the HDC learning philosophy:

- **Regex**: Hardcoded patterns that don't generalize
- **HDC**: Learned patterns that generalize to similar inputs

The TEACHING intent prototype naturally generalizes. Teaching "the capital of France is Paris" helps recognize "the population of Japan is 126 million" without explicit patterns.

### 6. Entity Extraction

Entities are extracted via vocabulary resonance.

#### Vocabulary Sources

1. **FactStore vocabulary**: Subjects/objects from stored facts
2. **Codebook cache**: Previously encoded tokens
3. **Learned entities**: Explicitly added during conversation

#### Extraction Algorithm

```python
def extract(text: str) -> List[Entity]:
    tokens = tokenize(text)
    entities = []

    for token in tokens:
        token_vec = encode(token)

        # Check against vocabulary
        for known_entity in vocabulary:
            known_vec = encode(known_entity)
            sim = cosine(token_vec, known_vec)

            if sim > ENTITY_THRESHOLD:
                entities.append(Entity(
                    surface_form=token,
                    canonical_form=known_entity,
                    confidence=sim
                ))

    return entities
```

## Configuration Constants

```python
# Intent Classification
INTENT_CONFIDENCE_THRESHOLD = 0.20
# Lower threshold accounts for holographic interference
# (bundled prototypes have reduced per-example similarity)

# Pattern Matching
PATTERN_MATCH_THRESHOLD = 0.15

# Learning Factors
LEARNING_STRENGTHEN_FACTOR = 1.5  # Successful patterns
LEARNING_WEAKEN_FACTOR = 0.8      # Failed patterns

# Style Adaptation
STYLE_ADAPTATION_MIN_MESSAGES = 3

# Memory
MAX_CONVERSATION_TURNS = 10
CONTEXT_LOOKBACK = 3
```

## API Reference

### IntentClassifier

```python
from hologram.conversation import IntentClassifier, IntentType

classifier = IntentClassifier(codebook)

# Classify input
result = classifier.classify("Hello there!")
print(result.intent)      # IntentType.GREETING
print(result.confidence)  # 0.36

# Teach new pattern
classifier.learn("yo what's up", IntentType.GREETING)

# Check learned examples
print(classifier.get_example_counts())
# {'greeting': 11, 'question': 16, 'statement': 10, 'farewell': 10}
```

### ConversationalChatbot

```python
from hologram.container import HologramContainer

container = HologramContainer(dimensions=10000)
chatbot = container.create_conversational_chatbot()

# Start session
greeting = chatbot.start_session()

# Converse (implicit learning happens automatically)
response = chatbot.respond("Hello!")

# Teach facts
chatbot.teach_fact("France", "capital", "Paris")

# Query learned facts
response = chatbot.respond("What is the capital of France?")
# → "The capital of France is Paris."

# Session statistics
stats = chatbot.get_session_stats()
# {
#   'turns': 3,
#   'inferred_style': 'neutral',
#   'style_confidence': 0.3,
#   'patterns_count': 15,
#   'messages_observed': 3
# }
```

### ResponsePatternStore

```python
from hologram.conversation import ResponsePatternStore

store = ResponsePatternStore(vector_space, codebook)

# Match patterns
matches = store.match(
    intent=IntentType.QUESTION,
    entities=["France", "capital"],
    context_vec=memory.get_context_vector()
)

# Strengthen successful pattern
store.strengthen_pattern("capital_query")

# Weaken failed pattern
store.weaken_pattern("generic_unknown")
```

## Learning Progression Example

### Day 1 (Fresh System)

```
User: "yo"
Bot: "I'm not sure I understand."  # No match
```

### After Teaching

```python
classifier.learn("yo", IntentType.GREETING)
classifier.learn("yo what's up", IntentType.GREETING)
classifier.learn("sup", IntentType.GREETING)
```

### Day 2 (After Learning)

```
User: "yo"
Bot: "Hello! How can I help you?"  # Matches greeting
```

### After Extended Use

- Patterns that lead to natural conversation flow get stronger
- Patterns that confuse users get weaker
- Style adapts to match user's communication style
- New entities and facts are learned from conversation

## Key Design Principles

1. **No hardcoded keywords**: All classification is learned from examples
2. **Holographic interference**: Bundled vectors "remember" all components
3. **Hebbian learning**: "Neurons that fire together wire together"
4. **Graceful degradation**: Unknown inputs return low confidence, not errors
5. **Pure HDC**: No external LLMs, no gradient descent, no neural networks

## Theoretical Background

The learning mechanisms are grounded in:

- **Hyperdimensional Computing** (Kanerva, 2009): High-dimensional vectors as distributed representations
- **Vector Symbolic Architectures** (Gayler, 2003): Algebraic operations on symbol-like vectors
- **Holographic Reduced Representations** (Plate, 1995): Circular convolution for binding
- **Hebbian Learning** (Hebb, 1949): Synaptic strengthening through co-activation

The key insight is that in high-dimensional spaces (~10,000D), random vectors are nearly orthogonal with high probability. This enables:

- **Clean bundling**: Superposition without destructive interference
- **Reversible binding**: Associations that can be queried
- **Content-addressable memory**: Retrieval via similarity

## Files

```
src/hologram/conversation/
├── __init__.py          # Package exports
├── intent.py            # IntentClassifier with example learning
├── entity.py            # EntityExtractor via resonance
├── memory.py            # ConversationMemory with traces
├── patterns.py          # ResponsePatternStore with Hebbian learning
├── selector.py          # ResponseSelector orchestration
├── style_tracker.py     # UserStyleTracker
└── chatbot.py           # ConversationalChatbot main class
```

## Fact Persistence (ChromaDB)

Facts taught during conversation now persist across sessions using ChromaDB.

### How It Works

Facts are stored as HDC vectors in ChromaDB with cosine distance:

```python
# In ChromaFactStore.add_fact()
key_vec = Operations.bind(subj_vec, pred_vec)  # Create query key
fact_vec = Operations.bundle(key_vec, obj_vec)  # Add answer

self._collection.upsert(
    ids=[fact_id],
    embeddings=[fact_vec.tolist()],  # HDC vector
    metadatas=[{"subject": subject, "predicate": predicate, "object": obj}],
)
```

### Usage

```python
from hologram.container import HologramContainer

# Create persistent chatbot (facts persist to disk)
container = HologramContainer(dimensions=10000)
chatbot = container.create_persistent_chatbot("./data/my_facts")

# Session 1: Teach facts
chatbot.respond("the capital of Germany is Berlin")  # Stored in ChromaDB

# Exit and restart...

# Session 2: Facts are still there
chatbot.respond("What is the capital of Germany?")  # Returns "Berlin"
```

### Configuration

```python
# Default persistence directory
ChatInterface(persist_dir="./data/hologram_facts")

# Disable persistence (in-memory only)
ChatInterface(persistent=False)
```

### ChromaDB Collection Settings

- **Distance metric**: Cosine (HDC vectors encode meaning in direction)
- **Storage**: Local persistent SQLite + Parquet
- **No external services**: Everything runs locally

## Future Directions

1. ~~**Persistent learning**: Save/load learned patterns across sessions~~ ✅ DONE (ChromaDB)
2. **Explicit feedback**: "That's wrong" → weaken pattern directly
3. **Multi-turn learning**: Learn from successful multi-turn sequences
4. **Vocabulary expansion**: Automatic entity discovery from conversation
5. **Style generation**: Use SesameModulator for style-appropriate responses
6. **Pattern persistence**: Save/load intent prototypes and response patterns

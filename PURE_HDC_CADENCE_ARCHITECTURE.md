# Pure-HDC Cadence Architecture: Design Innovation Report

**Date**: December 16, 2025
**Purpose**: Design a pure-HDC system for human-like, multi-sentence responses without external LLM
**Status**: Comprehensive Critique + Architectural Blueprint

---

## Executive Summary

The proposed approach is **conceptually sound and highly innovative** but faces **three critical bottlenecks**:

1. **Current ResonantGenerator is fundamentally 3-token limited** (S-V-O only)
2. **NeuralMemory can learn patterns but lacks structural supervision** to extract cadence
3. **Training pipeline doesn't isolate STRUCTURE from CONTENT**, making pattern learning indirect

**Verdict**: The idea will work IF we add three minimal layers:
- **StructureExtractor**: Extract sentence templates from training data
- **CadenceNetwork**: Neural net learning transitions between sentence patterns
- **CompositionController**: Multi-sentence orchestration layer

This report provides a complete design with minimal viable implementation path.

---

## Part 1: Current Architecture Critique

### 1.1 What ResonantGenerator Does Well

```python
# Current pipeline (ResonantGenerator.generate):
1. Thought Vector → Resonator → (subject, verb, object) words
2. (S, V, O) → TargetEncoder → constraint tensor
3. Constraint tensor → Token-by-token generation with verification
4. Final: "subject verb object" (3 tokens only)
```

**Strengths**:
- ✓ Perfectly constrains 3-token output (no hallucination)
- ✓ Verifies every token against facts (divergence checking)
- ✓ Style modulation via JazzTemplate works elegantly
- ✓ Dreamer enables creative exploration when low confidence

**Critical Limitation**:
- ✗ **CANNOT generate beyond 3 tokens** (architectural ceiling)
- ✗ **No sequential composition** (can't generate "sent1. sent2. sent3.")
- ✗ **No discourse modeling** (no concept of transition, elaboration, contrast)
- ✗ **No pronoun tracking** (each sentence is independent)

### 1.2 NeuralMemory Capabilities & Limits

**What it CAN do**:
```python
# Current NeuralMemory learns:
input_dim=10000 (HDC vector)
  ↓
hidden_dim=256 (learned representation)
  ↓
output_dim=1000 (vocabulary classification)

# Result: key_vector → predicted_label_index
# Example: fact_vector("Paris") → output_dim[123] → "capital_of_france"
```

**What it CANNOT currently do**:
```python
# Does NOT learn:
❌ Sentence structure patterns (templates)
❌ Position-dependent role sequences (e.g., DETERMINER before NOUN)
❌ Discourse transitions ("Also,", "Therefore,", "However,")
❌ Multi-token sequences (only single label predictions)
❌ Pronoun resolution (no context threading)
```

**Why Pattern Learning is Indirect**:
- Training extracts atomic facts: "Paris is the capital of France"
- System stores fact_vector → "Paris"
- But NEVER explicitly extracts or trains on:
  - The STRUCTURE "The [ENTITY] of [ENTITY2] is [FACT]"
  - The PATTERN that structures should be reusable
  - The CADENCE that makes multi-sentence flows natural

### 1.3 Current Crew Trainer Extraction (Lines 1128-1152)

```python
# crew_trainer.py extracts:
def extract_atomic_sentences(response: str):
    # Splits "The sun is a star. The sun produces light." into:
    # ["The sun is a star.", "The sun produces light."]

    # Then stores each with:
    # - sentence text
    # - context_vector (from the HDC system)
    # - intent classification
    # - entity extraction

    # BUT: Never extracts or stores:
    # - Template patterns from these sentences
    # - Structure vectors separate from content
    # - Transition patterns between sentences
    # - Cadence metadata (sentence length, complexity)
```

---

## Part 2: The Innovation - Pure-HDC Cadence System

### 2.1 Core Insight: Separate Structure from Content

**Current (implicit mixing)**:
```
Response = Fact ⊗ Style
Example: "Paris" ⊗ FORMAL → no template variation
```

**Proposed (explicit separation)**:
```
Response = Content ⊗ Structure ⊗ Transition_Pattern

Content Layer:        Structure Layer:      Transition Layer:
"Paris"              "The [X] is [Y]"     "INITIAL" or "ELABORATION"
"France"             Position-encoded      → chains sentences
"capital"            with <D NOUN> binding
```

### 2.2 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           PURE-HDC MULTI-SENTENCE GENERATION               │
└─────────────────────────────────────────────────────────────┘

Layer 3: DISCOURSE CONTROLLER
┌──────────────────────────────────────────┐
│ Multi-Sentence Orchestration             │
│ - Sentence count decision (1-3)          │
│ - Transition type selection              │
│ - Context threading (pronouns)           │
│ - Coherence verification                 │
└──────────────────────────────────────────┘
         ↑                      ↑
         │                      │ (train via crew_trainer)
         │                      │
Layer 2: CADENCE NETWORK (Neural)
┌──────────────────────────────────────────┐
│ NeuralMemory Extension                   │
│ - Input: context_vector                  │
│ - Predicts: next_sentence_template       │
│ - Predicts: transition_type              │
│ - Learns via: experience replay          │
└──────────────────────────────────────────┘
         ↑
         │
Layer 1: STRUCTURE EXTRACTOR (HDC + Neural)
┌──────────────────────────────────────────┐
│ Template Learning                        │
│ - Input: LLM sentence from training     │
│ - Extract: position-encoded template    │
│ - Store: cadence pattern vector         │
│ - Hebbian strengthen: successful uses   │
└──────────────────────────────────────────┘
         ↑
         │
      Crew Trainer Data
```

### 2.3 Three Core Components

#### Component 1: StructureExtractor (NEW)

**Purpose**: Extract TEMPLATES from training sentences, separate from FACTS

**Implementation**:
```python
class StructureExtractor:
    """
    Extract sentence templates from natural language examples.

    Example:
        Input:  "The capital of France is Paris"
        Output: Template: "The [ARTICLE] [NOUN] of [NOUN] is [NOUN]"
                Content: ["capital", "France", "Paris"]
                Roles:   ["ARTICLE", "NOUN", "OF", "NOUN", "IS", "NOUN"]
    """

    def extract(self, sentence: str) -> (template_vec, content_facts, role_sequence):
        """
        Use SequenceEncoder to create position-aware template.

        Step 1: POS tagging or simple heuristic rules
        Step 2: Replace content words with role markers
        Step 3: Encode template as position-encoded vector
        Step 4: Store separately from facts

        Example:
            sentence = "The sun produces heat"
            roles = [ARTICLE, NOUN, VERB, NOUN]
            template_tokens = ["the", "__NOUN_1__", "produces", "__NOUN_2__"]
            template_vec = SequenceEncoder.encode(template_tokens)
        """
        pass

    def match_content_to_template(self, facts, template):
        """
        Given facts (Paris, is, capital), fill template.

        Example:
            template = "The [NOUN] of [NOUN] is [NOUN]"
            facts = ["capital", "France", "Paris"]
            filled = "The capital of France is Paris"
        """
        pass
```

**Training Trigger**:
```python
# In crew_trainer.py, when LLM generates response:
response = "The sun is a star. Stars produce energy."

# Currently stores: entire response
# NEW: also extract templates

for sentence in response.split("."):
    template_vec = extractor.extract(sentence)
    # Store in pattern memory (see below)
```

#### Component 2: CadenceNetwork (Neural Extension)

**Purpose**: Learn TRANSITION PATTERNS and MULTI-SENTENCE STRUCTURES

**Architecture**:
```python
class CadenceNetwork(nn.Module):
    """
    Learns sentence-level transitions and discourse patterns.

    UNLIKE NeuralMemory (which predicts single outputs),
    CadenceNetwork predicts SEQUENCES of structures.
    """

    def __init__(self, input_dim=10000):
        super().__init__()
        # Encode context → hidden state
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Predict next sentence template
        self.template_predictor = nn.Linear(256, input_dim)

        # Predict transition type (INITIAL, ELABORATION, CONTRAST, etc.)
        self.transition_predictor = nn.Linear(256, 4)  # 4 transition types

        # Predict sentence count (1, 2, or 3 sentences)
        self.count_predictor = nn.Linear(256, 3)

    def forward(self, context_vec):
        """
        Input: Current thought + context
        Output:
            - next_template_vec (HDC template)
            - transition_type (one of 4)
            - num_sentences (1, 2, or 3)
        """
        hidden = self.context_encoder(context_vec)
        template = self.template_predictor(hidden)
        transition = self.transition_predictor(hidden)
        count = self.count_predictor(hidden)
        return template, transition, count
```

**Training via Crew Trainer**:
```python
# In crew_trainer.py, after LLM response:
response = "The sun is a star. It produces light."

# Extract sentence structures
sent1_template = extractor.extract("The sun is a star")
sent2_template = extractor.extract("It produces light")

# Create training fact for CadenceNetwork:
training_fact = ConsolidationFact(
    key_vector=context_vec,  # What we knew before generating
    value_label=f"TEMPLATE:{sent1_template}|TRANS:INITIAL|COUNT:2",
    # Also includes template for sent2
)

# Train via NeuralMemory (or dedicated CadenceNetwork instance)
cadence_network.consolidate([training_fact])
```

#### Component 3: DiscoursController (Orchestration)

**Purpose**: Manage multi-sentence generation with coherence

**Implementation**:
```python
class DiscourseController:
    """
    Orchestrates multi-sentence generation with context threading.
    """

    def __init__(self,
                 resonator: Resonator,
                 generator: ResonantGenerator,
                 cadence_network: CadenceNetwork,
                 max_sentences: int = 3):
        self.resonator = resonator
        self.generator = generator
        self.cadence_network = cadence_network
        self.max_sentences = max_sentences

    def generate_response(self,
                          thought_vec: torch.Tensor,
                          style: StyleType) -> str:
        """
        Generate multi-sentence response.

        Algorithm:
        1. Query cadence_network for sentence count and first template
        2. For each sentence:
           a. Decompose thought into (template_slot, remaining_thought)
           b. Fill template with facts
           c. Get transition type for next sentence
           d. Update context for pronoun resolution
        3. Verify coherence
        4. Return joined response
        """

        # Get discourse plan from CadenceNetwork
        template_vec, transition_type, num_sentences = \
            self.cadence_network(thought_vec)

        sentences = []
        current_context = thought_vec.clone()

        for sent_idx in range(num_sentences):
            # Generate sentence from template
            sentence = self._fill_template(
                template_vec,
                current_context,
                transition_type[sent_idx] if sent_idx > 0 else None
            )
            sentences.append(sentence)

            # Update context for next sentence (pronoun resolution)
            current_context = self._update_context(
                current_context,
                sentence,
                sent_idx
            )

            # Get next template if not last sentence
            if sent_idx < num_sentences - 1:
                template_vec, transition_type = \
                    self.cadence_network.get_next_template(current_context)

        # Verify coherence
        if not self._verify_coherence(sentences):
            # Fallback to single sentence
            return sentences[0]

        return ". ".join(sentences) + "."

    def _fill_template(self, template_vec, context, transition):
        """
        Given template and context facts, generate actual sentence.

        Example:
            template_vec → "The [NOUN] of [NOUN] is [NOUN]"
            facts from context → "sun", "sky", "star"
            result → "The sun of sky is star"
        """
        # Use SequenceEncoder.decode_at_position to extract template tokens
        # Then use ResonantGenerator to fill content
        pass

    def _update_context(self, context, sentence, sent_idx):
        """
        Update context for next sentence (pronoun resolution).

        Example:
            sentence = "The sun is a star"
            entities = ["sun", "star"]
            new_context = bind(context, encode("pronoun:it→sun"))
            # So next sentence can use "it" to refer to "sun"
        """
        # Extract entities from sentence
        # Create pronoun bindings
        # Bundle into context
        pass
```

---

## Part 3: Training Modifications

### 3.1 Modified Crew Trainer (Extended)

**Current flow**:
```
LLM generates → Store atomic sentences → Done
```

**Proposed flow**:
```
LLM generates
    ↓
├─ Extract atomic sentences (CURRENT)
│
├─ NEW: Extract templates from sentences
│   └─ Store in pattern memory
│
├─ NEW: Extract transitions (punctuation, connectives)
│   └─ "Also,", "However,", implicit continuations
│
├─ NEW: Create training facts for CadenceNetwork
│   └─ context_vec → (template, transition, count)
│
└─ Train both NeuralMemory (facts) + CadenceNetwork (cadence)
```

**Code sketch** (in crew_trainer.py):
```python
def train_on_llm_response(self, response: str, context_vec: torch.Tensor):
    """
    Enhanced training: extract BOTH facts and STRUCTURE patterns.
    """
    sentences = response.split(". ")

    # Existing: teach facts
    for sentence in sentences:
        self.container.chatbot.teach(sentence)

    # NEW: extract and teach structures
    templates = []
    transitions = []

    for i, sentence in enumerate(sentences):
        # Extract template (separate structure from content)
        template_vec = self.structure_extractor.extract(sentence)
        templates.append(template_vec)

        # Extract transition (what connects to next sentence)
        if i < len(sentences) - 1:
            transition = self._detect_transition(sentence, sentences[i+1])
            transitions.append(transition)

    # Create training fact for cadence network
    cadence_fact = ConsolidationFact(
        key_vector=context_vec,
        value_label=f"TEMPLATES:{templates}|TRANSITIONS:{transitions}|COUNT:{len(sentences)}",
        # In practice, store vectors, not strings
    )

    # Train cadence network
    self.cadence_network.consolidate([cadence_fact])
```

### 3.2 New Training Data Extraction

**Template patterns to extract**:
```python
STRUCTURAL_PATTERNS = {
    "SIMPLE_FACT": "The [NOUN] is [ADJECTIVE]",
    "PROPERTY": "The [NOUN] of [NOUN] is [NOUN]",
    "ACTION": "[NOUN] [VERB] [NOUN]",
    "CAUSAL": "[NOUN] [VERB] because [NOUN] [VERB]",
    "ELABORATION": "[NOUN] [VERB]. Also, [PRONOUN] [VERB] [NOUN]",
}

# Train system to recognize and reproduce these patterns
```

**Transition patterns to extract**:
```python
TRANSITION_TYPES = {
    "INITIAL": None,  # First sentence
    "ELABORATION": ["Also,", "Additionally,", "Furthermore,", implicit],
    "CAUSAL": ["Because", "Since", "Due to"],
    "CONTRAST": ["However,", "But", "On the other hand,"],
}

# Train system to select appropriate transitions
```

---

## Part 4: Minimal Viable Implementation

### Phase 1: Foundation (Week 1-2)

**1a. Implement StructureExtractor**
- File: `src/hologram/generation/structure_extractor.py`
- Use simple POS tagging (can be rule-based for MVP)
- Create position-encoded templates via SequenceEncoder
- Store templates as vectors in a pattern dictionary

**1b. Extend NeuralMemory with StructureVector outputs**
- Allow output_dim to be 10000 (HDC dimension) for template predictions
- Train on: context_vec → template_vec

**1c. Add pattern storage to crew_trainer.py**
- Extract templates when learning from LLM
- Store in new `pattern_memory` (similar to fact_store)

### Phase 2: Cadence Learning (Week 2-3)

**2a. Implement CadenceNetwork**
- File: `src/hologram/generation/cadence_network.py`
- Start with simple 2-layer network
- Train on: context_vec → (template, transition_type, count)

**2b. Integrate into crew_trainer.py**
- Create ConsolidationFact for cadence patterns
- Train via NeuralMemory consolidation
- Experience replay for Hebbian learning

### Phase 3: Orchestration (Week 3-4)

**3a. Implement DiscourseController**
- File: `src/hologram/generation/discourse_controller.py`
- Multi-sentence generation loop
- Context threading for pronouns

**3b. Integration testing**
- Modify chatbot.py to use DiscourseController instead of ResonantGenerator
- Test on varied inputs (facts, questions, statements)

### Phase 4: Refinement (Week 4+)

**4a. Pronoun resolution**
- Track entities across sentences
- Bind pronouns to entities in context

**4b. Coherence verification**
- Check sentence compatibility
- Prevent contradictions

**4c. Hebbian learning tuning**
- Adjust strength weights based on human feedback

---

## Part 5: Hard Problem Solutions

### 5.1 Pronoun Resolution

**Problem**: "Paris is a city. It is beautiful" - how does system know "it" = "Paris"?

**Solution**: Context threading
```python
def _update_context(self, context, sentence, sent_idx):
    """
    After generating sentence, extract main entity and bind pronoun.

    Example:
        sentence = "Paris is a city"
        main_entity = "Paris"  # Use dependency parsing or simple heuristic

        # Bind pronoun "it" to this entity
        pronoun_binding = bind(
            encode("it"),
            encode("refers_to:Paris")
        )

        # Update context for next sentence
        new_context = bundle(context, pronoun_binding)
    """
```

**Heuristic for MVP**: Subject of sentence = current pronoun referent

### 5.2 Topic Coherence

**Problem**: Cadence network might predict unrelated templates

**Solution**: Coherence constraint
```python
def _verify_coherence(self, sentences):
    """
    Check that sentences stay on topic.

    Heuristic: All sentences should have high similarity to original thought_vec
    """
    for sentence in sentences:
        sentence_vec = encode_sentence(sentence)
        sim = cosine(sentence_vec, self.thought_vec)
        if sim < 0.3:  # Threshold
            return False
    return True
```

### 5.3 Natural Transitions

**Problem**: "Also," is awkward every time

**Solution**: Learn from LLM data
```python
# During crew_trainer, extract actual transitions used:
# "The sun is a star. It produces light" → implicit continuation
# "The sun is a star. However, it's not sentient" → contrast

# System learns: context_vec → most likely transition
# Not template-based, but learned via neural network
```

---

## Part 6: Expected Outcomes

### 6.1 Before (Current ResonantGenerator)
```
Input: thought about "Paris"
Output: "Paris is capital" (3 tokens max)

Limitations:
- No elaboration
- No context
- No natural flow
```

### 6.2 After (Pure-HDC Cadence System)
```
Input: thought about "Paris"
Output: "Paris is the capital of France. It's a beautiful city.
         Many tourists visit there." (3 sentences, ~15 tokens)

Benefits:
- Multi-sentence coherence
- Natural transitions
- Elaboration from facts
- 0% hallucination (still pure HDC)
```

### 6.3 Quality Metrics

**Measurement**: After 1000 crew_trainer conversations

- **Cadence Score**: % of 2+ sentence responses (target: >60%)
- **Coherence Score**: Human rating of topic consistency (target: >8/10)
- **Transition Score**: % of natural transitions (target: >70%)
- **Hallucination Rate**: % of facts contradicting stored knowledge (target: 0%)
- **Latency**: Time to generate 3-sentence response (target: <500ms)

---

## Part 7: Architecture Diagram (Complete System)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION INTERFACE (chat.py)                     │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      DISCOURSE CONTROLLER (NEW)                         │
│  Orchestrates multi-sentence generation with context threading        │
├─────────────────────────────────────────────────────────────────────────┤
│  Input: thought_vec (10000-dim)                                         │
│  Output: "Sentence 1. Sentence 2. Sentence 3."                          │
└───┬────────────────┬────────────────┬────────────────────────────────────┘
    │                │                │
    ↓                ↓                ↓
  For each sentence in output:
    │
    ├─→ CADENCE NETWORK (NEW)        Get template + transition
    │   (Neural, learns cadence)
    │   input: context_vec
    │   output: template_vec, transition_type, count
    │
    ├─→ STRUCTURE EXTRACTOR (NEW)    Extract POS from template
    │   (HDC + heuristic)
    │   input: template_vec
    │   output: [NOUN], [VERB], [ADJ] sequence
    │
    └─→ RESONANT GENERATOR (existing) Fill template with facts
        (HDC, constrained)
        input: facts + template
        output: actual sentence text

        ├─ Resonator: thought → (S, V, O)
        ├─ TargetEncoder: (S, V, O) → constraint tensor
        ├─ ReEncoder: candidate tokens
        ├─ DivergenceCalculator: verify facts
        └─ SesameModulator: add style

    ↓ (after sentence generated, update context)

    Context Update:
    - Extract entities from sentence
    - Bind pronouns to entities
    - Update for next iteration
```

---

## Part 8: Implementation Checklist

### Phase 1: Foundation
- [ ] Create `StructureExtractor` class
- [ ] Implement POS tagging (rule-based for MVP)
- [ ] Create position-encoded templates
- [ ] Add `pattern_memory` to HologramContainer
- [ ] Modify crew_trainer to extract templates

### Phase 2: Cadence
- [ ] Create `CadenceNetwork` class
- [ ] Implement training in crew_trainer
- [ ] Add experience replay for cadence patterns
- [ ] Test on 100+ training examples

### Phase 3: Orchestration
- [ ] Create `DiscourseController` class
- [ ] Implement multi-sentence loop
- [ ] Add context threading
- [ ] Integration test with chatbot.py

### Phase 4: Refinement
- [ ] Pronoun resolution via entity tracking
- [ ] Coherence verification
- [ ] Transition naturalness tuning
- [ ] Human evaluation (10-20 samples)

### Phase 5: Production
- [ ] Performance optimization
- [ ] Caching templates
- [ ] Monitoring & metrics
- [ ] Documentation

---

## Part 9: Risk Analysis

### Risk 1: Template Extraction Accuracy
**Problem**: Simple POS tagging might fail on complex sentences
**Mitigation**: Start with simple sentences (crew_trainer uses simple sentences), add rules incrementally
**Fallback**: If template extraction fails, use single-sentence mode

### Risk 2: Cadence Network Hallucination
**Problem**: Network might predict templates that don't match thoughts
**Mitigation**: Coherence verification (check similarity to original thought)
**Fallback**: Filter by confidence threshold

### Risk 3: Pronoun Resolution Errors
**Problem**: "The capital of France is Paris. It has many museums" - what is "it"?
**Mitigation**: Conservative heuristic - only bind to clear subjects
**Fallback**: Use explicit noun repeats if pronoun resolution uncertain

### Risk 4: Training Data Bias
**Problem**: If crew_trainer only shows simple patterns, system won't learn complex discourse
**Mitigation**: Ensure crew_trainer uses diverse LLM outputs (varied topics, styles)
**Fallback**: Manual pattern seeding for important structures

---

## Part 10: Key Insights

### Insight 1: Content vs. Structure Separation
The breakthrough is realizing that **cadence is learnable independently from facts**. An HDC system trained on pure facts can ALSO learn how to structure those facts into natural discourse.

### Insight 2: Neural Network Learns Meta-Patterns
NeuralMemory can predict not just "what words to say" but also "how to structure multiple words". The key is training it on EXTRACTED STRUCTURES, not raw sentences.

### Insight 3: Crew Trainer as Automatic Curator
The existing crew_trainer.py is already collecting data. By adding structure extraction, we automatically get a dataset of (context → cadence_pattern) pairs.

### Insight 4: 0% Hallucination is Achievable
Unlike LLMs that generate text character-by-character, this system:
1. Learns patterns from data
2. Uses those patterns as TEMPLATES
3. Fills templates with FACTS from memory
4. Verifies every token against constraints

Result: Multi-sentence responses grounded in learned facts.

---

## Part 11: Questions for Refinement

1. **Sentence Segmentation**: Should StructureExtractor handle multi-clause sentences, or only simple SVO?
   - MVP Answer: Only simple sentences

2. **Pronoun Coverage**: Which pronouns to support (it, he, she, they, this)?
   - MVP Answer: Just "it" (most common)

3. **Transition Learning**: Should transitions be predicted by CadenceNetwork or hard-coded?
   - MVP Answer: Predict from neural net (learned from crew_trainer data)

4. **Hallucination Risk**: What confidence threshold for coherence check?
   - MVP Answer: 0.3 similarity to original thought

5. **Scalability**: How many templates to store before performance degrades?
   - MVP Answer: Start with 100, optimize if needed

---

## Conclusion

The **Pure-HDC Cadence Architecture is viable and elegant**:

✓ Stays true to HDC principles (hypervectors for everything)
✓ Eliminates hallucination (templates + fact verification)
✓ Learns from existing crew_trainer data
✓ Achieves human-like cadence through structure learning
✓ Minimal changes to existing codebase (3 new modules)

**Next Step**: Implement Phase 1 (StructureExtractor + pattern storage) as proof-of-concept, measure template extraction accuracy on crew_trainer data.

---

## References

- NeuralMemory: `/src/hologram/consolidation/neural_memory.py`
- SequenceEncoder: `/src/hologram/memory/sequence_encoder.py`
- Resonator: `/src/hologram/core/resonator.py`
- ResonantGenerator: `/src/hologram/generation/resonant_generator.py`
- JazzTemplate: `/src/hologram/generation/jazz.py`
- Dreamer: `/src/hologram/generation/dreamer.py`
- Crew Trainer: `/scripts/crew_trainer.py`
- Patterns: `/src/hologram/conversation/patterns.py`

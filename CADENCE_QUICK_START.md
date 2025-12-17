# Pure-HDC Cadence: Quick Start Guide

**For developers implementing the architecture**

---

## 30-Second Overview

Current: "Paris is capital" (3 tokens, robotic)
Goal: "Paris is the capital of France. It's a beautiful city." (3 sentences, natural)

How: Learn sentence TEMPLATES + TRANSITIONS (not facts) via neural consolidation

---

## The Three Components

### 1️⃣ StructureExtractor
**What it does**: Converts sentences to templates
```
Input:  "The sun is a star"
Output: "The [NOUN] is a [NOUN]" (as HDC vector)
```

**Implementation**: Rule-based POS tagging for MVP
**File**: `src/hologram/generation/structure_extractor.py`
**When ready**: Week 1

### 2️⃣ CadenceNetwork
**What it does**: Learns "what template comes next"
```
Input:  context_vec (thought about sun)
Output: (template_for_next_sentence, transition_type, count)
```

**Implementation**: Simple 2-layer neural net
**File**: `src/hologram/generation/cadence_network.py`
**When ready**: Week 2

### 3️⃣ DiscourseController
**What it does**: Orchestrates multi-sentence generation
```
For each sentence:
  1. Get template from CadenceNetwork
  2. Fill template with facts (using ResonantGenerator)
  3. Update pronoun context
  4. Verify coherence
Join sentences with transitions
```

**Implementation**: State machine + entity tracking
**File**: `src/hologram/generation/discourse_controller.py`
**When ready**: Week 3

---

## Data Flow Diagram

```
thought_vec (10000-dim)
    ↓
CadenceNetwork.predict(thought_vec)
    ↓ returns: (template_vec, transition, count)
    ├─→ StructureExtractor.match_template(template_vec)
    │   Returns: "The [NOUN] is a [NOUN]"
    │
    ├─→ ResonantGenerator.generate(...)
    │   Fills: "The sun is a star"
    │
    ├─→ EntityContext.update()
    │   Binds: "it" → "sun"
    │
    └─→ DiscourseController.verify_coherence()
        Check: Is "sun" on-topic? Yes → accept

Repeat for sentence 2, 3 (until count exhausted)
    ↓
Result: "The sun is a star. It produces light."
```

---

## Minimal Implementation Path

### Week 1: StructureExtractor

```python
# src/hologram/generation/structure_extractor.py

class StructureExtractor:
    def extract(self, sentence: str) -> ExtractedStructure:
        """
        "The sun is a star" → "The [NOUN] is a [NOUN]"
        """
        # Step 1: Tokenize
        tokens = sentence.split()

        # Step 2: Simple heuristic POS tagging
        roles = self._guess_roles(tokens)

        # Step 3: Create template tokens
        template_tokens = self._create_template(tokens, roles)

        # Step 4: Encode as position-aware vector
        template_vec = self.sequence_encoder.encode(template_tokens)

        return ExtractedStructure(
            template_vector=template_vec,
            template_tokens=template_tokens,
            content_words=[t for t, r in zip(tokens, roles) if r == "NOUN"],
        )
```

**Test**: Can extract templates from 100 crew_trainer sentences with >80% reconstruction accuracy

### Week 2: CadenceNetwork

```python
# src/hologram/generation/cadence_network.py

class CadenceNetwork(nn.Module):
    def forward(self, context_vec):
        """
        context_vec (10000) → (template_vec, transition_type, count)
        """
        hidden = self.encoder(context_vec)
        template = self.template_predictor(hidden)     # 10000-dim
        transition = self.transition_predictor(hidden) # 4-dim (0-3)
        count = self.count_predictor(hidden)           # 3-dim (1-3 sentences)
        return template, transition, count
```

**Test**: Transition prediction >60% accuracy on held-out crew_trainer data

### Week 3: DiscourseController

```python
# src/hologram/generation/discourse_controller.py

class DiscourseController:
    def generate_response(self, thought_vec, style):
        """
        Multi-sentence orchestration.
        """
        num_sentences = self.cadence_network.predict_count(thought_vec)
        sentences = []

        for i in range(num_sentences):
            # Get template
            template = self.cadence_network.predict_template(thought_vec)

            # Generate sentence
            sentence = self._fill_template(template, thought_vec)

            # Verify coherence
            if self._is_coherent(sentence, thought_vec):
                sentences.append(sentence)
            else:
                break  # Stop if losing coherence

        return ". ".join(sentences) + "."
```

**Test**: Generate 2+ sentence responses, verify 0% hallucination

---

## Integration Checklist

- [ ] Create `src/hologram/generation/structure_extractor.py`
- [ ] Create `src/hologram/generation/cadence_network.py`
- [ ] Create `src/hologram/generation/discourse_controller.py`
- [ ] Modify `scripts/crew_trainer.py`:
  - Add template extraction
  - Add CadenceNetwork training
- [ ] Modify `src/hologram/container.py`:
  - Add `create_discourse_controller()` method
- [ ] Modify `src/hologram/conversation/chatbot.py`:
  - Use DiscourseController instead of ResonantGenerator (for testing)
- [ ] Add tests in `tests/generation/test_discourse_controller.py`

---

## Common Pitfalls

### ❌ Pitfall 1: Templates are too specific
```
BAD: Template = exact sentence "the sun is a star"
     (Can't reuse for "the moon is a star")

GOOD: Template = "the [NOUN] is a [NOUN]"
      (Reusable for any two nouns)
```

**Fix**: Use role markers [NOUN], [VERB], [ADJ], not words

### ❌ Pitfall 2: CadenceNetwork overfits to training data
```
BAD: Network only learns 3 templates from crew_trainer
     (Produces repetitive outputs)

GOOD: Regularize via deduplication penalty
      (Encourage diverse templates)
```

**Fix**: Add loss term penalizing duplicate template predictions

### ❌ Pitfall 3: Coherence check too strict
```
BAD: Reject sentences with <0.9 similarity
     (Blocks all elaborations)

GOOD: Accept <0.5 similarity, reject <0.2
      (Allow variations, block off-topic)
```

**Fix**: Tune coherence_threshold empirically (start at 0.3)

### ❌ Pitfall 4: Pronoun resolution too aggressive
```
BAD: Bind "it" to any noun mentioned
     "The capital of France is Paris. It [what is it?] is beautiful."
     (Ambiguous)

GOOD: Bind only to clear subjects
      "Paris is a city. It [clearly Paris] is beautiful."
```

**Fix**: Use simple heuristic (subject of previous sentence only)

---

## Testing Strategy

### Unit Tests
```python
def test_structure_extraction():
    """Can templates be extracted and reconstructed?"""

def test_cadence_transition_prediction():
    """Does CadenceNetwork predict transitions >50% accuracy?"""

def test_discourse_coherence():
    """Are generated sentences on-topic?"""
```

### Integration Tests
```python
def test_end_to_end_generation():
    """Full pipeline: thought → multi-sentence response"""

def test_zero_hallucination():
    """No facts contradict fact store"""
```

### Data-Driven Tests
```python
def test_on_crew_trainer_data():
    """Run 100 crew_trainer examples through system"""
    # Measure:
    # - % with 2+ sentences
    # - Coherence score
    # - Hallucination rate
```

---

## Debugging Checklist

**Problem: Only generating 1 sentence**
- [ ] Check: CadenceNetwork.predict_count() returning 1
- [ ] Check: Coherence verification failing (threshold too high?)
- [ ] Check: ResonantGenerator failing silently

**Problem: Sentences are off-topic**
- [ ] Check: Coherence threshold (try 0.2)
- [ ] Check: Template extraction mangling input
- [ ] Check: CadenceNetwork predicting random templates

**Problem: Pronouns are wrong**
- [ ] Check: EntityContext.bind_pronoun() working
- [ ] Check: Entity extraction from sentences (simple split)
- [ ] Check: Pronoun detection logic

**Problem: Latency too high**
- [ ] Check: CadenceNetwork inference time (should be <10ms)
- [ ] Check: ResonantGenerator per-token latency
- [ ] Check: Coherence verification (do only per-sentence, not per-token)

---

## Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Template extraction accuracy | >80% | Can reconstruct sentence from template |
| Transition prediction | >60% | Better than random (25%) |
| Coherence check | >70% true negatives | Blocks mostly-off-topic sentences |
| Multi-sentence rate | >60% | Most responses use 2+ sentences |
| Latency | <500ms | Still feels interactive |
| Hallucination rate | 0% | Maintain current standard |

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feat/pure-hdc-cadence

# Week 1: StructureExtractor
git add src/hologram/generation/structure_extractor.py
git commit -m "feat: Implement StructureExtractor for template learning"

# Week 2: CadenceNetwork
git add src/hologram/generation/cadence_network.py
git commit -m "feat: Implement CadenceNetwork for discourse planning"

# Week 3: DiscourseController
git add src/hologram/generation/discourse_controller.py
git commit -m "feat: Implement DiscourseController for multi-sentence orchestration"

# Integration
git add scripts/crew_trainer.py src/hologram/container.py src/hologram/conversation/chatbot.py
git commit -m "feat: Integrate cadence components with crew trainer"

# Tests
git add tests/generation/test_discourse_controller.py
git commit -m "test: Add comprehensive discourse controller tests"

# Create PR
git push origin feat/pure-hdc-cadence
# Open PR, link to PURE_HDC_CADENCE_ARCHITECTURE.md
```

---

## Key Files to Read First

1. `src/hologram/memory/sequence_encoder.py` - Position-aware encoding
2. `src/hologram/consolidation/neural_memory.py` - Neural learning pattern
3. `src/hologram/generation/resonant_generator.py` - Current 3-token generation
4. `scripts/crew_trainer.py` - Training data pipeline
5. `src/hologram/conversation/patterns.py` - Pattern storage reference

---

## Questions to Ask Before Starting

1. **Scope**: Do we want all 3 components or just StructureExtractor?
2. **Templates**: Simple (templates only) or complex (with discourse relations)?
3. **Training**: Use only crew_trainer, or add manual seed patterns?
4. **Fallback**: Single-sentence if coherence fails, or try to fix?
5. **Evaluation**: What's acceptable hallucination rate? (Target: 0%)

---

## Success Criteria (Week 4 Review)

- [ ] StructureExtractor: 80%+ reconstruction accuracy
- [ ] CadenceNetwork: Trained on 1000+ examples, 60%+ transition accuracy
- [ ] DiscourseController: Generates 2+ sentence responses 60%+ of time
- [ ] Integration: Works with existing chatbot
- [ ] Hallucination: 0% (maintain current)
- [ ] Latency: <500ms average

**If all met**: Ship to production
**If 1-2 miss**: Fix and re-evaluate
**If 3+ miss**: Reduce scope to StructureExtractor only (single-sentence templates)

---

## Useful Debugging Commands

```bash
# Test StructureExtractor
python -m pytest tests/generation/test_structure_extractor.py -v

# Profile generation latency
python -c "from scripts.profile import profile_generation; profile_generation()"

# Validate templates on crew_trainer data
python -c "from scripts.validate_templates import validate_on_crew_trainer; validate_on_crew_trainer()"

# Interactive testing
python
>>> from hologram.generation.discourse_controller import DiscourseController
>>> controller = DiscourseController(...)
>>> response = controller.generate_response(thought_vec)
>>> print(response)
```

---

## Resources

- Architecture: `/PURE_HDC_CADENCE_ARCHITECTURE.md` (full design)
- Technical details: `/CADENCE_TECHNICAL_APPENDIX.md` (code sketches)
- Executive summary: `/CADENCE_EXECUTIVE_BRIEF.md` (decision-making)
- This document: Quick start for implementation

---

**Start with Week 1 (StructureExtractor) and validate before proceeding to Week 2.**

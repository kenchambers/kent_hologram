# Pure-HDC Cadence Architecture: Technical Appendix

**Deep Dive into Implementation Details, Code Sketches, and Validation Strategies**

---

## Appendix A: StructureExtractor - Complete Implementation Sketch

### A.1 MVP Implementation (Rule-Based)

```python
# File: src/hologram/generation/structure_extractor.py

from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
from hologram.memory.sequence_encoder import SequenceEncoder
from hologram.core.codebook import Codebook

@dataclass
class ExtractedStructure:
    """Result of structure extraction."""
    template_vector: torch.Tensor      # HDC vector of template
    template_tokens: List[str]         # ["The", "NOUN", "is", "NOUN"]
    content_slots: List[str]           # ["NOUN_1", "NOUN_2"]
    content_words: List[str]           # Words that go in slots
    roles: List[str]                   # ["DET", "NOUN", "VERB", "NOUN"]
    sentence: str                      # Original sentence

class StructureExtractor:
    """
    Extract sentence structure templates.

    MVP Strategy: Simple rule-based POS tagging
    - Use hardcoded patterns for common structures
    - Later: upgrade to actual POS tagger if needed

    Example sentences for MVP:
    - "The sun is a star" → "The [NOUN] is a [NOUN]"
    - "Paris is beautiful" → "[PROPER_NOUN] is [ADJ]"
    - "Water boils at 100 degrees" → "[NOUN] [VERB] at [NUMBER] [NOUN]"
    """

    # Hard-coded templates for MVP
    COMMON_TEMPLATES = {
        # (sentence pattern regex) → (structure pattern)
        r"^the\s+(\w+)\s+is\s+a?\s+(\w+)$": "The [NOUN] is a [NOUN]",
        r"^(\w+)\s+is\s+(\w+)$": "[NOUN] is [NOUN]",
        r"^(\w+)\s+(\w+)\s+(\w+)$": "[NOUN] [VERB] [NOUN]",
    }

    def __init__(self, codebook: Codebook, sequence_encoder: SequenceEncoder):
        self.codebook = codebook
        self.sequence_encoder = sequence_encoder

    def extract(self, sentence: str) -> ExtractedStructure:
        """
        Extract structure from sentence.

        Algorithm:
        1. Normalize sentence (lowercase, remove punctuation)
        2. Match against template patterns
        3. Replace content words with role markers
        4. Encode template as position-aware vector
        5. Extract content slots
        """
        normalized = sentence.lower().strip(".")

        # Step 1: Try to match hard-coded patterns
        for pattern, template in self.COMMON_TEMPLATES.items():
            import re
            match = re.match(pattern, normalized)
            if match:
                content_words = [g for g in match.groups()]
                return self._make_structure(
                    sentence, template, content_words
                )

        # Step 2: Fallback - simple heuristic tokenization
        tokens = normalized.split()
        if len(tokens) <= 6:
            # For short sentences, assume pattern: role_sequence
            roles = self._guess_roles(tokens)
            template = self._construct_template(tokens, roles)
            return self._make_structure(sentence, template, tokens)

        # Step 3: If all else fails, return raw sentence as pseudo-structure
        return ExtractedStructure(
            template_vector=self.sequence_encoder.encode_sentence(sentence),
            template_tokens=tokens,
            content_slots=tokens,
            content_words=tokens,
            roles=["UNKNOWN"] * len(tokens),
            sentence=sentence,
        )

    def _guess_roles(self, tokens: List[str]) -> List[str]:
        """
        Simple heuristic: guess POS tags.

        Rules:
        - First token after "the" or "a" is NOUN
        - "is", "are", "was" → VERB
        - Common verbs: "have", "produce", "create" → VERB
        - Numbers → NUM
        """
        roles = []
        articles = {"the", "a", "an"}

        for i, token in enumerate(tokens):
            if token in articles:
                roles.append("DET")
            elif token in {"is", "are", "was", "were", "be"}:
                roles.append("VERB")
            elif token in {"have", "has", "produce", "produces", "create"}:
                roles.append("VERB")
            elif token[0].isdigit():
                roles.append("NUM")
            elif i > 0 and tokens[i-1] in articles:
                roles.append("NOUN")
            else:
                roles.append("NOUN")  # Default

        return roles

    def _construct_template(self, tokens: List[str], roles: List[str]) -> str:
        """Construct template string from tokens and roles."""
        template_parts = []
        for token, role in zip(tokens, roles):
            if role in ["NOUN", "VERB", "ADJ"]:
                template_parts.append(f"[{role}]")
            else:
                template_parts.append(token)
        return " ".join(template_parts)

    def _make_structure(self,
                       sentence: str,
                       template: str,
                       content_words: List[str]) -> ExtractedStructure:
        """Construct ExtractedStructure object."""
        # Encode template as sequence
        template_tokens = template.lower().split()
        template_vec = self.sequence_encoder.encode(template_tokens)

        # Extract role markers
        roles = [t.strip("[]") for t in template_tokens if t.startswith("[")]

        return ExtractedStructure(
            template_vector=template_vec,
            template_tokens=template_tokens,
            content_slots=roles,
            content_words=content_words,
            roles=roles,
            sentence=sentence,
        )

    def fill_template(self,
                     template: str,
                     content_words: List[str]) -> str:
        """
        Fill template with content words.

        Example:
            template = "The [NOUN] is a [NOUN]"
            content_words = ["sun", "star"]
            result = "The sun is a star"
        """
        import re
        result = template
        for word in content_words:
            result = re.sub(r"\[[A-Z_]+\]", word, result, count=1)
        return result
```

### A.2 Extraction Quality Metrics

```python
class StructureExtractorValidator:
    """Validate extraction quality."""

    def evaluate(self, extractor, test_sentences: List[str]):
        """
        Run evaluation on test sentences.

        Metrics:
        - Template matches (did it find a template?)
        - Content preservation (can we reconstruct original?)
        - Role accuracy (are roles correct?)
        """
        metrics = {
            "total_sentences": len(test_sentences),
            "successful_extractions": 0,
            "reconstructed_correctly": 0,
            "role_errors": 0,
        }

        for sentence in test_sentences:
            structure = extractor.extract(sentence)

            # Can we reconstruct the sentence?
            reconstructed = extractor.fill_template(
                " ".join(structure.template_tokens),
                structure.content_words
            )

            if reconstructed.lower() == sentence.lower():
                metrics["reconstructed_correctly"] += 1

            metrics["successful_extractions"] += 1

        # Calculate success rate
        metrics["reconstruction_rate"] = (
            metrics["reconstructed_correctly"] /
            metrics["total_sentences"]
        )

        return metrics
```

---

## Appendix B: CadenceNetwork - Neural Implementation

### B.1 Core Network Architecture

```python
# File: src/hologram/generation/cadence_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class CadenceOutput:
    """Output from CadenceNetwork."""
    next_template_vec: torch.Tensor     # Next sentence template
    transition_type: int                 # 0=INITIAL, 1=ELAB, 2=CONTRAST, 3=CAUSAL
    confidence: float                    # Confidence in prediction
    num_sentences: Optional[int] = None  # Predicted total count

class CadenceNetwork(nn.Module):
    """
    Neural network for learning discourse patterns.

    UNLIKE NeuralMemory (classification head):
    - Input: context_vec (10000-dim)
    - Outputs:
      1. next_template_vec (10000-dim) → for template matching
      2. transition_type (one of 4)
      3. confidence (0-1) → how sure about prediction

    Training via experience replay:
    - Store (context, observed_template, transition, count) tuples
    - Replay during training
    - Hebbian strengthening of good predictions
    """

    TRANSITION_TYPES = {
        0: "INITIAL",
        1: "ELABORATION",
        2: "CONTRAST",
        3: "CAUSAL",
    }

    def __init__(self,
                 input_dim: int = 10000,
                 hidden_dim: int = 512,
                 num_transitions: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions

        # Context encoder: input_dim → hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Template predictor: predict next template vector
        self.template_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),  # Output is HDC-dim
        )

        # Transition predictor: predict transition type
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_transitions),
        )

        # Confidence predictor: predict confidence in template
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # Output 0-1
        )

        # Sentence count predictor (optional, for planning)
        self.count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),  # Predict 1, 2, or 3 sentences
        )

    def forward(self, context_vec: torch.Tensor) -> CadenceOutput:
        """
        Predict next sentence structure given context.

        Args:
            context_vec: Thought vector or current context (10000-dim)

        Returns:
            CadenceOutput with predictions
        """
        # Handle batch vs. single vector
        if context_vec.dim() == 1:
            context_vec = context_vec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Encode context to hidden representation
        hidden = self.context_encoder(context_vec)  # (batch, hidden_dim)

        # Predict next template
        next_template = self.template_predictor(hidden)  # (batch, input_dim)

        # Predict transition type
        transition_logits = self.transition_predictor(hidden)  # (batch, num_trans)
        transition_probs = F.softmax(transition_logits, dim=-1)
        transition_type = torch.argmax(transition_probs, dim=-1)  # (batch,)

        # Predict confidence
        confidence = self.confidence_predictor(hidden)  # (batch, 1)

        # Predict number of sentences (optional)
        count_logits = self.count_predictor(hidden)
        count_probs = F.softmax(count_logits, dim=-1)
        num_sentences = torch.argmax(count_probs, dim=-1) + 1  # 1, 2, or 3

        if squeeze_output:
            return CadenceOutput(
                next_template_vec=next_template.squeeze(0),
                transition_type=int(transition_type.item()),
                confidence=float(confidence.squeeze().item()),
                num_sentences=int(num_sentences.item()),
            )
        else:
            return CadenceOutput(
                next_template_vec=next_template,
                transition_type=transition_type,
                confidence=confidence.squeeze(-1),
                num_sentences=num_sentences,
            )
```

### B.2 Training on Cadence Patterns

```python
class CadenceTrainer:
    """Train CadenceNetwork via experience replay."""

    def __init__(self, network: CadenceNetwork, learning_rate: float = 1e-3):
        self.network = network
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.replay_buffer = []

    def add_training_example(self,
                            context_vec: torch.Tensor,
                            next_template_vec: torch.Tensor,
                            transition_type: int,
                            confidence_target: float):
        """
        Add a training example (from crew_trainer output).

        Args:
            context_vec: Thought/context before generating sentence
            next_template_vec: Template that was actually generated
            transition_type: Type of transition to next sentence
            confidence_target: Human confidence in prediction (0-1)
        """
        self.replay_buffer.append({
            "context": context_vec,
            "template": next_template_vec,
            "transition": transition_type,
            "confidence": confidence_target,
        })

    def train_step(self, batch_size: int = 16) -> float:
        """
        Run one training step on replay buffer.

        Loss = L_template + L_transition + L_confidence
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        import random
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))

        # Unpack batch
        contexts = torch.stack([b["context"] for b in batch])
        templates = torch.stack([b["template"] for b in batch])
        transitions = torch.tensor([b["transition"] for b in batch])
        confidences = torch.tensor([b["confidence"] for b in batch], dtype=torch.float32)

        # Forward pass
        self.optimizer.zero_grad()
        self.network.train()

        outputs = self.network(contexts)

        # Template loss: MSE between predicted and actual template
        template_loss = F.mse_loss(outputs.next_template_vec, templates)

        # Transition loss: cross-entropy for transition type prediction
        transition_logits = self.network.transition_predictor(
            self.network.context_encoder(contexts)
        )
        transition_loss = F.cross_entropy(transition_logits, transitions)

        # Confidence loss: MSE for confidence prediction
        confidence_loss = F.mse_loss(
            outputs.confidence.unsqueeze(-1),
            confidences.unsqueeze(-1)
        )

        # Total loss
        total_loss = template_loss + 0.5 * transition_loss + 0.3 * confidence_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return float(total_loss.item())
```

---

## Appendix C: DiscourseController - Multi-Sentence Orchestration

### C.1 Core Implementation

```python
# File: src/hologram/generation/discourse_controller.py

from typing import List, Optional, Dict, Tuple
import torch
from hologram.core.resonator import Resonator
from hologram.generation.resonant_generator import ResonantGenerator
from hologram.generation.cadence_network import CadenceNetwork
from hologram.generation.structure_extractor import StructureExtractor
from hologram.core.similarity import Similarity
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations

class EntityContext:
    """Track entities and pronouns for context threading."""

    def __init__(self, codebook: Codebook):
        self.codebook = codebook
        self.entities: Dict[str, torch.Tensor] = {}  # entity_name → vector
        self.pronoun_bindings: Dict[str, str] = {}   # pronoun → entity_name
        self.last_subject: Optional[str] = None      # For default pronoun resolution

    def add_entity(self, name: str, vector: torch.Tensor):
        """Add entity to context."""
        self.entities[name.lower()] = vector
        if self.last_subject is None:
            self.last_subject = name

    def bind_pronoun(self, pronoun: str, entity: str):
        """Bind pronoun to entity (e.g., "it" → "Paris")."""
        self.pronoun_bindings[pronoun.lower()] = entity.lower()

    def get_pronoun_vector(self, pronoun: str) -> Optional[torch.Tensor]:
        """Get vector for pronoun."""
        entity = self.pronoun_bindings.get(pronoun.lower())
        if entity:
            return self.entities.get(entity)
        return None

    def to_context_vector(self) -> torch.Tensor:
        """Bundle all entities and bindings into HDC vector."""
        vectors = list(self.entities.values())
        if not vectors:
            return torch.zeros(10000)  # Empty context
        return Operations.bundle(*vectors)


class DiscourseController:
    """
    Orchestrates multi-sentence generation.

    Pipeline:
    1. Query CadenceNetwork for discourse plan
    2. For each sentence:
       a. Get template from CadenceNetwork or pattern memory
       b. Generate sentence by filling template with facts
       c. Update entity context
    3. Verify coherence
    4. Join sentences with transitions
    """

    def __init__(self,
                 resonator: Resonator,
                 generator: ResonantGenerator,
                 cadence_network: CadenceNetwork,
                 structure_extractor: StructureExtractor,
                 codebook: Codebook,
                 max_sentences: int = 3,
                 coherence_threshold: float = 0.3):
        self.resonator = resonator
        self.generator = generator
        self.cadence_network = cadence_network
        self.structure_extractor = structure_extractor
        self.codebook = codebook
        self.max_sentences = max_sentences
        self.coherence_threshold = coherence_threshold

    def generate_response(self,
                         thought_vec: torch.Tensor,
                         style=None) -> str:
        """
        Generate multi-sentence response.

        Args:
            thought_vec: Initial thought/context vector
            style: Optional StyleType for modulation

        Returns:
            Multi-sentence response string
        """
        # Plan discourse
        cadence_output = self.cadence_network(thought_vec)
        num_sentences = cadence_output.num_sentences or 2

        sentences = []
        entity_context = EntityContext(self.codebook)
        current_thought = thought_vec.clone()
        current_sentence_idx = 0

        for sent_idx in range(min(num_sentences, self.max_sentences)):
            # Generate one sentence
            sentence, entities = self._generate_single_sentence(
                current_thought,
                sent_idx,
                entity_context,
                style
            )

            if not sentence:
                break  # Generation failed

            sentences.append(sentence)

            # Update context for next sentence
            for ent_name, ent_vec in entities.items():
                entity_context.add_entity(ent_name, ent_vec)

            # Update thought for next iteration (remove generated content)
            # In practice, use a more sophisticated approach
            current_thought = self._update_thought(current_thought, sentence)

            current_sentence_idx = sent_idx

        # Verify coherence
        if not self._verify_coherence(sentences, thought_vec):
            # Fallback to single sentence
            return sentences[0] if sentences else "I'm uncertain about that."

        # Join sentences with transitions
        response = self._join_sentences(sentences)
        return response

    def _generate_single_sentence(self,
                                  context_vec: torch.Tensor,
                                  sentence_idx: int,
                                  entity_context: EntityContext,
                                  style) -> Tuple[Optional[str], Dict[str, torch.Tensor]]:
        """
        Generate a single sentence from context.

        Returns:
            (sentence_text, extracted_entities)
        """
        # Get template from CadenceNetwork
        cadence_output = self.cadence_network(context_vec)
        template_vec = cadence_output.next_template_vec

        # Decompose thought using Resonator
        resonator_result = self.resonator.resonate(
            context_vec,
            self.generator._vocabulary["nouns"],
            self.generator._vocabulary["verbs"],
        )

        # Generate sentence from template
        subject = resonator_result.subject_word
        verb = resonator_result.verb_word
        obj = resonator_result.object_word

        # Very simple for MVP: just output S-V-O
        # In production: match template and fill slots
        sentence = f"{subject} {verb} {obj}."

        # Extract entities from sentence (simple heuristic)
        extracted_entities = {
            subject: self.codebook.encode(subject),
            obj: self.codebook.encode(obj),
        }

        return sentence, extracted_entities

    def _update_thought(self,
                       current_thought: torch.Tensor,
                       sentence: str) -> torch.Tensor:
        """
        Update thought for next sentence.

        Simple approach: remove this sentence's entities from thought.
        Could be more sophisticated.
        """
        # Extract entities from sentence
        # Unbind from thought
        # (MVP: just return thought - this is a simplification)
        return current_thought

    def _verify_coherence(self,
                         sentences: List[str],
                         original_thought: torch.Tensor) -> bool:
        """
        Verify that sentences stay on topic.

        Heuristic: all sentences should have non-negligible similarity
        to original thought vector.
        """
        for sentence in sentences:
            # Very simple encoding (in practice use full codebook)
            sentence_vec = torch.mean(
                torch.stack([
                    self.codebook.encode(word.lower().strip(",."))
                    for word in sentence.split()
                ]),
                dim=0
            )

            sim = Similarity.cosine(sentence_vec, original_thought)

            if sim < self.coherence_threshold:
                return False

        return True

    def _join_sentences(self, sentences: List[str]) -> str:
        """
        Join sentences with appropriate transitions.

        MVP: simple space-joining. In production, use transition types.
        """
        if len(sentences) == 1:
            return sentences[0]

        # Simple joining with periods
        result = " ".join(sentences)
        if not result.endswith("."):
            result += "."

        return result
```

---

## Appendix D: Integration with Crew Trainer

### D.1 Modified Crew Trainer Flow

```python
# In scripts/crew_trainer.py, modify training loop:

def process_llm_response(self, response: str, context_vec: torch.Tensor):
    """
    Process LLM response: teach facts AND structures.

    Current: just teaches facts
    New: also extract and teach structures
    """

    # Step 1: Existing - teach facts
    sentences = response.split(". ")
    for sentence in sentences:
        self.container.chatbot.teach(sentence)

    # Step 2: NEW - extract structures
    templates = []
    transitions = []

    for i, sentence in enumerate(sentences):
        # Extract template
        extracted = self.structure_extractor.extract(sentence)
        templates.append(extracted.template_vector)

        # Extract transition from punctuation/connectives
        if i < len(sentences) - 1:
            next_sentence = sentences[i + 1]
            transition_type = self._detect_transition_type(
                sentence, next_sentence
            )
            transitions.append(transition_type)

    # Step 3: NEW - create training facts for CadenceNetwork
    if len(templates) > 1:
        # Multi-sentence training example
        cadence_training_fact = ConsolidationFact(
            key_vector=context_vec,
            value_label=f"CADENCE:{len(sentences)}",
            # In practice, encode as vector, not string
        )

        # Train cadence network
        self.cadence_network.consolidate([cadence_training_fact])

        # Also add to pattern memory
        for i, template in enumerate(templates):
            transition = transitions[i] if i < len(transitions) else 0
            self.pattern_memory.add_pattern(
                context=context_vec,
                template=template,
                transition=transition,
                strength=1.0,
            )

def _detect_transition_type(self, sent1: str, sent2: str) -> int:
    """
    Detect transition type between sentences.

    Returns: 0=INITIAL, 1=ELABORATION, 2=CONTRAST, 3=CAUSAL
    """
    # Check for explicit connectives
    sent2_lower = sent2.lower()

    if any(word in sent2_lower for word in ["however", "but", "although"]):
        return 2  # CONTRAST

    if any(word in sent2_lower for word in ["because", "since", "therefore"]):
        return 3  # CAUSAL

    if any(word in sent2_lower for word in ["also", "additionally", "moreover"]):
        return 1  # ELABORATION

    # Default: elaboration (most common)
    return 1
```

---

## Appendix E: Validation Experiments

### E.1 Template Extraction Accuracy

```python
def evaluate_template_extraction():
    """
    Measure how well StructureExtractor works.

    Gold standard: hand-annotated templates for 100 sentences
    """
    test_sentences = [
        ("The sun is a star", "The [NOUN] is a [NOUN]"),
        ("Paris is in France", "[NOUN] is in [NOUN]"),
        ("Water boils at 100 degrees", "[NOUN] [VERB] at [NUM] [NOUN]"),
        # ... 97 more
    ]

    extractor = StructureExtractor(codebook, sequence_encoder)
    correct = 0

    for sentence, expected_template in test_sentences:
        extracted = extractor.extract(sentence)
        predicted_template = " ".join(extracted.template_tokens)

        if predicted_template == expected_template:
            correct += 1

    accuracy = correct / len(test_sentences)
    print(f"Template Extraction Accuracy: {accuracy:.1%}")
    return accuracy
```

### E.2 CadenceNetwork Prediction Accuracy

```python
def evaluate_cadence_network():
    """
    Measure how well CadenceNetwork predicts transitions.
    """
    # Collect test examples from crew_trainer
    test_examples = [
        # (context_vec, next_template_vec, transition_type)
    ]

    network = CadenceNetwork()
    correct_transitions = 0
    correct_templates = 0

    for context, true_template, true_transition in test_examples:
        output = network(context)

        # Check transition prediction
        if output.transition_type == true_transition:
            correct_transitions += 1

        # Check template prediction (cosine similarity)
        sim = Similarity.cosine(output.next_template_vec, true_template)
        if sim > 0.8:
            correct_templates += 1

    print(f"Transition Prediction: {correct_transitions / len(test_examples):.1%}")
    print(f"Template Prediction: {correct_templates / len(test_examples):.1%}")
```

### E.3 Multi-Sentence Coherence

```python
def evaluate_discourse_controller():
    """
    Measure coherence of generated multi-sentence responses.

    Metrics:
    - % of responses with 2+ sentences
    - Average similarity to thought vector
    - Human evaluation (need 10-20 samples)
    """
    test_thoughts = [
        encode_sentence("The sun is bright"),
        encode_sentence("Paris is a city"),
        # ... more
    ]

    controller = DiscourseController(...)
    responses = []
    coherence_scores = []

    for thought in test_thoughts:
        response = controller.generate_response(thought)
        responses.append(response)

        # Measure coherence
        response_vec = encode_sentence(response)
        coherence = Similarity.cosine(response_vec, thought)
        coherence_scores.append(coherence)

    multi_sentence_count = sum(1 for r in responses if "." in r)

    print(f"Multi-sentence responses: {multi_sentence_count}/{len(test_thoughts)}")
    print(f"Average coherence: {sum(coherence_scores)/len(coherence_scores):.3f}")
```

---

## Appendix F: Performance Profiling

### F.1 Latency Analysis

```python
import time

def profile_generation():
    """
    Measure generation latency breakdown.
    """

    controller = DiscourseController(...)
    thought = encode_sentence("Paris is the capital of France")

    # Warmup
    controller.generate_response(thought)

    # Profile
    timings = {}
    iterations = 100

    for _ in range(iterations):
        # CadenceNetwork query
        start = time.time()
        cadence_output = cadence_network(thought)
        timings["cadence_query"] = timings.get("cadence_query", 0) + (time.time() - start)

        # Sentence generation
        start = time.time()
        sentence, _ = controller._generate_single_sentence(thought, 0, None, None)
        timings["resonant_gen"] = timings.get("resonant_gen", 0) + (time.time() - start)

        # Coherence verification
        start = time.time()
        controller._verify_coherence([sentence], thought)
        timings["coherence_check"] = timings.get("coherence_check", 0) + (time.time() - start)

    print("Average latency per component (ms):")
    for component, total_time in timings.items():
        avg_ms = (total_time / iterations) * 1000
        print(f"  {component}: {avg_ms:.1f}ms")
```

---

## Appendix G: Common Failure Modes & Fixes

### G.1 Failure: Template Overfitting

**Symptom**: System learns only 3-4 templates, generates repetitive sentences

**Cause**: Limited training data or weak pattern regularization

**Fix**:
```python
# Add regularization to pattern memory
class PatternMemory:
    def add_pattern(self, template_vec, strength=1.0):
        # Penalize duplicate templates
        similar_patterns = self.find_similar(template_vec, threshold=0.95)
        if similar_patterns:
            strength *= 0.7  # Reduce strength for duplicates
```

### G.2 Failure: Hallucination Despite Constraints

**Symptom**: System generates "Paris is a vegetable" (contradicts facts)

**Cause**: Coherence check threshold too low

**Fix**:
```python
# Increase coherence threshold
controller = DiscourseController(
    ...,
    coherence_threshold=0.5,  # Was 0.3
)

# Or add fact verification
def verify_facts(sentence):
    tokens = sentence.split()
    for word in tokens:
        if word not in fact_store and word not in vocabulary:
            return False  # Reject unknown words
    return True
```

### G.3 Failure: Pronoun Errors

**Symptom**: "The sun is bright. It is red" (correct) vs. "Paris is a city. It is a country" (wrong)

**Cause**: No semantic checking of pronoun bindings

**Fix**:
```python
def bind_pronoun_safely(pronoun, entity):
    """Only bind pronouns to semantically compatible entities."""
    # Check: entity must be singular noun
    if entity not in singular_nouns:
        return None  # Reject binding
    # Bind safely
    entity_context.bind_pronoun(pronoun, entity)
```

---

## Appendix H: Future Enhancements (Beyond MVP)

### H.1 Advanced Template Extraction
- Real POS tagger instead of heuristics
- Dependency parsing for complex sentences
- Clause-level templates (compound sentences)

### H.2 Pronoun Resolution
- Track all entities, not just last subject
- Gender/number agreement checking
- Resolve "that", "this", "which" references

### H.3 Discourse Coherence
- Use entity grids (tracking entity salience)
- Semantic role labeling (who did what to whom)
- Information structure (given vs. new information)

### H.4 Transition Learning
- Learn transition types from crew_trainer
- Hebbian strengthening of transitions
- Context-aware transition selection

### H.5 Multi-Modal Generation
- Support different output formats (bullet points, lists)
- Adapt to user style preferences
- Generate with explanatory chains

---

## Appendix I: Test Suite Skeleton

```python
# File: tests/generation/test_discourse_controller.py

import pytest
import torch
from hologram.generation.discourse_controller import DiscourseController

class TestDiscourseController:

    def test_single_sentence_generation(self):
        """Baseline: can generate a single sentence?"""
        controller = setup_discourse_controller()
        thought = torch.randn(10000)
        response = controller.generate_response(thought)
        assert len(response) > 0
        assert "." in response

    def test_multi_sentence_generation(self):
        """Main test: can generate 2+ sentences?"""
        controller = setup_discourse_controller()
        thought = torch.randn(10000)
        response = controller.generate_response(thought)
        num_sentences = response.count(".") - 1
        assert num_sentences >= 1

    def test_coherence_verification(self):
        """Coherence check filters bad generations?"""
        controller = setup_discourse_controller()
        thought = torch.randn(10000)
        bad_sentences = ["apple zebra blah", "unknown xyz algorithm"]
        is_coherent = controller._verify_coherence(bad_sentences, thought)
        assert not is_coherent

    def test_entity_context_threading(self):
        """Entity context carries across sentences?"""
        entity_context = EntityContext(codebook)
        entity_context.add_entity("Paris", torch.randn(10000))
        entity_context.bind_pronoun("it", "Paris")
        assert entity_context.get_pronoun_vector("it") is not None

    def test_transition_detection(self):
        """Can detect transition types?"""
        trainer = CrewTrainer()
        trans = trainer._detect_transition_type(
            "The sun is bright",
            "However, it's not the largest star"
        )
        assert trans == 2  # CONTRAST
```

---

End of Technical Appendix

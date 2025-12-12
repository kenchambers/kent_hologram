"""
Intent classification using HDC example-based learning.

Classifies user input into intent categories by comparing against
learned prototype vectors. Prototypes are built from example phrases,
not hardcoded keywords - the system learns patterns holographically.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch

from hologram.config.constants import (
    INTENT_CONFIDENCE_THRESHOLD,
    QUESTION_START_WORDS,
    GREETING_WORDS,
    FAREWELL_WORDS,
    TEACHING_PATTERNS,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


class IntentType(Enum):
    """User intent categories."""

    GREETING = "greeting"
    QUESTION = "question"
    STATEMENT = "statement"
    TEACHING = "teaching"  # User is teaching a fact
    FAREWELL = "farewell"
    COMMAND = "command"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    all_scores: Dict[str, float]

    def __repr__(self) -> str:
        return f"IntentResult({self.intent.value}, conf={self.confidence:.2f})"


# Seed examples for bootstrapping (learned, not keyword lists)
# Each example teaches the system what this intent "sounds like"
# More examples = better generalization, but also more interference
SEED_EXAMPLES = {
    IntentType.GREETING: [
        "hello",
        "hi",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "greetings",
        "hey there",
        "howdy",
        "whats up",
        "what is up",
        "how are you",
        "how is it going",
        "yo",
    ],
    IntentType.QUESTION: [
        # Direct question starters
        "what is the",
        "what are the",
        "who is the",
        "where is the",
        "when is the",
        "why is the",
        "how is the",
        # Question with "of"
        "what is the capital of",
        "what is the color of",
        "what is the name of",
        "who is the creator of",
        # Full questions
        "what is the capital of france",
        "what is the capital of germany",
        "who created python",
        "where is tokyo",
        # Question markers
        "can you tell me",
        "do you know",
        "could you explain",
        "would you help",
        "tell me about",
        "what do you know about",
    ],
    IntentType.STATEMENT: [
        # Agreement/Opinion
        "i think so",
        "that sounds right",
        "yes definitely",
        "no way",
        "actually i believe",
        "in my opinion",
        "i know that",
        "interesting point",
        "makes sense",
        "i agree",
        "yeah for sure",
        "totally",
        "i see what you mean",
        # Observations
        "the weather is nice",
        "it is a beautiful day",
        "the sky looks great",
        "i like that movie",
        "this is interesting",
        "that is really cool",
        "wow that is amazing",
        "the dog is cute",
        "the food tastes good",
        "the music is loud",
        # Reflections
        "i wonder if",
        "i was thinking",
        "that is cool",
        "wow",
        "imagine that",
        "that makes sense to me",
        "it seems like",
        "i notice that",
        "i remember when",
        # Conversational statements (common in multi-party conversations)
        "you know i was just thinking",
        "i was just thinking about that",
        "isnt it kinda wild",
        "isnt it weird how",
        "you ever think about",
        "i find that interesting",
        "thats fascinating",
        "thats pretty cool",
        "i find that pretty interesting",
        "thats a good point",
        "i see what you mean",
        "thats true",
        "youre right about that",
    ],
    IntentType.TEACHING: [
        # "X is Y" patterns
        "the capital of france is paris",
        "the capital of germany is berlin",
        "paris is the capital of france",
        "berlin is the capital of germany",
        # Possessive patterns
        "france's capital is paris",
        "germany's capital is berlin",
        # General teaching with "is"
        "the sky is blue",
        "water boils at 100 degrees",
        "the sun is a star",
        "dogs are mammals",
        "tokyo is in japan",
        "the color of the ocean is blue",
        # Teaching with "was/were" (past tense facts)
        "python was created by guido",
        "ada lovelace was the first programmer",
        "the first computer was invented in the 1940s",
        "the pyramids were built by egyptians",
        # Short proper noun facts (important for learning)
        "einstein is a physicist",
        "einstein was a physicist",
        "newton was a scientist",
        "shakespeare was a playwright",
        "beethoven was a composer",
        "darwin was a naturalist",
        "tesla was an inventor",
        "curie was a scientist",
        # Inverted patterns (object-first)
        "the first programmer was ada lovelace",
        "the inventor of the telephone was alexander graham bell",
        "the creator of python was guido",
        # Remember/learn patterns
        "remember that",
        "know that",
        "the answer is",
        "it is called",
        # Direct property statements
        "the population of tokyo is 14 million",
        "the height of mount everest is 8848 meters",
    ],
    IntentType.FAREWELL: [
        "goodbye",
        "bye",
        "see you later",
        "thanks bye",
        "thank you goodbye",
        "later",
        "take care",
        "have a good day",
        "see you",
        "thanks for your help",
    ],
}


class IntentClassifier:
    """
    HDC-based intent classification using example learning.

    Instead of hardcoded keywords, builds intent prototypes from
    example phrases. The system learns the "shape" of each intent
    through holographic superposition of examples.

    Each example is encoded and bundled into the intent's prototype
    vector. Classification finds the most similar prototype.

    Attributes:
        _codebook: Shared Codebook for encoding
        _intent_vectors: Learned intent prototype vectors
        _example_counts: Number of examples per intent (for tracking)
        _threshold: Minimum confidence for classification

    Example:
        >>> classifier = IntentClassifier(codebook)
        >>> result = classifier.classify("Hello, how are you?")
        >>> print(result.intent)  # IntentType.GREETING
        >>>
        >>> # Teach new pattern
        >>> classifier.learn("yo whats up", IntentType.GREETING)
    """

    def __init__(
        self,
        codebook: Codebook,
        threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        seed: bool = True,
    ):
        """
        Initialize intent classifier.

        Args:
            codebook: Shared Codebook instance
            threshold: Minimum similarity for classification
            seed: Whether to seed with initial examples (default True)
        """
        self._codebook = codebook
        self._threshold = threshold
        self._intent_vectors: Dict[IntentType, torch.Tensor] = {}
        self._example_counts: Dict[IntentType, int] = {}

        # Initialize empty prototypes
        for intent in IntentType:
            if intent not in (IntentType.COMMAND, IntentType.UNKNOWN):
                self._intent_vectors[intent] = None
                self._example_counts[intent] = 0

        # Seed with initial examples
        if seed:
            self._seed_from_examples()

    def _seed_from_examples(self) -> None:
        """Bootstrap intent prototypes from seed examples."""
        for intent, examples in SEED_EXAMPLES.items():
            for example in examples:
                self.learn(example, intent)

    def _encode_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode text into a holographic vector."""
        tokens = self._tokenize(text)
        if not tokens:
            return None

        # Bundle all token vectors
        token_vectors = [self._codebook.encode(t) for t in tokens]
        result = token_vectors[0]
        for vec in token_vectors[1:]:
            result = Operations.bundle(result, vec)
        return result

    def classify(self, text: str) -> IntentResult:
        """
        Classify user input by intent.

        Args:
            text: User input text

        Returns:
            IntentResult with intent type and confidence
        """
        text = text.strip()

        # Check for slash commands first
        if text.startswith("/"):
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=1.0,
                all_scores={"command": 1.0},
            )

        # Encode user input
        input_vec = self._encode_text(text)
        if input_vec is None:
            return self._unknown_result()

        # Compare against each intent prototype
        scores: Dict[str, float] = {}
        for intent_type, prototype in self._intent_vectors.items():
            if prototype is None:
                continue
            sim = Similarity.cosine(input_vec, prototype)
            scores[intent_type.value] = float(sim)

        # Heuristic adjustments for common patterns
        # (HDC with bundled prototypes has low per-example similarity)
        text_lower = text.lower().strip()
        starts_with_question = any(text_lower.startswith(qw) for qw in QUESTION_START_WORDS)
        ends_with_question_mark = text.endswith("?")
        
        # Question heuristic: boost QUESTION for question-like inputs
        if starts_with_question or ends_with_question_mark:
            if "question" in scores:
                # Strong boost for clear question patterns
                scores["question"] = max(scores["question"] * 10.0, 0.25)
            # Reduce other scores
            if "statement" in scores:
                scores["statement"] *= 0.5
            if "teaching" in scores:
                scores["teaching"] *= 0.5
        elif not starts_with_question and not ends_with_question_mark:
            # Likely a statement, not a question - boost STATEMENT score
            if "statement" in scores:
                scores["statement"] *= 1.3
            # And slightly reduce QUESTION score
            if "question" in scores:
                scores["question"] *= 0.8
        
        # Greeting heuristic: boost for common greetings
        if any(text_lower.startswith(gw) for gw in GREETING_WORDS) or text_lower in GREETING_WORDS:
            if "greeting" in scores:
                scores["greeting"] = max(scores["greeting"] * 10.0, 0.25)
        
        # Farewell heuristic: boost for common farewells
        if any(fw in text_lower for fw in FAREWELL_WORDS):
            if "farewell" in scores:
                scores["farewell"] = max(scores["farewell"] * 10.0, 0.25)
        
        # Teaching heuristic: boost for declarative fact patterns
        if any(tp in text_lower for tp in TEACHING_PATTERNS):
            if "teaching" in scores:
                scores["teaching"] = max(scores["teaching"] * 10.0, 0.25)

        # Select best intent
        if not scores:
            return self._unknown_result()

        best_intent_name = max(scores, key=scores.get)
        best_score = scores[best_intent_name]

        if best_score < self._threshold:
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=best_score,
                all_scores=scores,
            )

        return IntentResult(
            intent=IntentType(best_intent_name),
            confidence=best_score,
            all_scores=scores,
        )

    def learn(self, text: str, intent: IntentType) -> None:
        """
        Teach the classifier a new example for an intent.

        This is the core learning mechanism. Each example is encoded
        and bundled into the intent's prototype, strengthening the
        pattern through holographic superposition (Hebbian learning).

        Args:
            text: Example text for this intent
            intent: The correct intent type
        """
        if intent in (IntentType.COMMAND, IntentType.UNKNOWN):
            return

        example_vec = self._encode_text(text)
        if example_vec is None:
            return

        # Bundle into existing prototype or initialize
        if self._intent_vectors.get(intent) is None:
            self._intent_vectors[intent] = example_vec
        else:
            self._intent_vectors[intent] = Operations.bundle(
                self._intent_vectors[intent], example_vec
            )

        self._example_counts[intent] = self._example_counts.get(intent, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace/punctuation."""
        import re

        # Remove punctuation except apostrophes, lowercase
        text = re.sub(r"[^\w\s']", " ", text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 0]

    def _unknown_result(self) -> IntentResult:
        """Return an UNKNOWN intent result."""
        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            all_scores={},
        )

    def get_example_counts(self) -> Dict[str, int]:
        """Get the number of examples learned per intent."""
        return {k.value: v for k, v in self._example_counts.items()}

    def __repr__(self) -> str:
        total = sum(self._example_counts.values())
        return f"IntentClassifier(examples={total}, threshold={self._threshold})"

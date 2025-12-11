"""
Conversational Vocabulary: Extract and categorize words from training.

Builds vocabulary dictionaries for ResonantGenerator by extracting
and categorizing words from conversational training data.
"""

import re
from typing import Dict, List, Set


class ConversationalVocabulary:
    """
    Build vocabulary from conversational training data.

    Extracts nouns, verbs, adjectives, and connectors from training
    conversations to build vocabulary for token-level generation.

    Attributes:
        nouns: Set of noun words
        verbs: Set of verb words
        adjectives: Set of adjective words
        connectors: Set of connector words/phrases
        stop_words: Common stop words to filter

    Example:
        >>> vocab = ConversationalVocabulary()
        >>> vocab.learn_from_text("The cat eats fish. That's interesting!")
        >>> vocab.get_vocabulary_dict()
        {'nouns': ['cat', 'fish'], 'verbs': ['eats'], ...}
    """

    # Common stop words to filter out
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "her", "its",
        "our", "their", "what", "which", "who", "whom", "whose", "where",
        "when", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "now", "then", "here",
        "there", "up", "down", "out", "off", "over", "under", "again",
        "further", "once", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "among", "around", "against",
    }

    # Common connector phrases
    CONNECTOR_PHRASES = [
        "that's", "it's", "i'm", "you're", "we're", "they're", "he's", "she's",
        "that is", "it is", "i am", "you are", "we are", "they are",
        "interesting", "cool", "yeah", "okay", "sure", "right", "hmm",
        "well", "actually", "really", "pretty", "quite", "very", "so",
        "too", "also", "even", "just", "still", "already", "yet",
    ]

    def __init__(self):
        """Initialize empty vocabulary."""
        self.nouns: Set[str] = set()
        self.verbs: Set[str] = set()
        self.adjectives: Set[str] = set()
        self.connectors: Set[str] = set()

    def learn_from_text(self, text: str) -> None:
        """
        Extract and categorize words from text.

        Uses simple heuristics to categorize words:
        - Nouns: capitalized words, common nouns
        - Verbs: common action words
        - Adjectives: descriptive words ending in -ing, -ed, -ly, etc.
        - Connectors: common phrases and filler words

        Args:
            text: Text to learn from
        """
        # Normalize text
        text = text.lower()
        
        # Extract connector phrases first (before tokenization)
        for phrase in self.CONNECTOR_PHRASES:
            if phrase in text:
                self.connectors.add(phrase)

        # Tokenize: split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)
        
        # Filter stop words
        tokens = [t for t in tokens if t not in self.STOP_WORDS]

        # Simple heuristics for categorization
        for token in tokens:
            # Skip if already in connectors
            if token in self.connectors:
                continue

            # Common verb endings
            if token.endswith(("ed", "ing", "es", "s")) and len(token) > 3:
                base = token.rstrip("edings")
                if base:
                    self.verbs.add(base)
                    self.verbs.add(token)  # Also add inflected form

            # Common adjective endings
            if token.endswith(("ly", "ful", "less", "able", "ible", "ic", "ical")):
                self.adjectives.add(token)
            elif token.endswith(("ing", "ed")) and len(token) > 4:
                # Could be adjective or verb - add to both
                self.adjectives.add(token)

            # Common verbs (action words)
            common_verbs = {
                "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did",
                "get", "got", "go", "went", "come", "came",
                "see", "saw", "know", "think", "say", "said",
                "make", "made", "take", "took", "give", "gave",
                "find", "found", "tell", "told", "ask", "asked",
                "work", "worked", "try", "tried", "use", "used",
                "want", "wanted", "need", "needed", "like", "liked",
                "look", "looked", "seem", "seemed", "feel", "felt",
            }
            if token in common_verbs:
                self.verbs.add(token)

            # Common adjectives
            common_adjectives = {
                "good", "bad", "big", "small", "new", "old", "young",
                "long", "short", "high", "low", "right", "wrong",
                "different", "same", "important", "interesting", "cool",
                "nice", "great", "best", "better", "worse", "worst",
            }
            if token in common_adjectives:
                self.adjectives.add(token)

            # Everything else is potentially a noun
            if len(token) > 2:  # Skip very short tokens
                self.nouns.add(token)

    def get_vocabulary_dict(self) -> Dict[str, List[str]]:
        """
        Get vocabulary as dictionary for ResonantGenerator.

        Returns:
            Dictionary with "nouns" and "verbs" keys (as required by ResonantGenerator)
        """
        # Convert sets to sorted lists for consistency
        # ResonantGenerator requires "nouns" and "verbs" keys
        return {
            "nouns": sorted(list(self.nouns)),
            "verbs": sorted(list(self.verbs)),
        }

    def get_all_words(self) -> Set[str]:
        """Get all words in vocabulary."""
        return self.nouns | self.verbs | self.adjectives | self.connectors

    def merge(self, other: "ConversationalVocabulary") -> None:
        """
        Merge another vocabulary into this one.

        Args:
            other: Another ConversationalVocabulary to merge
        """
        self.nouns.update(other.nouns)
        self.verbs.update(other.verbs)
        self.adjectives.update(other.adjectives)
        self.connectors.update(other.connectors)

    def get_stats(self) -> dict:
        """Get vocabulary statistics."""
        return {
            "nouns": len(self.nouns),
            "verbs": len(self.verbs),
            "adjectives": len(self.adjectives),
            "connectors": len(self.connectors),
            "total_words": len(self.get_all_words()),
        }

    def clear(self) -> None:
        """Clear all vocabulary."""
        self.nouns.clear()
        self.verbs.clear()
        self.adjectives.clear()
        self.connectors.clear()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ConversationalVocabulary(nouns={stats['nouns']}, "
            f"verbs={stats['verbs']}, total={stats['total_words']})"
        )

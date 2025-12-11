"""
SequenceEncoder: Positional encoding for ordered sequences.

Encodes sequences (sentences, lists) preserving order using position-binding:
bind(word, position) for each word, then bundle all.

This ensures "Dog bites Man" ≠ "Man bites Dog" because positions differ.
"""

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


class SequenceEncoder:
    """
    Encodes sequences with positional information.

    Uses position-binding to preserve word order:
    For each word at position i:
        bound = bind(word_vector, position_vector)
    Result = bundle(all bound vectors)

    This creates a unique vector for each sequence where order matters.

    Attributes:
        _codebook: Codebook for word and position vectors
        max_length: Maximum sequence length (default: 1000)

    Example:
        >>> codebook = Codebook(VectorSpace())
        >>> encoder = SequenceEncoder(codebook)
        >>> v1 = encoder.encode_sentence("dog bites man")
        >>> v2 = encoder.encode_sentence("man bites dog")
        >>> similarity = cosine_similarity(v1, v2)
        >>> similarity < 0.5  # Different order = low similarity
        True
    """

    def __init__(self, codebook: Codebook, max_length: int = 1000):
        """
        Initialize sequence encoder.

        Args:
            codebook: Codebook for generating word and position vectors
            max_length: Maximum sequence length to support
        """
        self._codebook = codebook
        self.max_length = max_length

    def encode(self, tokens: list[str]) -> torch.Tensor:
        """
        Encode a sequence of tokens.

        For each token at position i:
            word_vec = encode(token)
            pos_vec = get_positional(i)
            bound = bind(word_vec, pos_vec)
        Result = bundle(all bound vectors)

        Args:
            tokens: List of token strings

        Returns:
            Sequence vector encoding both content and order

        Raises:
            ValueError: If sequence exceeds max_length

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> encoder = SequenceEncoder(codebook)
            >>> seq_vec = encoder.encode(["the", "cat", "sat"])
            >>> seq_vec.shape
            torch.Size([10000])
        """
        if len(tokens) == 0:
            raise ValueError("Cannot encode empty sequence")
        if len(tokens) > self.max_length:
            raise ValueError(
                f"Sequence length {len(tokens)} exceeds max_length {self.max_length}"
            )

        bound_tokens = []
        for pos, token in enumerate(tokens):
            word_vec = self._codebook.encode(token)
            pos_vec = self._codebook.get_positional(pos)
            bound = Operations.bind(word_vec, pos_vec)
            bound_tokens.append(bound)

        return Operations.bundle(*bound_tokens)

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """
        Encode a sentence (tokenizes on whitespace).

        Args:
            sentence: Sentence string

        Returns:
            Sequence vector

        Example:
            >>> encoder = SequenceEncoder(Codebook(VectorSpace()))
            >>> vec = encoder.encode_sentence("The quick brown fox")
            >>> vec.shape
            torch.Size([10000])
        """
        tokens = sentence.lower().split()
        return self.encode(tokens)

    def decode_at_position(
        self,
        sequence_vec: torch.Tensor,
        position: int,
        vocabulary: list[str]
    ) -> tuple[str, float]:
        """
        Decode token at specific position.

        Unbinds the positional vector and finds best match in vocabulary.
        This demonstrates that position information is preserved.

        Args:
            sequence_vec: Encoded sequence vector
            position: Position to decode (0-indexed)
            vocabulary: List of possible tokens

        Returns:
            Tuple of (best_match, confidence)

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> encoder = SequenceEncoder(codebook)
            >>> seq = encoder.encode(["cat", "dog", "bird"])
            >>> vocab = ["cat", "dog", "bird", "fish"]
            >>> word, conf = encoder.decode_at_position(seq, 1, vocab)
            >>> word
            'dog'
        """
        if position >= self.max_length:
            raise ValueError(f"Position {position} exceeds max_length {self.max_length}")

        # Unbind position to get word at that position
        pos_vec = self._codebook.get_positional(position)
        word_at_pos = Operations.unbind(sequence_vec, pos_vec)

        # Find best match in vocabulary
        candidates = torch.stack([
            self._codebook.encode(word) for word in vocabulary
        ])
        similarities = Similarity.cosine_batch(word_at_pos, candidates)
        best_idx = torch.argmax(similarities).item()
        confidence = float(similarities[best_idx].item())

        return vocabulary[best_idx], confidence

    def compare_sequences(
        self,
        seq1: list[str],
        seq2: list[str]
    ) -> float:
        """
        Similarity between two sequences.

        Same words in different order will have low similarity,
        demonstrating that positional encoding preserves order.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Cosine similarity between encoded sequences

        Example:
            >>> encoder = SequenceEncoder(Codebook(VectorSpace()))
            >>> sim1 = encoder.compare_sequences(["a", "b"], ["a", "b"])
            >>> sim2 = encoder.compare_sequences(["a", "b"], ["b", "a"])
            >>> sim1 > 0.9  # Same sequence = high similarity
            True
            >>> sim2 < 0.5  # Different order = low similarity
            True
        """
        vec1 = self.encode(seq1)
        vec2 = self.encode(seq2)
        return Similarity.cosine(vec1, vec2)

    def __repr__(self) -> str:
        return f"SequenceEncoder(max_length={self.max_length})"

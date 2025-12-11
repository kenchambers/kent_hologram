"""
ReEncoder: Projects tokens back into HDC space for verification.

The ReEncoder is the inverse of text generation - it takes generated
tokens and converts them back to hypervectors so we can verify they
align with the target constraints.

This enables the "closed loop" of the Resonant Cavity: generate -> re-encode -> verify.
"""

from typing import List, Type

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations


class ReEncoder:
    """
    Projects tokens back to HDC space for verification.

    Handles single tokens and sequences, using permutation for
    positional encoding in sequences.

    The re-encoding process mirrors the original encoding:
    - Single token: encode(token)
    - Sequence: bundle(permute(encode(t0), 0), permute(encode(t1), 1), ...)

    This allows direct comparison between target and generated vectors.

    Attributes:
        _codebook: Codebook for encoding tokens
        _ops: Operations class for HDC operations

    Example:
        >>> re_encoder = ReEncoder(codebook)
        >>> vec = re_encoder.encode_token("cat")
        >>> seq_vec = re_encoder.encode_sequence(["the", "cat", "sat"])
    """

    def __init__(
        self,
        codebook: Codebook,
        operations: Type[Operations] = Operations,
    ):
        """
        Initialize re-encoder.

        Args:
            codebook: Codebook for encoding tokens
            operations: Operations class (default: Operations)
        """
        self._codebook = codebook
        self._ops = operations

    def encode_token(self, token: str) -> torch.Tensor:
        """
        Encode single token to HDC vector.

        Args:
            token: Token string to encode

        Returns:
            Hypervector representation of the token

        Example:
            >>> vec = re_encoder.encode_token("cat")
            >>> vec.shape
            torch.Size([10000])
        """
        return self._codebook.encode(token)

    def encode_sequence(self, tokens: List[str]) -> torch.Tensor:
        """
        Encode token sequence with positional permutation.

        For each token at position i:
            permuted = permute(encode(token), i)
        Result = bundle(all permuted)

        This preserves word order: "cat eats fish" != "fish eats cat"

        Args:
            tokens: List of token strings in order

        Returns:
            Single hypervector representing the ordered sequence

        Example:
            >>> vec = re_encoder.encode_sequence(["the", "cat", "sat"])
            >>> vec.shape
            torch.Size([10000])
        """
        if not tokens:
            raise ValueError("Cannot encode empty sequence")

        if len(tokens) == 1:
            return self.encode_token(tokens[0])

        # Encode each token with positional permutation
        permuted_vectors = []
        for position, token in enumerate(tokens):
            token_vec = self.encode_token(token)
            permuted = self._ops.permute(token_vec, position)
            permuted_vectors.append(permuted)

        # Bundle all permuted vectors
        return self._ops.bundle(*permuted_vectors)

    def encode_partial_sentence(
        self,
        tokens_so_far: List[str],
        new_token: str,
    ) -> torch.Tensor:
        """
        Incrementally encode growing sentence.

        Combines existing tokens with new token and returns the
        full HDC representation. Useful during generation when
        building output token-by-token.

        Args:
            tokens_so_far: Already generated tokens
            new_token: New token to add

        Returns:
            HDC vector representing full partial sentence

        Example:
            >>> partial = re_encoder.encode_partial_sentence(["the", "cat"], "sat")
            >>> # Equivalent to: encode_sequence(["the", "cat", "sat"])
        """
        all_tokens = tokens_so_far + [new_token]
        return self.encode_sequence(all_tokens)

    def encode_with_roles(
        self,
        subject: str,
        verb: str,
        obj: str,
    ) -> torch.Tensor:
        """
        Encode S-V-O triple with role binding.

        Creates the same structure as the Resonator output:
        (subject ⊗ R_subj) ⊕ (verb ⊗ R_verb) ⊕ (object ⊗ R_obj)

        This is useful for direct comparison with target tensors.

        Args:
            subject: Subject word
            verb: Verb word
            obj: Object word

        Returns:
            Bundled HDC vector with role bindings

        Example:
            >>> vec = re_encoder.encode_with_roles("cat", "eats", "fish")
            >>> # Can now compare with target tensor
        """
        s_vec = self.encode_token(subject)
        v_vec = self.encode_token(verb)
        o_vec = self.encode_token(obj)

        r_subj = self._codebook.get_role("SUBJECT")
        r_verb = self._codebook.get_role("VERB")
        r_obj = self._codebook.get_role("OBJECT")

        return self._ops.bundle(
            self._ops.bind(s_vec, r_subj),
            self._ops.bind(v_vec, r_verb),
            self._ops.bind(o_vec, r_obj),
        )

    def __repr__(self) -> str:
        return f"ReEncoder(codebook={self._codebook})"

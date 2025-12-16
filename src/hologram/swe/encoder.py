"""
CodeEncoder: Encode code elements as holographic vectors.

Follows the ObjectEncoder pattern from hologram.arc.encoder:
- Uses role binding for structured encoding
- Pre-encodes vocabulary for fast lookup
- Supports both patch encoding and issue encoding
"""

from typing import List, Tuple, Optional
import hashlib
import torch

from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.core.operations import Operations
from hologram.swe.types import CodePatch, OPERATIONS, LOCATION_TYPES


class CodeEncoder:
    """
    Encode code elements as holographic vectors.

    Structure matches TransformationResonator factorization:
        patch_vec = bind(operation_vec, role_operation) +
                    bind(file_vec, role_file) +
                    bind(location_vec, role_location)

    Attributes:
        _fractal_space: FractalSpace for content hashing
        _codebook: Codebook for vocabulary encoding
    """

    def __init__(
        self,
        fractal_space: FractalSpace,
        codebook: Codebook,
    ):
        """
        Initialize encoder.

        Args:
            fractal_space: FractalSpace for content encoding
            codebook: Codebook for vocabulary
        """
        self._fractal_space = fractal_space
        self._codebook = codebook
        self._ops = Operations

        # Pre-encode role vectors (matches TransformationResonator naming)
        self._role_operation = self._codebook.encode("__ROLE_ACTION__")  # Reuse ACTION role
        self._role_file = self._codebook.encode("__ROLE_TARGET__")       # Reuse TARGET role
        self._role_location = self._codebook.encode("__ROLE_MODIFIER__") # Reuse MODIFIER role

        # Pre-encode operation vocabulary
        self._operation_vectors = {
            op: self._codebook.encode(f"operation_{op}")
            for op in OPERATIONS
        }

        # Pre-encode location type vocabulary
        self._location_vectors = {
            loc: self._codebook.encode(f"location_{loc}")
            for loc in LOCATION_TYPES
        }

        # Dynamic file vocabulary (grows during training)
        self._file_vectors: dict[str, torch.Tensor] = {}

    def encode_patch(self, patch: CodePatch) -> torch.Tensor:
        """
        Encode a code patch as holographic vector.

        Structure:
            bind(operation, role_op) + bind(file, role_file) + bind(location, role_loc)

        Args:
            patch: CodePatch to encode

        Returns:
            HDC vector representing the patch
        """
        # Operation vector
        op_vec = self._operation_vectors.get(
            patch.operation,
            self._codebook.encode(f"operation_{patch.operation}")
        )

        # File vector (hash-based for unseen files)
        file_vec = self._get_or_create_file_vector(patch.file)

        # Location vector (combine type + specific location)
        loc_type = self._infer_location_type(patch.location)
        loc_type_vec = self._location_vectors.get(loc_type, self._location_vectors["line_number"])
        loc_content_vec = self._encode_location_content(patch.location)
        loc_vec = self._ops.bind(loc_type_vec, loc_content_vec)

        # Combine with role binding (matches TransformationResonator structure)
        observation = self._ops.bundle(
            self._ops.bind(op_vec, self._role_operation),
            self._ops.bind(file_vec, self._role_file),
            self._ops.bind(loc_vec, self._role_location),
        )

        return observation

    def encode_issue(self, issue_text: str) -> torch.Tensor:
        """
        Encode issue text as holographic vector.

        Extracts key terms and bundles their encodings.

        Args:
            issue_text: Natural language issue description

        Returns:
            HDC vector representing the issue
        """
        # Extract key terms (simple tokenization)
        words = issue_text.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w.isalnum()][:20]

        if not key_terms:
            return self._codebook.encode("empty_issue")

        # Encode each term and bundle
        term_vecs = [self._codebook.encode(f"term_{t}") for t in key_terms]
        return self._ops.bundle(*term_vecs)

    def _get_or_create_file_vector(self, filepath: str) -> torch.Tensor:
        """Get or create vector for a file path."""
        if filepath not in self._file_vectors:
            # Use hash-based encoding for new files
            seed = int(hashlib.md5(filepath.encode()).hexdigest()[:8], 16) % (2**31)
            self._file_vectors[filepath] = self._fractal_space.random_vector(seed)
        return self._file_vectors[filepath]

    def _infer_location_type(self, location: str) -> str:
        """Infer location type from location string."""
        if location.isdigit():
            return "line_number"
        elif location.startswith("def ") or "(" in location:
            return "function_name"
        elif location.startswith("class "):
            return "class_name"
        elif location == "module" or location == "top":
            return "module_level"
        elif location == "imports":
            return "after_import"
        return "line_number"

    def _encode_location_content(self, location: str) -> torch.Tensor:
        """Encode the specific location content."""
        return self._codebook.encode(f"loc_{location}")

    def get_operation_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """Get operation vocabulary for resonator."""
        names = list(self._operation_vectors.keys())
        vectors = torch.stack([self._operation_vectors[n] for n in names])
        return names, vectors

    def get_file_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """Get file vocabulary for resonator."""
        if not self._file_vectors:
            return ["unknown"], torch.stack([self._codebook.encode("unknown_file")])
        names = list(self._file_vectors.keys())
        vectors = torch.stack([self._file_vectors[n] for n in names])
        return names, vectors

    def get_location_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """Get location vocabulary for resonator."""
        names = list(self._location_vectors.keys())
        vectors = torch.stack([self._location_vectors[n] for n in names])
        return names, vectors

    def register_file(self, filepath: str) -> None:
        """Register a file in the vocabulary."""
        self._get_or_create_file_vector(filepath)

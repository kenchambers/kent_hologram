"""
ObjectEncoder: Encode ARC objects as holographic vectors using fractal DNA.

Each object is encoded as:
    object_vec = bind(shape_dna, bind(color_vec, size_vec))

Where:
- shape_dna: Fractal DNA from shape hash (64d → 10,000d)
- color_vec: Codebook vector for color
- size_vec: Codebook vector for size category

The fractal DNA ensures shape information is recoverable from fragments,
enabling robust pattern matching even with noise.
"""

from typing import List, Optional, Tuple
import torch

from hologram.arc.types import Object, Color, ACTIONS, TARGETS, MODIFIERS
from hologram.core.fractal import FractalSpace
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations


class ObjectEncoder:
    """
    Encode ARC objects as holographic vectors.

    Uses fractal DNA for shapes and codebook vectors for properties.
    This enables pattern matching via cosine similarity while maintaining
    the no-hallucination guarantee (only known shapes/colors can be decoded).

    Attributes:
        _fractal_space: FractalSpace for shape DNA encoding
        _codebook: Codebook for color/size/transformation vocabulary

    Example:
        >>> from hologram.core.vector_space import VectorSpace
        >>> space = VectorSpace(dimensions=10000)
        >>> fractal = FractalSpace(dimensions=10000)
        >>> codebook = Codebook(space)
        >>> encoder = ObjectEncoder(fractal, codebook)
        >>> vec = encoder.encode_object(obj)
        >>> vec.shape
        torch.Size([10000])
    """

    # Size categories for discretization
    SIZE_CATEGORIES = ["tiny", "small", "medium", "large", "huge"]

    def __init__(
        self,
        fractal_space: FractalSpace,
        codebook: Codebook,
    ):
        """
        Initialize encoder.

        Args:
            fractal_space: FractalSpace for shape encoding
            codebook: Codebook for vocabulary encoding
        """
        self._fractal_space = fractal_space
        self._codebook = codebook
        self._ops = Operations

        # Pre-encode color vocabulary
        self._color_vectors = {
            color: self._codebook.encode(f"color_{color.name.lower()}")
            for color in Color
        }

        # Pre-encode size vocabulary
        self._size_vectors = {
            size: self._codebook.encode(f"size_{size}")
            for size in self.SIZE_CATEGORIES
        }

        # Pre-encode transformation vocabulary
        self._action_vectors = {
            action: self._codebook.encode(f"action_{action}")
            for action in ACTIONS
        }
        self._target_vectors = {
            target: self._codebook.encode(f"target_{target}")
            for target in TARGETS
        }
        self._modifier_vectors = {
            modifier: self._codebook.encode(f"modifier_{modifier}")
            for modifier in MODIFIERS
        }

    def encode_object(self, obj: Object) -> torch.Tensor:
        """
        Encode object as holographic vector.

        Structure: bind(shape_dna, bind(color_vec, size_vec))

        Args:
            obj: Object to encode

        Returns:
            10,000-dim holographic vector
        """
        # Shape DNA from shape hash
        shape_dna = self._fractal_space.random_vector(obj.shape_hash())

        # Color vector
        color_vec = self._color_vectors[obj.color]

        # Size category
        size_category = self._categorize_size(obj.size)
        size_vec = self._size_vectors[size_category]

        # Combine: bind(shape, bind(color, size))
        color_size = self._ops.bind(color_vec, size_vec)
        object_vec = self._ops.bind(shape_dna, color_size)

        return object_vec

    def encode_position_delta(
        self,
        delta_row: int,
        delta_col: int,
    ) -> torch.Tensor:
        """
        Encode position change as vector.

        Args:
            delta_row: Change in row (-1, 0, 1, etc.)
            delta_col: Change in column

        Returns:
            Position delta vector
        """
        # Discretize to direction
        if delta_row < 0:
            row_dir = "up"
        elif delta_row > 0:
            row_dir = "down"
        else:
            row_dir = "none"

        if delta_col < 0:
            col_dir = "left"
        elif delta_col > 0:
            col_dir = "right"
        else:
            col_dir = "none"

        row_vec = self._codebook.encode(f"delta_row_{row_dir}")
        col_vec = self._codebook.encode(f"delta_col_{col_dir}")

        return self._ops.bind(row_vec, col_vec)

    def encode_color_change(
        self,
        old_color: Color,
        new_color: Color,
    ) -> torch.Tensor:
        """
        Encode color change as vector.

        Args:
            old_color: Original color
            new_color: New color

        Returns:
            Color change vector
        """
        old_vec = self._color_vectors[old_color]
        new_vec = self._color_vectors[new_color]

        # Use bind to create unique color change representation
        return self._ops.bind(old_vec, new_vec)

    def encode_transformation_observation(
        self,
        input_obj: Optional[Object],
        output_obj: Optional[Object],
    ) -> torch.Tensor:
        """
        Encode observed transformation between input and output objects.

        IMPORTANT: The observation must be structured to align with how
        the TransformationResonator factorizes it. We encode as:

            observation = bind(action_role, action) + bind(target_role, target) + bind(modifier_role, modifier)

        Where action/target/modifier are inferred from the object changes.

        Args:
            input_obj: Object from input grid (or None if created)
            output_obj: Object from output grid (or None if deleted)

        Returns:
            Transformation observation vector
        """
        if input_obj is None and output_obj is None:
            raise ValueError("At least one object must be provided")

        # Get role vectors (same as Resonator uses)
        role_action = self._codebook.encode("__ROLE_ACTION__")
        role_target = self._codebook.encode("__ROLE_TARGET__")
        role_modifier = self._codebook.encode("__ROLE_MODIFIER__")

        if input_obj is None:
            # Object was created
            action_vec = self._action_vectors["copy"]
            target_vec = self._target_vectors["all_objects"]
            modifier_vec = self._modifier_vectors["none"]

        elif output_obj is None:
            # Object was deleted
            action_vec = self._action_vectors["delete"]
            target_vec = self._infer_target(input_obj)
            modifier_vec = self._modifier_vectors["none"]

        else:
            # Both objects exist - infer the transformation
            action_vec, modifier_vec = self._infer_action_and_modifier(input_obj, output_obj)
            target_vec = self._infer_target(input_obj)

        # Structure observation same way Resonator expects to factorize it
        observation = self._ops.bundle(
            self._ops.bind(action_vec, role_action),
            self._ops.bind(target_vec, role_target),
            self._ops.bind(modifier_vec, role_modifier),
        )

        return observation

    def _infer_action_and_modifier(
        self,
        input_obj: Object,
        output_obj: Object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Infer action and modifier from object changes.

        Returns:
            Tuple of (action_vector, modifier_vector)
        """
        # Check for position change
        in_center = input_obj.bbox.center
        out_center = output_obj.bbox.center
        delta_row = int(round(out_center[0] - in_center[0]))
        delta_col = int(round(out_center[1] - in_center[1]))

        # Check for color change
        color_changed = input_obj.color != output_obj.color

        # Check for shape change
        shape_changed = input_obj.mask != output_obj.mask

        # Prioritize: translate > recolor > rotate > identity
        if delta_row != 0 or delta_col != 0:
            # Translation detected
            action_vec = self._action_vectors["translate"]

            # Determine direction modifier
            if delta_row < 0 and delta_col == 0:
                modifier_vec = self._modifier_vectors["up"]
            elif delta_row > 0 and delta_col == 0:
                modifier_vec = self._modifier_vectors["down"]
            elif delta_col < 0 and delta_row == 0:
                modifier_vec = self._modifier_vectors["left"]
            elif delta_col > 0 and delta_row == 0:
                modifier_vec = self._modifier_vectors["right"]
            else:
                # Diagonal - use closest cardinal
                if abs(delta_row) > abs(delta_col):
                    modifier_vec = self._modifier_vectors["up" if delta_row < 0 else "down"]
                else:
                    modifier_vec = self._modifier_vectors["left" if delta_col < 0 else "right"]

        elif color_changed:
            # Recolor detected
            action_vec = self._action_vectors["recolor"]

            # Determine color modifier
            new_color = output_obj.color.name.lower()
            modifier_key = f"to_{new_color}"
            if modifier_key in self._modifier_vectors:
                modifier_vec = self._modifier_vectors[modifier_key]
            else:
                modifier_vec = self._modifier_vectors["none"]

        elif shape_changed:
            # Shape changed - check for rotation/flip
            # Simple heuristic: if dimensions swapped, likely rotation
            in_h, in_w = input_obj.bbox.height, input_obj.bbox.width
            out_h, out_w = output_obj.bbox.height, output_obj.bbox.width

            if (in_h, in_w) == (out_w, out_h):
                action_vec = self._action_vectors["rotate"]
                modifier_vec = self._modifier_vectors["90_degrees"]
            else:
                action_vec = self._action_vectors["flip"]
                modifier_vec = self._modifier_vectors["horizontal"]

        else:
            # No change detected
            action_vec = self._action_vectors["identity"]
            modifier_vec = self._modifier_vectors["none"]

        return action_vec, modifier_vec

    def _infer_target(self, obj: Object) -> torch.Tensor:
        """
        Infer target from object properties.

        Returns:
            Target vector
        """
        # Use color-based targeting if object has distinctive color
        color_targets = {
            Color.RED: "red",
            Color.BLUE: "blue",
            Color.GREEN: "green",
            Color.YELLOW: "yellow",
        }

        if obj.color in color_targets:
            return self._target_vectors[color_targets[obj.color]]

        # Default to all_objects
        return self._target_vectors["all_objects"]

    def get_action_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """
        Get action vocabulary for resonator.

        Returns:
            Tuple of (action names, stacked action vectors)
        """
        names = list(self._action_vectors.keys())
        vectors = torch.stack([self._action_vectors[n] for n in names])
        return names, vectors

    def get_target_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """
        Get target vocabulary for resonator.

        Returns:
            Tuple of (target names, stacked target vectors)
        """
        names = list(self._target_vectors.keys())
        vectors = torch.stack([self._target_vectors[n] for n in names])
        return names, vectors

    def get_modifier_vocabulary(self) -> Tuple[List[str], torch.Tensor]:
        """
        Get modifier vocabulary for resonator.

        Returns:
            Tuple of (modifier names, stacked modifier vectors)
        """
        names = list(self._modifier_vectors.keys())
        vectors = torch.stack([self._modifier_vectors[n] for n in names])
        return names, vectors

    def _categorize_size(self, pixel_count: int) -> str:
        """
        Categorize object size into discrete buckets.

        Args:
            pixel_count: Number of pixels in object

        Returns:
            Size category string
        """
        if pixel_count <= 2:
            return "tiny"
        elif pixel_count <= 9:
            return "small"
        elif pixel_count <= 25:
            return "medium"
        elif pixel_count <= 100:
            return "large"
        else:
            return "huge"

    def encode_task_signature(
        self,
        training_observations: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Create a signature vector for an ARC task.

        This is used for skill memory lookup - similar tasks
        should have similar signatures.

        Args:
            training_observations: Bundled observations from training pairs

        Returns:
            Task signature vector (64-dim DNA seed for efficiency)
        """
        # Bundle all observations
        if not training_observations:
            return self._codebook.encode("empty_task")

        bundled = self._ops.bundle(*training_observations)

        # Extract DNA from first block for compact representation
        # This loses some info but enables O(1) lookup
        dna = self._fractal_space.extract_block(bundled, block_index=0)

        # Expand back to full dimensions for consistency
        return self._fractal_space._expand(dna)

    # ========== Grid-Level Encoding (NEW) ==========

    def encode_grid_transformation(
        self,
        input_grid_shape: tuple,
        output_grid_shape: tuple,
    ) -> Optional[torch.Tensor]:
        """
        Encode grid-level transformation observation (e.g., tiling).

        This detects dimensional changes between input and output grids
        and encodes them as transformation observations.

        Args:
            input_grid_shape: (height, width) of input grid
            output_grid_shape: (height, width) of output grid

        Returns:
            Transformation observation vector, or None if no grid transform
        """
        in_h, in_w = input_grid_shape
        out_h, out_w = output_grid_shape

        # Check for tiling (output is multiple of input)
        tiling_info = self.detect_tiling(input_grid_shape, output_grid_shape)
        if tiling_info is not None:
            n_rows, n_cols = tiling_info
            return self._encode_tiling_observation(n_rows, n_cols)

        # No grid-level transformation detected
        return None

    def detect_tiling(
        self,
        input_shape: tuple,
        output_shape: tuple,
    ) -> Optional[tuple]:
        """
        Detect if output is a tiled version of input.

        Only detects SELF-REFERENTIAL tiling where n_rows == n_cols == input_height
        (for square inputs) indicating the input pattern determines tile placement.

        Args:
            input_shape: (height, width) of input grid
            output_shape: (height, width) of output grid

        Returns:
            (n_rows, n_cols) if self-referential tiling detected, None otherwise
        """
        in_h, in_w = input_shape
        out_h, out_w = output_shape

        # Check if output dimensions are multiples of input
        if out_h % in_h == 0 and out_w % in_w == 0:
            n_rows = out_h // in_h
            n_cols = out_w // in_w

            # Only detect SELF-REFERENTIAL tiling:
            # - Dimensions must actually increase (not identity)
            # - Input must be square (n×n)
            # - Tiling factor must equal input dimensions (n×n → n*n × n*n)
            # - Output must be within ARC limits (≤30×30)
            dimensions_increased = n_rows > 1 or n_cols > 1
            is_square_input = in_h == in_w
            is_self_referential = n_rows == in_h and n_cols == in_w
            output_valid = out_h <= 30 and out_w <= 30

            if dimensions_increased and is_square_input and is_self_referential and output_valid:
                return (n_rows, n_cols)

        return None

    def _encode_tiling_observation(
        self,
        n_rows: int,
        n_cols: int,
        is_self_referential: bool = True,
    ) -> torch.Tensor:
        """
        Encode a tiling transformation observation.

        Args:
            n_rows: Number of tile rows
            n_cols: Number of tile columns
            is_self_referential: If True, use by_pattern (tile based on input pattern)
                                If False, use tile_NxN (uniform tiling)

        Returns:
            Transformation observation vector for tiling
        """
        # Get role vectors
        role_action = self._codebook.encode("__ROLE_ACTION__")
        role_target = self._codebook.encode("__ROLE_TARGET__")
        role_modifier = self._codebook.encode("__ROLE_MODIFIER__")

        # Action is always "tile" for grid tiling
        action_vec = self._action_vectors["tile"]

        # Target is all_objects (the whole grid)
        target_vec = self._target_vectors["all_objects"]

        # Modifier depends on tiling type
        # For ARC tasks, self-referential tiling (by_pattern) is most common
        # where the input pattern determines both content AND placement
        if is_self_referential:
            # Self-referential: tile based on input colored cells
            modifier_vec = self._modifier_vectors["by_pattern"]
        else:
            # Uniform tiling: tile_NxN
            if n_rows == n_cols:
                modifier_key = f"tile_{n_rows}x{n_cols}"
                if modifier_key in self._modifier_vectors:
                    modifier_vec = self._modifier_vectors[modifier_key]
                else:
                    modifier_vec = self._modifier_vectors["by_pattern"]
            else:
                modifier_vec = self._modifier_vectors["by_pattern"]

        # Structure observation same way Resonator expects
        observation = self._ops.bundle(
            self._ops.bind(action_vec, role_action),
            self._ops.bind(target_vec, role_target),
            self._ops.bind(modifier_vec, role_modifier),
        )

        return observation

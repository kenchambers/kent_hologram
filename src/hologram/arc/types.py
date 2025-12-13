"""
ARC Types: Core data structures for ARC-AGI-2 tasks.

Defines Grid, Object, BoundingBox, and ARCTask types used throughout
the holographic ARC solver.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Set, FrozenSet
import numpy as np


class Color(IntEnum):
    """ARC color palette (0-9)."""
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    CYAN = 8
    MAROON = 9

    @classmethod
    def from_int(cls, value: int) -> "Color":
        """Convert int to Color, clamping to valid range."""
        return cls(max(0, min(9, value)))


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box for an object."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> Tuple[float, float]:
        """Center of bounding box (row, col)."""
        return (
            (self.min_row + self.max_row) / 2,
            (self.min_col + self.max_col) / 2
        )


@dataclass(frozen=True)
class Object:
    """
    A detected object in an ARC grid.

    Objects are connected components of non-background pixels.
    Each object has a color, mask, and bounding box.

    Attributes:
        color: Primary color of the object (most common non-background color)
        pixels: Frozenset of (row, col) coordinates
        bbox: Bounding box enclosing the object
        mask: Normalized binary mask (relative to bbox)
    """
    color: Color
    pixels: FrozenSet[Tuple[int, int]]
    bbox: BoundingBox
    mask: Tuple[Tuple[int, ...], ...]  # Tuple of tuples for hashability

    @property
    def size(self) -> int:
        """Number of pixels in the object."""
        return len(self.pixels)

    @property
    def normalized_mask(self) -> np.ndarray:
        """Return mask as numpy array."""
        return np.array(self.mask, dtype=np.uint8)

    def translate(self, delta_row: int, delta_col: int) -> "Object":
        """Return object translated by (delta_row, delta_col)."""
        new_pixels = frozenset(
            (r + delta_row, c + delta_col) for r, c in self.pixels
        )
        new_bbox = BoundingBox(
            min_row=self.bbox.min_row + delta_row,
            min_col=self.bbox.min_col + delta_col,
            max_row=self.bbox.max_row + delta_row,
            max_col=self.bbox.max_col + delta_col,
        )
        return Object(
            color=self.color,
            pixels=new_pixels,
            bbox=new_bbox,
            mask=self.mask,
        )

    def recolor(self, new_color: Color) -> "Object":
        """Return object with different color."""
        return Object(
            color=new_color,
            pixels=self.pixels,
            bbox=self.bbox,
            mask=self.mask,
        )

    def shape_hash(self) -> int:
        """
        Hash of the normalized mask shape.

        Used to generate deterministic fractal DNA for this shape.
        Invariant to position and color.
        """
        return hash(self.mask)


@dataclass
class Grid:
    """
    An ARC grid (2D array of colors).

    Grids are 1x1 to 30x30 arrays of colors (0-9).
    """
    data: np.ndarray

    def __post_init__(self):
        """Validate grid dimensions and values."""
        if self.data.ndim != 2:
            raise ValueError(f"Grid must be 2D, got {self.data.ndim}D")
        if self.data.shape[0] < 1 or self.data.shape[0] > 30:
            raise ValueError(f"Grid height must be 1-30, got {self.data.shape[0]}")
        if self.data.shape[1] < 1 or self.data.shape[1] > 30:
            raise ValueError(f"Grid width must be 1-30, got {self.data.shape[1]}")
        # Ensure integer type
        self.data = self.data.astype(np.int8)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def __getitem__(self, key) -> int:
        return int(self.data[key])

    def __eq__(self, other: "Grid") -> bool:
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def copy(self) -> "Grid":
        return Grid(self.data.copy())

    @classmethod
    def from_list(cls, data: List[List[int]]) -> "Grid":
        """Create Grid from nested list."""
        return cls(np.array(data, dtype=np.int8))

    @classmethod
    def empty(cls, height: int, width: int, fill: int = 0) -> "Grid":
        """Create empty grid filled with a color."""
        return cls(np.full((height, width), fill, dtype=np.int8))

    def to_list(self) -> List[List[int]]:
        """Convert to nested list (for JSON serialization)."""
        return self.data.tolist()

    def render_objects(
        self,
        objects: List[Object],
        background: int = 0,
        preserve_dimensions: bool = True,
    ) -> "Grid":
        """
        Render objects onto a new grid.

        Args:
            objects: List of objects to render
            background: Background color
            preserve_dimensions: If True, use original grid dimensions.
                               If False, fit grid to objects (for rotate/scale).

        Returns:
            New grid with objects rendered
        """
        if not objects:
            return Grid.empty(self.height, self.width, fill=background)

        # Compute required grid size from object pixels
        all_pixels = set()
        for obj in objects:
            all_pixels.update(obj.pixels)

        if not all_pixels:
            return Grid.empty(self.height, self.width, fill=background)

        max_row = max(r for r, c in all_pixels)
        max_col = max(c for r, c in all_pixels)
        min_row = min(r for r, c in all_pixels)
        min_col = min(c for r, c in all_pixels)

        if preserve_dimensions:
            # Use original grid dimensions
            height = self.height
            width = self.width
            row_offset = 0
            col_offset = 0
        else:
            # Fit grid to objects (for dimension-changing transforms)
            row_offset = -min_row
            col_offset = -min_col
            height = max_row - min_row + 1
            width = max_col - min_col + 1

        result = Grid.empty(height, width, fill=background)
        for obj in objects:
            for r, c in obj.pixels:
                nr, nc = r + row_offset, c + col_offset
                if 0 <= nr < height and 0 <= nc < width:
                    result.data[nr, nc] = obj.color
        return result


@dataclass
class TrainingPair:
    """A single training example (input → output)."""
    input: Grid
    output: Grid


@dataclass
class ARCTask:
    """
    Complete ARC task with training pairs and test input.

    Attributes:
        task_id: Unique identifier for the task
        training: List of (input, output) training pairs
        test_input: The test input grid to solve
        test_output: Ground truth output (None if unknown)
    """
    task_id: str
    training: List[TrainingPair]
    test_input: Grid
    test_output: Optional[Grid] = None

    @property
    def num_training(self) -> int:
        return len(self.training)

    @classmethod
    def from_dict(cls, data: dict, task_id: str = "unknown") -> "ARCTask":
        """
        Create ARCTask from ARC JSON format.

        Args:
            data: Dictionary with 'train' and 'test' keys
            task_id: Task identifier

        Returns:
            ARCTask instance
        """
        training = []
        for pair in data.get("train", []):
            training.append(TrainingPair(
                input=Grid.from_list(pair["input"]),
                output=Grid.from_list(pair["output"]),
            ))

        test_data = data.get("test", [{}])[0]
        test_input = Grid.from_list(test_data.get("input", [[0]]))
        test_output = None
        if "output" in test_data:
            test_output = Grid.from_list(test_data["output"])

        return cls(
            task_id=task_id,
            training=training,
            test_input=test_input,
            test_output=test_output,
        )


# Transformation vocabulary constants
# Phase 3: Conservative vocabulary expansion (Dr. Nexus approved)

ACTIONS = [
    "identity",     # No change
    "rotate",       # Rotate object
    "translate",    # Move object
    "recolor",      # Change color
    "flip",         # Mirror object
    "scale",        # Resize object
    "delete",       # Remove object
    "copy",         # Duplicate object
    # Grid-level operations
    "tile",         # Tile input pattern across output grid
    "expand",       # Expand grid dimensions
    "fill",         # Fill region with pattern
    # Phase 3: Counting operations
    "duplicate",    # Copy object N times in specified direction
    "enumerate",    # Assign sequential colors to objects
]

TARGETS = [
    "all_objects",  # Apply to all
    "largest",      # Largest by pixel count
    "smallest",     # Smallest by pixel count
    "red",          # Objects of color red
    "blue",         # Objects of color blue
    "green",        # Objects of color green
    "yellow",       # Objects of color yellow
    "by_position",  # Select by position
    "by_color",     # Select by color match
    # Phase 3: New targets for spatial operations
    "background",       # Non-object pixels
    "bounding_box",     # Rectangle containing all objects
    "between_objects",  # Space between detected objects
    "by_adjacency",     # Objects adjacent to specified target
]

MODIFIERS = [
    "none",         # No modifier
    "90_degrees",   # Rotate 90 CW
    "180_degrees",  # Rotate 180
    "270_degrees",  # Rotate 270 CW (90 CCW)
    "up",           # Translate up
    "down",         # Translate down
    "left",         # Translate left
    "right",        # Translate right
    "to_red",       # Recolor to red
    "to_blue",      # Recolor to blue
    "to_green",     # Recolor to green
    "horizontal",   # Flip horizontally
    "vertical",     # Flip vertically
    "by_2x",        # Scale 2x
    "by_half",      # Scale 0.5x
    # Tiling modifiers
    "tile_2x2",     # Tile into 2x2 grid
    "tile_3x3",     # Tile into 3x3 grid
    "tile_4x4",     # Tile into 4x4 grid
    "by_pattern",   # Tile based on input pattern
    "to_yellow",    # Recolor to yellow
    "to_grey",      # Recolor to grey
    "to_magenta",   # Recolor to magenta
    "to_orange",    # Recolor to orange
    "to_cyan",      # Recolor to cyan
    # Phase 3: Counting modifiers (bounded)
    "count_1",      # Count = 1
    "count_2",      # Count = 2
    "count_3",      # Count = 3
    "count_4",      # Count = 4
    "count_5",      # Count = 5
    "count_n",      # Count = input dimension (self-referential)
    # Phase 3: Symmetry modifiers
    "diagonal_main",    # Mirror across top-left to bottom-right
    "diagonal_anti",    # Mirror across top-right to bottom-left
    "point_center",     # 180-degree rotation around center (point symmetry)
    # Phase 3: Fill pattern modifiers
    "fill_solid",       # Solid color fill
    "fill_checkerboard", # Alternating pattern
    "fill_border",      # Fill only border pixels
    "fill_interior",    # Fill only interior pixels
]

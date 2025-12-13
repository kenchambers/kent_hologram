"""
TransformationExecutor: Apply (ACTION, TARGET, MODIFIER) to grids.

Takes a transformation specification from the Resonator and applies
it to a test input grid to produce the output.

This is the "execution" step that converts symbolic transformations
back into concrete grid operations.
"""

from typing import List, Optional, Callable
import numpy as np

from hologram.arc.types import Grid, Object, Color, BoundingBox
from hologram.arc.detector import ObjectDetector


class TransformationExecutor:
    """
    Execute transformations on ARC grids.

    Given an (ACTION, TARGET, MODIFIER) specification and a list of objects,
    applies the transformation to produce a new grid.

    Example:
        >>> executor = TransformationExecutor()
        >>> objects = detector.detect(input_grid)
        >>> output = executor.execute("rotate", "all_objects", "90_degrees", objects, input_grid)
    """

    def __init__(self, detector: Optional[ObjectDetector] = None):
        """
        Initialize executor.

        Args:
            detector: ObjectDetector for re-detecting objects after transforms
        """
        self._detector = detector or ObjectDetector()

        # Map action names to methods
        self._actions = {
            "identity": self._identity,
            "rotate": self._rotate,
            "translate": self._translate,
            "recolor": self._recolor,
            "flip": self._flip,
            "scale": self._scale,
            "delete": self._delete,
            "copy": self._copy,
            # Grid-level operations
            "tile": self._tile,
            "expand": self._expand,
            "fill": self._fill,
            # Phase 3: Counting operations
            "duplicate": self._duplicate,
            "enumerate": self._enumerate,
        }

        # Grid-level actions that need special handling
        self._grid_level_actions = {"tile", "expand", "fill"}

        # Map target names to filter functions
        # For color targets, fall back to all objects if no match (generalization)
        self._target_filters = {
            "all_objects": lambda objs: objs,
            "largest": lambda objs: [max(objs, key=lambda o: o.size)] if objs else [],
            "smallest": lambda objs: [min(objs, key=lambda o: o.size)] if objs else [],
            "red": lambda objs: [o for o in objs if o.color == Color.RED] or objs,
            "blue": lambda objs: [o for o in objs if o.color == Color.BLUE] or objs,
            "green": lambda objs: [o for o in objs if o.color == Color.GREEN] or objs,
            "yellow": lambda objs: [o for o in objs if o.color == Color.YELLOW] or objs,
            "by_position": lambda objs: objs,  # Would need additional context
            "by_color": lambda objs: objs,     # Would need additional context
            # Phase 3: New spatial targets
            "background": lambda objs: [],    # No objects (background pixels)
            "bounding_box": lambda objs: objs, # All objects (bbox computed in action)
            "between_objects": lambda objs: objs,  # Computed spatially
            "by_adjacency": lambda objs: objs,     # Computed relationally
        }

    def execute(
        self,
        action: str,
        target: str,
        modifier: str,
        objects: List[Object],
        grid: Grid,
    ) -> Grid:
        """
        Execute transformation on grid.

        Args:
            action: Action to perform (e.g., "rotate")
            target: Target selector (e.g., "all_objects")
            modifier: Modifier (e.g., "90_degrees")
            objects: Detected objects in grid
            grid: Original grid

        Returns:
            Transformed grid
        """
        # Handle grid-level actions specially (they operate on the whole grid)
        if action in self._grid_level_actions:
            action_fn = self._actions.get(action, self._identity)
            return action_fn(objects, modifier, grid)

        # Filter objects by target
        filter_fn = self._target_filters.get(target, lambda x: x)
        target_objects = filter_fn(objects)

        # Get non-target objects (keep unchanged)
        target_set = set(id(o) for o in target_objects)
        other_objects = [o for o in objects if id(o) not in target_set]

        # Apply action to target objects
        action_fn = self._actions.get(action, self._identity)
        transformed_objects = action_fn(target_objects, modifier, grid)

        # Combine with unchanged objects
        all_objects = other_objects + transformed_objects

        # Determine if this is a dimension-changing transform
        preserve_dims = action not in ("rotate", "scale")

        # Render to new grid
        return grid.render_objects(
            all_objects,
            background=0,
            preserve_dimensions=preserve_dims,
        )

    def _identity(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """No transformation - return objects unchanged."""
        return objects

    def _rotate(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Rotate objects by specified amount."""
        rotations = {
            "90_degrees": 1,
            "180_degrees": 2,
            "270_degrees": 3,
            "none": 0,
        }
        k = rotations.get(modifier, 0)

        result = []
        for obj in objects:
            rotated = self._rotate_object(obj, k)
            result.append(rotated)
        return result

    def _rotate_object(self, obj: Object, k: int) -> Object:
        """
        Rotate object by k * 90 degrees clockwise.

        Args:
            obj: Object to rotate
            k: Number of 90-degree rotations (0-3)

        Returns:
            Rotated object
        """
        if k == 0:
            return obj

        # Normalize k to 0-3
        k = k % 4

        # Rotate mask using numpy
        mask_arr = np.array(obj.mask, dtype=np.uint8)
        # np.rot90 rotates counter-clockwise by default, so use -k for clockwise
        rotated_mask = np.rot90(mask_arr, k=-k)

        # Get the top-left position of the original object
        min_row, min_col = obj.bbox.min_row, obj.bbox.min_col

        # Compute new pixels from rotated mask
        new_pixels = set()
        for r in range(rotated_mask.shape[0]):
            for c in range(rotated_mask.shape[1]):
                if rotated_mask[r, c] == 1:
                    new_pixels.add((min_row + r, min_col + c))

        if not new_pixels:
            return obj

        # Compute new bounding box
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = BoundingBox(
            min_row=min(rows),
            min_col=min(cols),
            max_row=max(rows),
            max_col=max(cols),
        )

        # Convert mask to tuples
        new_mask = tuple(tuple(int(x) for x in row) for row in rotated_mask.tolist())

        return Object(
            color=obj.color,
            pixels=frozenset(new_pixels),
            bbox=new_bbox,
            mask=new_mask,
        )

    def _translate(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Translate objects in specified direction."""
        deltas = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "none": (0, 0),
        }
        delta = deltas.get(modifier, (0, 0))

        return [obj.translate(delta[0], delta[1]) for obj in objects]

    def _recolor(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Recolor objects to specified color."""
        color_map = {
            "to_red": Color.RED,
            "to_blue": Color.BLUE,
            "to_green": Color.GREEN,
            "to_yellow": Color.YELLOW,
            "none": None,
        }
        new_color = color_map.get(modifier)

        if new_color is None:
            return objects

        return [obj.recolor(new_color) for obj in objects]

    def _flip(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Flip objects horizontally, vertically, or diagonally."""
        result = []
        for obj in objects:
            # Phase 3: Handle diagonal symmetry modifiers
            if modifier == "diagonal_main":
                flipped = self._flip_diagonal(obj, anti=False)
            elif modifier == "diagonal_anti":
                flipped = self._flip_diagonal(obj, anti=True)
            elif modifier == "point_center":
                # 180-degree rotation is equivalent to point symmetry
                flipped = self._rotate_object(obj, k=2)
            else:
                flipped = self._flip_object(obj, modifier)
            result.append(flipped)
        return result

    def _flip_object(self, obj: Object, modifier: str) -> Object:
        """
        Flip object horizontally or vertically.

        Args:
            obj: Object to flip
            modifier: "horizontal" or "vertical"

        Returns:
            Flipped object
        """
        center = obj.bbox.center
        new_pixels = set()

        for r, c in obj.pixels:
            if modifier == "horizontal":
                # Flip across vertical axis (mirror left-right)
                new_c = int(round(2 * center[1] - c))
                new_pixels.add((r, new_c))
            elif modifier == "vertical":
                # Flip across horizontal axis (mirror up-down)
                new_r = int(round(2 * center[0] - r))
                new_pixels.add((new_r, c))
            else:
                new_pixels.add((r, c))

        # Compute new bounding box
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = BoundingBox(
            min_row=min(rows),
            min_col=min(cols),
            max_row=max(rows),
            max_col=max(cols),
        )

        # Flip mask
        mask_arr = np.array(obj.mask, dtype=np.uint8)
        if modifier == "horizontal":
            flipped_mask = np.fliplr(mask_arr)
        elif modifier == "vertical":
            flipped_mask = np.flipud(mask_arr)
        else:
            flipped_mask = mask_arr

        new_mask = tuple(tuple(row) for row in flipped_mask.tolist())

        return Object(
            color=obj.color,
            pixels=frozenset(new_pixels),
            bbox=new_bbox,
            mask=new_mask,
        )

    def _scale(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Scale objects by specified factor."""
        scale_map = {
            "by_2x": 2,
            "by_half": 0.5,
            "none": 1,
        }
        factor = scale_map.get(modifier, 1)

        if factor == 1:
            return objects

        result = []
        for obj in objects:
            scaled = self._scale_object(obj, factor)
            result.append(scaled)
        return result

    def _scale_object(self, obj: Object, factor: float) -> Object:
        """
        Scale object by factor.

        Args:
            obj: Object to scale
            factor: Scale factor (2 = double, 0.5 = half)

        Returns:
            Scaled object
        """
        center = obj.bbox.center
        new_pixels = set()

        if factor >= 1:
            # Upscale: multiply each pixel
            scale = int(factor)
            for r, c in obj.pixels:
                # Scale relative to center
                dr, dc = r - center[0], c - center[1]
                for di in range(scale):
                    for dj in range(scale):
                        nr = int(round(center[0] + dr * scale + di))
                        nc = int(round(center[1] + dc * scale + dj))
                        new_pixels.add((nr, nc))
        else:
            # Downscale: sample pixels
            for r, c in obj.pixels:
                dr, dc = r - center[0], c - center[1]
                nr = int(round(center[0] + dr * factor))
                nc = int(round(center[1] + dc * factor))
                new_pixels.add((nr, nc))

        if not new_pixels:
            # If scaling made object disappear, keep original
            return obj

        # Compute new bounding box
        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = BoundingBox(
            min_row=min(rows),
            min_col=min(cols),
            max_row=max(rows),
            max_col=max(cols),
        )

        # Recompute mask for new bbox
        mask = []
        for r in range(new_bbox.min_row, new_bbox.max_row + 1):
            row_mask = []
            for c in range(new_bbox.min_col, new_bbox.max_col + 1):
                row_mask.append(1 if (r, c) in new_pixels else 0)
            mask.append(tuple(row_mask))

        return Object(
            color=obj.color,
            pixels=frozenset(new_pixels),
            bbox=new_bbox,
            mask=tuple(mask),
        )

    def _delete(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Delete objects (return empty list)."""
        return []

    def _copy(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """Copy objects (duplicate them)."""
        # For now, just return the objects unchanged
        # A full implementation would place copies based on modifier
        return objects

    # ========== Grid-Level Operations (NEW) ==========

    def _tile(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> Grid:
        """
        Tile the input grid into a larger output grid.

        This is the key operation for ARC tasks like 007bbfb7 where a 3x3
        input becomes a 9x9 output by tiling based on colored cell positions.

        Args:
            objects: Detected objects (unused for grid-level tiling)
            modifier: Tiling modifier (tile_2x2, tile_3x3, tile_4x4, by_pattern)
            grid: Input grid to tile

        Returns:
            Tiled output grid
        """
        # Determine tile dimensions
        tile_dims = {
            "tile_2x2": (2, 2),
            "tile_3x3": (3, 3),
            "tile_4x4": (4, 4),
            "by_pattern": None,  # Use input pattern to determine tiling
            "none": (1, 1),
        }

        dims = tile_dims.get(modifier)

        if modifier == "by_pattern" or dims is None:
            # Self-referential tiling: tile based on colored cells in input
            return self._tile_by_pattern(grid)

        n_rows, n_cols = dims
        out_height = grid.height * n_rows
        out_width = grid.width * n_cols

        # Create output grid
        result = Grid.empty(out_height, out_width, fill=0)

        # Tile the input pattern
        for tr in range(n_rows):
            for tc in range(n_cols):
                # Copy input grid to this tile position
                for r in range(grid.height):
                    for c in range(grid.width):
                        out_r = tr * grid.height + r
                        out_c = tc * grid.width + c
                        result.data[out_r, out_c] = grid.data[r, c]

        return result

    def _tile_by_pattern(self, grid: Grid) -> Grid:
        """
        Tile based on colored cells in input (self-referential tiling).

        For ARC task 007bbfb7: input pattern determines BOTH the content
        AND the placement of tiles. Where input has colored cells, place
        a copy of the input pattern.

        Args:
            grid: Input grid (acts as both pattern and placement rule)

        Returns:
            Tiled output grid
        """
        # Output is input.height × input.height (square tiling)
        n = grid.height  # Assume square input
        out_height = grid.height * n
        out_width = grid.width * n

        # ARC grids are limited to 30x30 - if output would exceed, fall back
        if out_height > 30 or out_width > 30:
            # Cannot tile this grid within ARC limits, return identity
            return grid.copy()

        result = Grid.empty(out_height, out_width, fill=0)

        # For each cell in input, if colored, place the input pattern there
        for tr in range(grid.height):
            for tc in range(grid.width):
                if grid.data[tr, tc] != 0:  # Colored cell
                    # Place input pattern at this tile position
                    for r in range(grid.height):
                        for c in range(grid.width):
                            out_r = tr * grid.height + r
                            out_c = tc * grid.width + c
                            if out_r < out_height and out_c < out_width:
                                result.data[out_r, out_c] = grid.data[r, c]

        return result

    def _expand(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> Grid:
        """
        Expand grid dimensions by a factor.

        Args:
            objects: Detected objects (unused)
            modifier: Expansion factor (by_2x, tile_2x2, etc.)
            grid: Input grid

        Returns:
            Expanded output grid
        """
        factors = {
            "by_2x": 2,
            "tile_2x2": 2,
            "tile_3x3": 3,
            "tile_4x4": 4,
            "by_half": 1,  # No expansion for shrinking
            "none": 1,
        }
        factor = factors.get(modifier, 1)

        if factor == 1:
            return grid.copy()

        out_height = grid.height * factor
        out_width = grid.width * factor

        result = Grid.empty(out_height, out_width, fill=0)

        # Scale each pixel
        for r in range(grid.height):
            for c in range(grid.width):
                color = grid.data[r, c]
                for dr in range(factor):
                    for dc in range(factor):
                        result.data[r * factor + dr, c * factor + dc] = color

        return result

    def _fill(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> Grid:
        """
        Fill regions with a pattern or color.

        Args:
            objects: Detected objects (regions to fill)
            modifier: Fill modifier
            grid: Input grid

        Returns:
            Grid with filled regions
        """
        # Simple fill: just recolor all non-background cells
        color_map = {
            "to_red": Color.RED,
            "to_blue": Color.BLUE,
            "to_green": Color.GREEN,
            "to_yellow": Color.YELLOW,
            "to_grey": Color.GREY,
            "to_magenta": Color.MAGENTA,
            "to_orange": Color.ORANGE,
            "to_cyan": Color.CYAN,
            "none": None,
        }

        new_color = color_map.get(modifier)
        if new_color is None:
            return grid.copy()

        result = grid.copy()

        # Handle fill pattern modifiers
        if modifier == "fill_border":
            return self._fill_border(grid, new_color or Color.RED)
        elif modifier == "fill_interior":
            return self._fill_interior(grid, new_color or Color.RED)
        elif modifier == "fill_checkerboard":
            return self._fill_checkerboard(grid)

        # Default: fill all non-background cells
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r, c] != 0:
                    result.data[r, c] = new_color

        return result

    # ========== Phase 3: Fill Pattern Helpers ==========

    def _fill_border(self, grid: Grid, color: Color) -> Grid:
        """Fill only border pixels of objects."""
        result = grid.copy()
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r, c] != 0:
                    # Check if this is a border pixel (adjacent to background)
                    is_border = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.height and 0 <= nc < grid.width:
                            if grid.data[nr, nc] == 0:
                                is_border = True
                                break
                        else:
                            is_border = True  # Edge of grid counts as border
                            break
                    if is_border:
                        result.data[r, c] = color
        return result

    def _fill_interior(self, grid: Grid, color: Color) -> Grid:
        """Fill only interior pixels of objects."""
        result = grid.copy()
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r, c] != 0:
                    # Check if this is an interior pixel (not adjacent to background)
                    is_interior = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.height and 0 <= nc < grid.width:
                            if grid.data[nr, nc] == 0:
                                is_interior = False
                                break
                        else:
                            is_interior = False
                            break
                    if is_interior:
                        result.data[r, c] = color
        return result

    def _fill_checkerboard(self, grid: Grid) -> Grid:
        """Fill with alternating pattern."""
        result = grid.copy()
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r, c] != 0:
                    # Checkerboard: alternate based on position parity
                    if (r + c) % 2 == 0:
                        result.data[r, c] = Color.BLACK
        return result

    # ========== Phase 3: Counting Operations ==========

    def _duplicate(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """
        Duplicate objects N times in specified direction.

        Args:
            objects: Objects to duplicate
            modifier: Count and direction (count_N or direction)
            grid: Input grid

        Returns:
            List of original + duplicated objects
        """
        # Parse count from modifier
        count_map = {
            "count_1": 1,
            "count_2": 2,
            "count_3": 3,
            "count_4": 4,
            "count_5": 5,
            "count_n": grid.height,  # Self-referential: use input dimension
        }
        count = count_map.get(modifier, 1)

        # Default direction is right
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        direction = direction_map.get(modifier, (0, 1))

        result = list(objects)  # Keep originals
        for obj in objects:
            # Compute spacing based on object size
            spacing_r = (obj.bbox.height + 1) if direction[0] != 0 else 0
            spacing_c = (obj.bbox.width + 1) if direction[1] != 0 else 0

            for i in range(1, count + 1):
                delta_r = direction[0] * spacing_r * i
                delta_c = direction[1] * spacing_c * i
                copy = obj.translate(delta_r, delta_c)
                result.append(copy)

        return result

    def _enumerate(
        self,
        objects: List[Object],
        modifier: str,
        grid: Grid,
    ) -> List[Object]:
        """
        Assign sequential colors to objects.

        Args:
            objects: Objects to enumerate
            modifier: Starting color or pattern
            grid: Input grid

        Returns:
            Objects with sequential colors assigned
        """
        # Color sequence for enumeration
        colors = [
            Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW,
            Color.GREY, Color.MAGENTA, Color.ORANGE, Color.CYAN,
        ]

        result = []
        for i, obj in enumerate(objects):
            new_color = colors[i % len(colors)]
            result.append(obj.recolor(new_color))

        return result

    # ========== Phase 3: Enhanced Flip with Symmetry ==========

    def _flip_diagonal(self, obj: Object, anti: bool = False) -> Object:
        """
        Flip object across diagonal.

        Args:
            obj: Object to flip
            anti: If True, use anti-diagonal (top-right to bottom-left)

        Returns:
            Flipped object
        """
        mask_arr = np.array(obj.mask, dtype=np.uint8)

        if anti:
            # Anti-diagonal: flip both, then transpose
            flipped_mask = np.fliplr(np.flipud(mask_arr)).T
        else:
            # Main diagonal: just transpose
            flipped_mask = mask_arr.T

        # Compute new pixels
        min_row, min_col = obj.bbox.min_row, obj.bbox.min_col
        new_pixels = set()
        for r in range(flipped_mask.shape[0]):
            for c in range(flipped_mask.shape[1]):
                if flipped_mask[r, c] == 1:
                    new_pixels.add((min_row + r, min_col + c))

        if not new_pixels:
            return obj

        rows = [r for r, c in new_pixels]
        cols = [c for r, c in new_pixels]
        new_bbox = BoundingBox(
            min_row=min(rows),
            min_col=min(cols),
            max_row=max(rows),
            max_col=max(cols),
        )

        new_mask = tuple(tuple(int(x) for x in row) for row in flipped_mask.tolist())

        return Object(
            color=obj.color,
            pixels=frozenset(new_pixels),
            bbox=new_bbox,
            mask=new_mask,
        )

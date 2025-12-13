"""
ObjectDetector: Extract objects from ARC grids via flood-fill.

Uses classical image processing (no ML) to segment grids into
connected components. Each component becomes an Object with:
- Color (most common non-background color)
- Pixel set (coordinates)
- Bounding box
- Normalized mask (for shape hashing)
"""

from collections import deque
from typing import List, Set, Tuple, Optional
import numpy as np

from hologram.arc.types import Grid, Object, BoundingBox, Color


class ObjectDetector:
    """
    Detect objects in ARC grids using flood-fill segmentation.

    Objects are defined as connected components of non-background pixels.
    Uses 4-connectivity (up, down, left, right neighbors).

    Attributes:
        background_color: Color to treat as background (default: 0/black)
        min_size: Minimum object size in pixels (default: 1)

    Example:
        >>> detector = ObjectDetector()
        >>> grid = Grid.from_list([[0,1,0], [0,1,0], [0,0,0]])
        >>> objects = detector.detect(grid)
        >>> len(objects)
        1
        >>> objects[0].color
        Color.BLUE
    """

    def __init__(
        self,
        background_color: int = 0,
        min_size: int = 1,
        connectivity: int = 4,
    ):
        """
        Initialize object detector.

        Args:
            background_color: Color to treat as background
            min_size: Minimum pixels for valid object
            connectivity: 4 or 8 for neighbor connectivity
        """
        self.background_color = background_color
        self.min_size = min_size
        self.connectivity = connectivity

        # Neighbor offsets based on connectivity
        if connectivity == 4:
            self._neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == 8:
            self._neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        else:
            raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    def detect(self, grid: Grid) -> List[Object]:
        """
        Detect all objects in a grid.

        Args:
            grid: Input grid

        Returns:
            List of detected objects, sorted by size (largest first)
        """
        visited = set()
        objects = []

        for row in range(grid.height):
            for col in range(grid.width):
                if (row, col) in visited:
                    continue

                pixel_value = grid[row, col]
                if pixel_value == self.background_color:
                    visited.add((row, col))
                    continue

                # Found unvisited non-background pixel - flood fill
                component = self._flood_fill(grid, row, col, visited)

                if len(component) >= self.min_size:
                    obj = self._pixels_to_object(component, grid)
                    objects.append(obj)

        # Sort by size (largest first)
        objects.sort(key=lambda o: o.size, reverse=True)
        return objects

    def detect_by_color(self, grid: Grid, color: int) -> List[Object]:
        """
        Detect objects of a specific color.

        Args:
            grid: Input grid
            color: Target color

        Returns:
            List of objects with the specified color
        """
        all_objects = self.detect(grid)
        return [obj for obj in all_objects if obj.color == color]

    def _flood_fill(
        self,
        grid: Grid,
        start_row: int,
        start_col: int,
        visited: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        """
        Flood fill to find connected component.

        Uses BFS to find all connected non-background pixels.

        Args:
            grid: Input grid
            start_row: Starting row
            start_col: Starting column
            visited: Set of already visited pixels (modified in place)

        Returns:
            Set of (row, col) coordinates in the component
        """
        component = set()
        queue = deque([(start_row, start_col)])

        while queue:
            row, col = queue.popleft()

            if (row, col) in visited:
                continue

            if not (0 <= row < grid.height and 0 <= col < grid.width):
                continue

            pixel_value = grid[row, col]
            if pixel_value == self.background_color:
                visited.add((row, col))
                continue

            visited.add((row, col))
            component.add((row, col))

            # Add neighbors to queue
            for dr, dc in self._neighbors:
                nr, nc = row + dr, col + dc
                if (nr, nc) not in visited:
                    queue.append((nr, nc))

        return component

    def _pixels_to_object(
        self,
        pixels: Set[Tuple[int, int]],
        grid: Grid,
    ) -> Object:
        """
        Convert pixel set to Object.

        Args:
            pixels: Set of (row, col) coordinates
            grid: Original grid (for color lookup)

        Returns:
            Object instance
        """
        # Find bounding box
        rows = [r for r, c in pixels]
        cols = [c for r, c in pixels]
        bbox = BoundingBox(
            min_row=min(rows),
            min_col=min(cols),
            max_row=max(rows),
            max_col=max(cols),
        )

        # Determine dominant color
        color_counts = {}
        for r, c in pixels:
            color = grid[r, c]
            color_counts[color] = color_counts.get(color, 0) + 1

        dominant_color = max(color_counts, key=color_counts.get)

        # Create normalized mask (relative to bbox)
        mask = []
        for r in range(bbox.min_row, bbox.max_row + 1):
            row_mask = []
            for c in range(bbox.min_col, bbox.max_col + 1):
                row_mask.append(1 if (r, c) in pixels else 0)
            mask.append(tuple(row_mask))

        return Object(
            color=Color.from_int(dominant_color),
            pixels=frozenset(pixels),
            bbox=bbox,
            mask=tuple(mask),
        )

    def match_objects(
        self,
        input_objects: List[Object],
        output_objects: List[Object],
    ) -> List[Tuple[Optional[Object], Optional[Object]]]:
        """
        Match objects between input and output grids.

        Uses multiple heuristics to find corresponding objects:
        1. Same shape and color (perfect match)
        2. Same color, different shape (transformation)
        3. Same pixel count, different shape (transformation)
        4. Closest position (fallback)

        Args:
            input_objects: Objects from input grid
            output_objects: Objects from output grid

        Returns:
            List of matched pairs
        """
        matches = []
        used_output = set()

        # First pass: match by same color (robust to transformations)
        for in_obj in input_objects:
            best_match = None
            best_score = -1

            for i, out_obj in enumerate(output_objects):
                if i in used_output:
                    continue

                score = 0

                # Color match is strong signal
                if in_obj.color == out_obj.color:
                    score += 100

                # Shape match
                if in_obj.shape_hash() == out_obj.shape_hash():
                    score += 50

                # Similar size
                size_ratio = min(in_obj.size, out_obj.size) / max(in_obj.size, out_obj.size)
                score += size_ratio * 30

                # Position proximity (inverse distance)
                in_center = in_obj.bbox.center
                out_center = out_obj.bbox.center
                distance = (
                    (in_center[0] - out_center[0]) ** 2 +
                    (in_center[1] - out_center[1]) ** 2
                ) ** 0.5
                score += max(0, 20 - distance * 2)

                if score > best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score > 50:  # Must have some signal
                matches.append((in_obj, output_objects[best_match]))
                used_output.add(best_match)
            else:
                # Input object may have been deleted
                matches.append((in_obj, None))

        # Second pass: match remaining by position/size if we have same count
        if len(input_objects) == len(output_objects):
            unmatched_inputs = [in_obj for in_obj, out_obj in matches if out_obj is None]
            unmatched_outputs = [out_obj for i, out_obj in enumerate(output_objects) if i not in used_output]

            if len(unmatched_inputs) == len(unmatched_outputs) == 1:
                # Only one unmatched each - they must correspond
                matches = [(in_obj, out_obj) for in_obj, out_obj in matches if out_obj is not None]
                matches.append((unmatched_inputs[0], unmatched_outputs[0]))
                used_output.add(output_objects.index(unmatched_outputs[0]))

        # Add unmatched output objects (newly created)
        for i, out_obj in enumerate(output_objects):
            if i not in used_output:
                matches.append((None, out_obj))

        return matches

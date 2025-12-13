"""
Object Detection Tests for ARC-AGI-2 Holographic Reasoning.

Tests the ObjectDetector component which extracts objects from ARC grids
via flood-fill segmentation.

Run with: uv run pytest tests/arc/test_object_detection.py -v
"""

import pytest
import numpy as np

from hologram.arc.types import Grid, Object, BoundingBox, Color
from hologram.arc.detector import ObjectDetector


class TestObjectDetectorBasic:
    """Basic object detection tests."""

    def test_detect_single_object(self):
        """Detect a single vertical line object."""
        grid = Grid.from_list([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        assert objects[0].color == Color.BLUE
        assert objects[0].size == 2
        assert (1, 1) in objects[0].pixels
        assert (0, 1) in objects[0].pixels

    def test_detect_multiple_objects(self):
        """Detect multiple separate objects."""
        grid = Grid.from_list([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 4
        colors = {obj.color for obj in objects}
        assert colors == {Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW}

    def test_detect_connected_component(self):
        """Detect L-shaped connected component as single object."""
        grid = Grid.from_list([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        assert objects[0].size == 5
        assert objects[0].color == Color.BLUE

    def test_empty_grid(self):
        """Empty grid should produce no objects."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 0

    def test_full_grid(self):
        """Grid full of one color is one object."""
        grid = Grid.from_list([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        assert objects[0].size == 9


class TestObjectDetectorBoundingBox:
    """Tests for bounding box calculation."""

    def test_bounding_box_single_pixel(self):
        """Single pixel should have 1x1 bounding box."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        bbox = objects[0].bbox
        assert bbox.height == 1
        assert bbox.width == 1
        assert bbox.min_row == 1
        assert bbox.min_col == 1

    def test_bounding_box_rectangle(self):
        """Rectangular object should have correct bounding box."""
        grid = Grid.from_list([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        bbox = objects[0].bbox
        assert bbox.height == 3
        assert bbox.width == 2
        assert bbox.min_row == 1
        assert bbox.min_col == 1
        assert bbox.max_row == 3
        assert bbox.max_col == 2


class TestObjectDetectorMask:
    """Tests for normalized mask generation."""

    def test_mask_solid_rectangle(self):
        """Solid rectangle should have all-ones mask."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        mask = objects[0].normalized_mask
        assert mask.shape == (2, 2)
        assert np.all(mask == 1)

    def test_mask_l_shape(self):
        """L-shape should have correct mask with zeros."""
        grid = Grid.from_list([
            [1, 0],
            [1, 0],
            [1, 1],
        ])
        detector = ObjectDetector()
        objects = detector.detect(grid)

        assert len(objects) == 1
        mask = objects[0].normalized_mask
        expected = np.array([
            [1, 0],
            [1, 0],
            [1, 1],
        ], dtype=np.uint8)
        assert np.array_equal(mask, expected)

    def test_shape_hash_invariant_to_position(self):
        """Same shape at different positions should have same hash."""
        grid1 = Grid.from_list([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ])
        grid2 = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ])
        detector = ObjectDetector()
        
        obj1 = detector.detect(grid1)[0]
        obj2 = detector.detect(grid2)[0]

        assert obj1.shape_hash() == obj2.shape_hash()


class TestObjectDetectorConnectivity:
    """Tests for 4-connectivity vs 8-connectivity."""

    def test_4_connectivity_diagonal_separate(self):
        """Diagonal pixels should be separate objects with 4-connectivity."""
        grid = Grid.from_list([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        detector = ObjectDetector(connectivity=4)
        objects = detector.detect(grid)

        assert len(objects) == 3

    def test_8_connectivity_diagonal_connected(self):
        """Diagonal pixels should be one object with 8-connectivity."""
        grid = Grid.from_list([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        detector = ObjectDetector(connectivity=8)
        objects = detector.detect(grid)

        assert len(objects) == 1
        assert objects[0].size == 3


class TestObjectDetectorColorFiltering:
    """Tests for color-based object detection."""

    def test_detect_by_color(self):
        """Should filter objects by specific color."""
        grid = Grid.from_list([
            [1, 0, 2],
            [1, 0, 2],
            [3, 0, 3],
        ])
        detector = ObjectDetector()
        
        red_objects = detector.detect_by_color(grid, color=2)
        assert len(red_objects) == 1
        assert red_objects[0].color == Color.RED

        blue_objects = detector.detect_by_color(grid, color=1)
        assert len(blue_objects) == 1
        assert blue_objects[0].color == Color.BLUE


class TestObjectMatching:
    """Tests for matching objects between grids."""

    def test_match_same_color(self):
        """Objects with same color should match."""
        input_grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [1, 0, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)

        assert len(matches) == 1
        in_obj, out_obj = matches[0]
        assert in_obj is not None
        assert out_obj is not None
        assert in_obj.color == out_obj.color

    def test_match_deleted_object(self):
        """Deleted object should have None as output match."""
        input_grid = Grid.from_list([
            [1, 0, 2],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [1, 0, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)

        # One object matched, one deleted
        has_none = any(out_obj is None for _, out_obj in matches)
        assert has_none

    def test_match_created_object(self):
        """Created object should have None as input match."""
        input_grid = Grid.from_list([
            [1, 0, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [1, 0, 2],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)

        # Check for created object (input is None)
        has_created = any(in_obj is None for in_obj, _ in matches)
        assert has_created


class TestObjectOperations:
    """Tests for Object methods like translate and recolor."""

    def test_translate_object(self):
        """Object should translate correctly."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        obj = detector.detect(grid)[0]

        translated = obj.translate(delta_row=-1, delta_col=1)

        assert (0, 2) in translated.pixels
        assert translated.bbox.min_row == 0
        assert translated.bbox.min_col == 2

    def test_recolor_object(self):
        """Object should recolor correctly."""
        grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
        ])
        detector = ObjectDetector()
        obj = detector.detect(grid)[0]

        recolored = obj.recolor(Color.RED)

        assert recolored.color == Color.RED
        assert recolored.pixels == obj.pixels  # Same position

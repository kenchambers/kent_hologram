#!/usr/bin/env python3
"""
ARC Holographic Reasoning Demo.

Demonstrates the novel HDC-native approach to ARC-AGI-2:
1. Object detection via flood-fill
2. Transformation observation encoding
3. Resonator factorization into (ACTION, TARGET, MODIFIER)
4. Execution and verification

Run with: uv run python scripts/arc_demo.py
"""

import sys
sys.path.insert(0, "src")

from hologram.arc import (
    HolographicARCSolver,
    create_simple_task,
    Grid,
    ObjectDetector,
)


def print_grid(grid: Grid, name: str = "Grid"):
    """Pretty-print a grid."""
    print(f"\n{name}:")
    colors = "⬛🟦🟥🟩🟨⬜🟪🟧🔵🟫"  # 0-9 as colored squares
    for row in range(grid.height):
        line = ""
        for col in range(grid.width):
            val = grid[row, col]
            line += colors[val] if val < len(colors) else str(val)
        print(f"  {line}")


def demo_translate_up():
    """Demo: Objects translate upward."""
    print("\n" + "="*60)
    print("DEMO 1: Translate Up")
    print("="*60)

    # Training examples: object moves up by 1 row
    task = create_simple_task(
        train_inputs=[
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 2, 0]],
        ],
        train_outputs=[
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 2, 0],
             [0, 0, 0]],
        ],
        test_input=[
            [0, 0, 0],
            [0, 0, 0],
            [3, 0, 0]],
        test_output=[
            [0, 0, 0],
            [3, 0, 0],
            [0, 0, 0]],
        task_id="translate_up",
    )

    for i, pair in enumerate(task.training):
        print_grid(pair.input, f"Train {i+1} Input")
        print_grid(pair.output, f"Train {i+1} Output")

    print_grid(task.test_input, "Test Input")
    print_grid(task.test_output, "Expected Output")

    solver = HolographicARCSolver()
    result = solver.solve(task)

    print(f"\nResult: {result.message}")
    print(f"Confidence: {result.confidence:.3f}")

    if result.output:
        print_grid(result.output, "Predicted Output")
        correct = result.output == task.test_output
        print(f"Correct: {'✓' if correct else '✗'}")
    else:
        print("Solver refused (confidence too low)")

    return result


def demo_recolor():
    """Demo: Objects change color."""
    print("\n" + "="*60)
    print("DEMO 2: Recolor to Blue")
    print("="*60)

    task = create_simple_task(
        train_inputs=[
            [[0, 0, 0],
             [0, 2, 0],
             [0, 0, 0]],
            [[3, 0],
             [0, 0]],
        ],
        train_outputs=[
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[1, 0],
             [0, 0]],
        ],
        test_input=[
            [0, 4, 0],
            [0, 0, 0]],
        test_output=[
            [0, 1, 0],
            [0, 0, 0]],
        task_id="recolor_to_blue",
    )

    for i, pair in enumerate(task.training):
        print_grid(pair.input, f"Train {i+1} Input")
        print_grid(pair.output, f"Train {i+1} Output")

    print_grid(task.test_input, "Test Input")
    print_grid(task.test_output, "Expected Output")

    solver = HolographicARCSolver()
    result = solver.solve(task)

    print(f"\nResult: {result.message}")
    print(f"Confidence: {result.confidence:.3f}")

    if result.output:
        print_grid(result.output, "Predicted Output")
        correct = result.output == task.test_output
        print(f"Correct: {'✓' if correct else '✗'}")

    return result


def demo_rotate():
    """Demo: Objects rotate 90 degrees."""
    print("\n" + "="*60)
    print("DEMO 3: Rotate 90 Degrees")
    print("="*60)

    # L-shape rotates 90 degrees
    task = create_simple_task(
        train_inputs=[
            [[1, 0],
             [1, 0],
             [1, 1]],
        ],
        train_outputs=[
            [[1, 1, 1],
             [1, 0, 0]],
        ],
        test_input=[
            [2, 0],
            [2, 0],
            [2, 2]],
        test_output=[
            [2, 2, 2],
            [2, 0, 0]],
        task_id="rotate_90",
    )

    for i, pair in enumerate(task.training):
        print_grid(pair.input, f"Train {i+1} Input")
        print_grid(pair.output, f"Train {i+1} Output")

    print_grid(task.test_input, "Test Input")
    print_grid(task.test_output, "Expected Output")

    solver = HolographicARCSolver()
    result = solver.solve(task)

    print(f"\nResult: {result.message}")
    print(f"Confidence: {result.confidence:.3f}")

    if result.transformation:
        print(f"Transformation: {result.transformation}")

    if result.output:
        print_grid(result.output, "Predicted Output")
        correct = result.output == task.test_output
        print(f"Correct: {'✓' if correct else '✗'}")

    return result


def demo_object_detection():
    """Demo: Object detection capabilities."""
    print("\n" + "="*60)
    print("DEMO: Object Detection")
    print("="*60)

    grid = Grid.from_list([
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0, 2, 2],
        [3, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0],
    ])

    print_grid(grid, "Input Grid")

    detector = ObjectDetector()
    objects = detector.detect(grid)

    print(f"\nDetected {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: color={obj.color.name}, "
              f"size={obj.size}, bbox={obj.bbox}")


def demo_resonator_details():
    """Demo: Show resonator internals."""
    print("\n" + "="*60)
    print("DEMO: Resonator Factorization Details")
    print("="*60)

    task = create_simple_task(
        train_inputs=[
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
        ],
        train_outputs=[
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
        ],
        test_input=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 2, 0]],
        task_id="resonator_demo",
    )

    solver = HolographicARCSolver()
    result = solver.solve(task)

    if result.transformation:
        t = result.transformation
        print(f"\nResonator Result:")
        print(f"  ACTION:   {t.action:15} (confidence: {t.confidence.get('action', 0):.3f})")
        print(f"  TARGET:   {t.target:15} (confidence: {t.confidence.get('target', 0):.3f})")
        print(f"  MODIFIER: {t.modifier:15} (confidence: {t.confidence.get('modifier', 0):.3f})")
        print(f"  Converged: {t.converged} in {t.iterations} iterations")
        print(f"  Overall confidence: {result.confidence:.3f}")


def main():
    """Run all demos."""
    print("="*60)
    print("ARC-AGI-2 HOLOGRAPHIC REASONING DEMO")
    print("="*60)
    print("\nThis demo shows the novel HDC-native approach to ARC:")
    print("- Objects encoded as fractal DNA vectors")
    print("- Transformations factorized via Resonator into (ACTION, TARGET, MODIFIER)")
    print("- No hallucination: can only output vocabulary items")

    # Run demos
    demo_object_detection()
    demo_resonator_details()
    demo_translate_up()
    demo_recolor()
    demo_rotate()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey insights:")
    print("- The Resonator factorizes observations holographically")
    print("- Confidence indicates how well the observation matches vocabulary")
    print("- Low confidence → refuse (saves second attempt)")
    print("- This is a novel approach - expect ~10-25% on ARC-AGI-2")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic usage example for Hologram holographic memory system.

Demonstrates:
1. Creating a holographic memory
2. Storing facts
3. Querying facts
4. Sequence encoding
"""

from hologram.container import HologramContainer
from hologram.core.similarity import Similarity


def main():
    print("=" * 60)
    print("Hologram: Bentov-Style Holographic Memory Demo")
    print("=" * 60)
    print()

    # Initialize the system
    print("Initializing 10,000-dimensional holographic memory...")
    container = HologramContainer(dimensions=10000)
    fact_store = container.create_fact_store()
    sequence_encoder = container.create_sequence_encoder()
    print(f"✓ {container}")
    print()

    # Store some facts
    print("Storing facts...")
    facts = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Spain", "capital", "Madrid"),
        ("Italy", "capital", "Rome"),
        ("Earth", "shape", "round"),
        ("Water", "boiling_point", "100C"),
    ]

    for subject, predicate, obj in facts:
        fact_store.add_fact(subject, predicate, obj)
        print(f"  ✓ {subject} --{predicate}--> {obj}")

    print(f"\n{fact_store}")
    print()

    # Query facts
    print("Querying holographic memory...")
    queries = [
        ("France", "capital"),
        ("Germany", "capital"),
        ("Earth", "shape"),
        ("Moon", "color"),  # Unknown fact - should have low confidence
    ]

    for subject, predicate in queries:
        answer, confidence = fact_store.query(subject, predicate)
        status = "✓" if confidence > 0.6 else "?"
        print(f"  {status} {subject} --{predicate}--> {answer} (confidence: {confidence:.3f})")

    print()

    # Demonstrate sequence encoding
    print("Demonstrating sequence encoding (order matters)...")
    sentence1 = "dog bites man"
    sentence2 = "man bites dog"
    sentence3 = "dog bites man"  # Same as sentence1

    vec1 = sequence_encoder.encode_sentence(sentence1)
    vec2 = sequence_encoder.encode_sentence(sentence2)
    vec3 = sequence_encoder.encode_sentence(sentence3)

    sim_12 = Similarity.cosine(vec1, vec2)
    sim_13 = Similarity.cosine(vec1, vec3)

    print(f"  '{sentence1}' vs '{sentence2}': similarity = {sim_12:.3f}")
    print(f"  '{sentence1}' vs '{sentence3}': similarity = {sim_13:.3f}")
    print(f"  ✓ Different order = low similarity ({sim_12 < 0.5})")
    print(f"  ✓ Same order = high similarity ({sim_13 > 0.9})")
    print()

    # Test graceful degradation
    print("Testing graceful degradation (adding noise)...")
    original = fact_store.query("France", "capital")
    print(f"  Original: {original[0]} (confidence: {original[1]:.3f})")

    # Add noise to memory trace (simulating corruption)
    noisy_trace = fact_store._memory.corrupt(noise_ratio=0.3)
    # Create temporary fact store with noisy memory
    temp_fs = container.create_fact_store()
    temp_fs._memory = noisy_trace
    temp_fs._facts = fact_store._facts
    temp_fs._value_vocab = fact_store._value_vocab

    noisy = temp_fs.query("France", "capital")
    print(f"  With 30% noise: {noisy[0]} (confidence: {noisy[1]:.3f})")

    if noisy[0] == original[0]:
        print(f"  ✓ Answer unchanged (holographic recovery)")
    print(f"  ✓ Confidence degraded: {original[1]:.3f} → {noisy[1]:.3f}")
    print()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

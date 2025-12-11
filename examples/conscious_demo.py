#!/usr/bin/env python3
"""
Conscious Hologram Demo: Demonstrating Fractal Shards + Metacognition

This demo shows two key features:
1. Fractal Shard Recovery: Slice a vector, recover the whole concept
2. Metacognitive Retry: System "thinks harder" when initial query fails
"""

import torch
from hologram.container import HologramContainer
from hologram.core.similarity import Similarity


def demo_fractal_recovery():
    """Demonstrate fractal shard recovery."""
    print("=" * 60)
    print("DEMO 1: Fractal Shard Recovery")
    print("=" * 60)
    
    # Create container with FractalSpace
    container = HologramContainer(dimensions=10000, use_fractal=True)
    codebook = container.codebook
    space = container.vector_space
    
    # Generate a concept vector ("King")
    print("\n1. Generating 'King' vector using FractalSpace...")
    king_vec = codebook.encode("King")
    print(f"   Vector shape: {king_vec.shape}")
    print(f"   Vector norm: {torch.norm(king_vec):.4f}")
    
    # Slice the vector (simulate corruption/transmission loss)
    print("\n2. Slicing vector (simulating corruption)...")
    shard_start = 5000
    shard_end = 5064
    shard = king_vec[shard_start:shard_end]
    print(f"   Shard shape: {shard.shape} (dimensions {shard_start}-{shard_end})")
    
    # Recover DNA from shard
    print("\n3. Recovering DNA from shard...")
    if hasattr(space, 'recover_dna'):
        block_index = shard_start // space.dna_dimensions
        recovered_dna = space.recover_dna(shard, block_index)
        print(f"   Recovered DNA shape: {recovered_dna.shape}")
        
        # Generate original DNA for comparison
        original_seed = container.codebook._hash_to_seed("King")
        original_dna_gen = torch.Generator().manual_seed(original_seed)
        original_dna = torch.randn(space.dna_dimensions, generator=original_dna_gen)
        original_dna = original_dna / torch.norm(original_dna)
        
        # Compare similarity
        similarity = Similarity.cosine(recovered_dna, original_dna)
        print(f"   DNA recovery similarity: {similarity:.4f}")
        
        if similarity > 0.9:
            print("   ✓ SUCCESS: Recovered DNA matches original!")
        else:
            print(f"   ⚠ PARTIAL: Recovered DNA is similar but not identical (expected due to rotation)")
    else:
        print("   ⚠ FractalSpace not active (using standard VectorSpace)")
    
    print()


def demo_metacognitive_retry():
    """Demonstrate metacognitive retry loop."""
    print("=" * 60)
    print("DEMO 2: Metacognitive Retry Loop")
    print("=" * 60)
    
    # Create container with metacognition enabled
    container = HologramContainer(dimensions=10000, use_fractal=True)
    
    # Create chatbot with metacognition
    print("\n1. Creating chatbot with MetacognitiveLoop enabled...")
    chatbot = container.create_persistent_chatbot(
        persist_dir="./data/conscious_demo",
        enable_metacognition=True,
    )
    
    # Teach a fact
    print("\n2. Teaching a fact...")
    chatbot.teach_fact("France", "capital", "Paris")
    print("   ✓ Learned: France --capital--> Paris")
    
    # Query with high confidence (should succeed immediately)
    print("\n3. Querying with high confidence (should succeed)...")
    response1 = chatbot.respond("What is the capital of France?")
    print(f"   Response: {response1}")
    
    if chatbot._metacognitive:
        print(f"   Metacognitive state: {chatbot._metacognitive.state}")
        print(f"   Mood: {chatbot._metacognitive.state.mood.value}")
    
    # Query with ambiguous question (low confidence, should trigger retry)
    print("\n4. Querying with ambiguous question (should trigger metacognitive retry)...")
    response2 = chatbot.respond("What is France?")
    print(f"   Response: {response2}")
    
    if chatbot._metacognitive:
        print(f"   Metacognitive state: {chatbot._metacognitive.state}")
        print(f"   Mood: {chatbot._metacognitive.state.mood.value}")
        print(f"   Confidence history: {chatbot._metacognitive.state.confidence_history[-3:]}")
        trend = chatbot._metacognitive.state.get_confidence_trend()
        print(f"   Confidence trend: {trend:+.4f}")
    
    print()


def demo_combined_system():
    """Demonstrate both features working together."""
    print("=" * 60)
    print("DEMO 3: Combined System (Fractal + Metacognition)")
    print("=" * 60)
    
    container = HologramContainer(dimensions=10000, use_fractal=True)
    chatbot = container.create_persistent_chatbot(
        persist_dir="./data/conscious_demo",
        enable_metacognition=True,
    )
    
    print("\n1. Teaching multiple facts...")
    facts = [
        ("France", "capital", "Paris"),
        ("Japan", "capital", "Tokyo"),
        ("Germany", "capital", "Berlin"),
    ]
    for subject, predicate, obj in facts:
        chatbot.teach_fact(subject, predicate, obj)
        print(f"   ✓ Learned: {subject} --{predicate}--> {obj}")
    
    print("\n2. Testing query with metacognitive retry...")
    response = chatbot.respond("What is the capital of France?")
    print(f"   Response: {response}")
    
    if chatbot._metacognitive:
        print(f"\n3. Metacognitive State Analysis:")
        print(f"   Current mood: {chatbot._metacognitive.state.mood.value}")
        print(f"   Confidence history: {chatbot._metacognitive.state.confidence_history}")
        print(f"   Self-vector norm: {torch.norm(chatbot._metacognitive.state.self_vector):.4f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONSCIOUS HOLOGRAM DEMO")
    print("=" * 60)
    print("\nThis demo showcases:")
    print("  • Fractal Shard Recovery: Robust to corruption")
    print("  • Metacognitive Retry: Self-monitoring and adaptation")
    print()
    
    try:
        demo_fractal_recovery()
        demo_metacognitive_retry()
        demo_combined_system()
        
        print("=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

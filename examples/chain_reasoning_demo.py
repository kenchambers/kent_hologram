"""
Chain Reasoning Demo: Multi-step reasoning via transformer-style attention.

Demonstrates how ChainReasoner performs multi-hop reasoning over indexed facts,
bypassing the superposition noise of bundled memory traces.
"""

from hologram.container import HologramContainer


def main():
    """Demonstrate chain reasoning with multi-hop queries."""
    print("=" * 70)
    print("CHAIN REASONING DEMO")
    print("=" * 70)
    print()

    # Create container and fact stores
    container = HologramContainer(dimensions=10000)

    # Create ChromaDB fact store (indexed, like transformer K-V pairs)
    print("📊 Setting up ChromaDB fact store...")
    chroma_store = container.create_chroma_fact_store(
        persist_dir="./data/chain_demo_facts"
    )

    # Create chain reasoner
    chain_reasoner = container.create_chain_reasoner(chroma_store)

    # Add geography facts for chain reasoning
    print("📚 Teaching geography facts...")
    facts = [
        ("Paris", "country", "France"),
        ("France", "continent", "Europe"),
        ("France", "capital", "Paris"),
        ("France", "population", "67 million"),
        ("Berlin", "country", "Germany"),
        ("Germany", "continent", "Europe"),
        ("Germany", "capital", "Berlin"),
        ("Tokyo", "country", "Japan"),
        ("Japan", "continent", "Asia"),
    ]

    for subject, predicate, obj in facts:
        chroma_store.add_fact(subject, predicate, obj, source="demo")
        print(f"  ✓ {subject} --{predicate}--> {obj}")

    print()
    print("=" * 70)
    print("SINGLE-HOP REASONING")
    print("=" * 70)
    print()

    # Example 1: Single-hop query
    query = "What is the capital of France?"
    print(f"Query: {query}")
    result = chain_reasoner.chain_reason("France", ["capital"])
    if result:
        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.min_confidence:.3f}")
        print(f"Steps: {len(result.steps)}")
        for i, step in enumerate(result.steps, 1):
            print(f"  Step {i}: {step.subject} --{step.predicate}--> {step.answer} (conf={step.confidence:.3f})")
    print()

    print("=" * 70)
    print("MULTI-HOP REASONING (Transformer-Style Attention)")
    print("=" * 70)
    print()

    # Example 2: Two-hop chain reasoning
    query = "What continent is Paris in?"
    print(f"Query: {query}")
    print("Chain: Paris --country--> France --continent--> Europe")
    result = chain_reasoner.chain_reason_with_context("Paris", ["country", "continent"])
    if result:
        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.min_confidence:.3f}")
        print(f"Steps: {len(result.steps)}")
        for i, step in enumerate(result.steps, 1):
            print(f"  Step {i}: {step.subject} --{step.predicate}--> {step.answer} (conf={step.confidence:.3f})")
    print()

    # Example 3: Different starting point, same endpoint
    query = "What continent is Berlin in?"
    print(f"Query: {query}")
    print("Chain: Berlin --country--> Germany --continent--> Europe")
    result = chain_reasoner.chain_reason_with_context("Berlin", ["country", "continent"])
    if result:
        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.min_confidence:.3f}")
        print(f"Steps: {len(result.steps)}")
        for i, step in enumerate(result.steps, 1):
            print(f"  Step {i}: {step.subject} --{step.predicate}--> {step.answer} (conf={step.confidence:.3f})")
    print()

    # Example 4: Chain with refusal (unknown predicate)
    query = "What is the unknown_property of France?"
    print(f"Query: {query}")
    result = chain_reasoner.chain_reason("France", ["unknown_property"])
    if result:
        print(f"Answer: {result.final_answer}")
    else:
        print("Answer: [REFUSED - fact not found]")
        print("(Bounded hallucination: refuses instead of guessing)")
    print()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()
    print("✓ ChainReasoner uses ChromaDB (attention-style) instead of bundled traces")
    print("✓ Each step queries individual facts → no superposition noise")
    print("✓ Hard cleanup between steps → bounded hallucination")
    print("✓ Residual context accumulation → disambiguates multi-step chains")
    print("✓ Refuses when confidence is low → knows what it doesn't know")
    print()
    print("This is the 'chain reasoning' counterpart to bundled MemoryTrace:")
    print("  - MemoryTrace (Hopfield): Single-step, graceful degradation")
    print("  - ChainReasoner (Attention): Multi-step, precise chains")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Code Enhancement Demo: Holographic RAG for Code Generation

This demo shows how the Holographic Code Enhancement system works:
1. Pre-train with general coding concepts (Design Patterns, Algorithms)
2. Index a specific codebase (Project Context)
3. Generate code using dual retrieval (Concepts + Context)
4. Verify generated code against stored facts

This prevents hallucination by grounding the LLM in verified knowledge.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hologram.container import HologramContainer
from hologram.memory.fact_store import FactStore
from hologram.generation.ventriloquist import VentriloquistGenerator


def demo_concept_ingestion():
    """Demo: Pre-train with general coding concepts."""
    print("\n" + "="*70)
    print("STEP 1: Pre-training with General Coding Concepts")
    print("="*70)
    
    # Initialize container
    container = HologramContainer(dimensions=10000)
    concept_store = FactStore(
        space=container._vector_space,
        codebook=container._codebook
    )
    
    # Manually add some design patterns (simulating ingest_concepts.py)
    concepts = [
        ("Singleton", "purpose", "Ensure a class has only one instance"),
        ("Singleton", "type", "Creational Pattern"),
        ("Factory", "purpose", "Create objects without specifying exact class"),
        ("Factory", "type", "Creational Pattern"),
        ("BinarySearch", "time_complexity", "O(log n)"),
        ("BinarySearch", "type", "Search Algorithm"),
    ]
    
    for subject, predicate, obj in concepts:
        concept_store.add_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source="Concept Catalog"
        )
    
    print(f"✓ Loaded {concept_store.fact_count} general coding concepts")
    print(f"  Vocabulary: {concept_store.vocabulary_size} unique terms")
    
    # Test retrieval
    print("\nTesting concept retrieval:")
    answer, conf = concept_store.query("BinarySearch", "time_complexity")
    print(f"  Query: BinarySearch --time_complexity--> ?")
    print(f"  Answer: {answer} (confidence: {conf:.3f})")
    
    return concept_store


def demo_code_indexing():
    """Demo: Index a specific codebase."""
    print("\n" + "="*70)
    print("STEP 2: Indexing Project-Specific Codebase")
    print("="*70)
    
    # Initialize container
    container = HologramContainer(dimensions=10000)
    code_store = FactStore(
        space=container._vector_space,
        codebook=container._codebook
    )
    
    # Manually add some code facts (simulating code_indexer.py)
    code_facts = [
        ("FactStore", "type", "class"),
        ("FactStore", "inherits", "object"),
        ("add_fact", "type", "function"),
        ("add_fact", "signature", "(subject, predicate, obj, source, confidence)"),
        ("add_fact", "returns", "Optional[Fact]"),
        ("add_fact", "calls", "encode"),
        ("query", "type", "function"),
        ("query", "signature", "(subject, predicate)"),
        ("query", "returns", "tuple[str, float]"),
    ]
    
    for subject, predicate, obj in code_facts:
        code_store.add_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source="hologram/memory/fact_store.py"
        )
    
    print(f"✓ Indexed {code_store.fact_count} project-specific facts")
    print(f"  Vocabulary: {code_store.vocabulary_size} unique terms")
    
    # Test retrieval
    print("\nTesting code index retrieval:")
    answer, conf = code_store.query("add_fact", "signature")
    print(f"  Query: add_fact --signature--> ?")
    print(f"  Answer: {answer} (confidence: {conf:.3f})")
    
    # Test reverse query
    print("\nTesting reverse query (call graph):")
    subject, conf = code_store.query_subject("calls", "encode")
    print(f"  Query: ? --calls--> encode")
    print(f"  Answer: {subject} (confidence: {conf:.3f})")
    
    return code_store


def demo_dual_retrieval(concept_store, code_store):
    """Demo: Dual retrieval for code generation."""
    print("\n" + "="*70)
    print("STEP 3: Dual Retrieval (Concepts + Context)")
    print("="*70)
    
    # Initialize Ventriloquist (requires NOVITA_API_KEY)
    try:
        ventriloquist = VentriloquistGenerator()
    except ValueError as e:
        print(f"⚠ Skipping generation demo: {e}")
        print("  Set NOVITA_API_KEY in .env to enable code generation")
        return
    
    # Test query
    query = "implement factory pattern for User class"
    
    print(f"Query: {query}")
    print("\nRetrieving context...")
    
    concept_facts, project_facts = ventriloquist.retrieve_dual_context(
        query=query,
        fact_store=code_store,
        concept_store=concept_store,
        max_facts=3
    )
    
    print("\nGeneral Concepts Retrieved:")
    for fact in concept_facts:
        print(f"  - {fact}")
    
    print("\nProject Context Retrieved:")
    for fact in project_facts:
        print(f"  - {fact}")
    
    print("\n✓ Dual retrieval combines general knowledge + project context")


def demo_code_verification(concept_store, code_store):
    """Demo: Verify generated code."""
    print("\n" + "="*70)
    print("STEP 4: Code Verification (Hallucination Prevention)")
    print("="*70)
    
    # Initialize Ventriloquist
    try:
        ventriloquist = VentriloquistGenerator()
    except ValueError as e:
        print(f"⚠ Skipping verification demo: {e}")
        return
    
    # Test code with known functions
    good_code = """
def process_data(data):
    fact_store = FactStore()
    fact_store.add_fact("test", "predicate", "value")
    result = fact_store.query("test", "predicate")
    return result
"""
    
    # Test code with hallucinated functions
    bad_code = """
def process_data(data):
    fact_store = FactStore()
    fact_store.magic_query("test", "predicate")  # Doesn't exist!
    result = fact_store.super_encode()  # Hallucination!
    return result
"""
    
    print("Testing VALID code:")
    print(good_code)
    
    result = ventriloquist.verify_code(
        good_code,
        fact_store=code_store,
        concept_store=concept_store
    )
    
    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Issues: {result['issues'] or 'None'}")
    
    print("\nTesting INVALID code (with hallucinations):")
    print(bad_code)
    
    result = ventriloquist.verify_code(
        bad_code,
        fact_store=code_store,
        concept_store=concept_store
    )
    
    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Issues:")
    for issue in result['issues']:
        print(f"    - {issue}")
    
    print("\n✓ Verification prevents hallucinated APIs from being accepted")


def main():
    """Run the complete demo."""
    print("\n" + "="*70)
    print("HOLOGRAPHIC CODE ENHANCEMENT DEMO")
    print("="*70)
    print("\nThis demo shows how the Hologram prevents code hallucination")
    print("by grounding generation in verified facts (RAG for Code).")
    
    # Step 1: Pre-train with concepts
    concept_store = demo_concept_ingestion()
    
    # Step 2: Index specific codebase
    code_store = demo_code_indexing()
    
    # Step 3: Dual retrieval
    demo_dual_retrieval(concept_store, code_store)
    
    # Step 4: Verification
    demo_code_verification(concept_store, code_store)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Concept Store = General coding knowledge (patterns, algorithms)")
    print("2. Code Store = Project-specific context (your codebase)")
    print("3. Dual Retrieval = Combine both for creative + grounded generation")
    print("4. Verification = Reject hallucinated APIs before they cause bugs")
    print("\nNext Steps:")
    print("- Run: python scripts/ingest_concepts.py --test")
    print("- Run: python scripts/code_indexer.py --directory ./src --test")
    print("- Try: Generate code with your own prompts")
    print("="*70)


if __name__ == "__main__":
    main()


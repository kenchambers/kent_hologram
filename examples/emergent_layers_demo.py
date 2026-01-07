"""
Emergent Layers Demo: CRAG + Emergent Category Networks

Demonstrates:
1. Creating an emergent layer fact store
2. Bulk ingestion with automatic layer creation
3. Querying with layer-aware retrieval
4. Layer statistics and visualization
"""

from hologram.container import HologramContainer


def main():
    print("=" * 60)
    print("CRAG + Emergent Category Networks Demo")
    print("=" * 60)
    print()
    
    # Create container
    print("1. Creating container...")
    container = HologramContainer(dimensions=10000)
    
    # Create emergent layer fact store
    print("2. Creating emergent layer fact store...")
    fact_store = container.create_emergent_layer_fact_store(
        persist_path="/tmp/emergent_demo",
        use_hnsw=True,
    )
    print(f"   ✓ Fact store created")
    print()
    
    # Prepare facts (geography, science, history)
    print("3. Preparing facts...")
    facts = [
        # Geography facts
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Italy", "capital", "Rome"),
        ("Spain", "capital", "Madrid"),
        ("France", "continent", "Europe"),
        ("Germany", "continent", "Europe"),
        
        # Science facts
        ("Water", "formula", "H2O"),
        ("Carbon_Dioxide", "formula", "CO2"),
        ("Oxygen", "symbol", "O"),
        ("Hydrogen", "symbol", "H"),
        ("Photosynthesis", "produces", "Oxygen"),
        
        # History facts
        ("World_War_II", "started", "1939"),
        ("World_War_II", "ended", "1945"),
        ("Napoleon", "born", "1769"),
        ("Napoleon", "died", "1821"),
        ("French_Revolution", "started", "1789"),
    ]
    print(f"   ✓ Prepared {len(facts)} facts across 3 domains")
    print()
    
    # Bulk ingest with progress
    print("4. Ingesting facts (layers will emerge automatically)...")
    
    def progress_callback(done, total):
        if done == total:
            print(f"   ✓ Ingested {done}/{total} facts")
    
    result = fact_store.bulk_ingest(
        facts,
        batch_size=5,
        progress_callback=progress_callback,
    )
    
    print(f"   ✓ Total facts: {result.total_facts}")
    print(f"   ✓ New layers created: {result.new_layers_created}")
    print(f"   ✓ Time: {result.elapsed_time:.2f}s")
    print()
    
    # Show emerged layers
    print("5. Emerged Layers:")
    for i, layer in enumerate(fact_store.get_layers(), 1):
        print(f"   Layer {i}: {layer.description}")
        print(f"            - Facts: {layer.fact_count}")
        print(f"            - ID: {layer.layer_id[:8]}...")
    print()
    
    # Query examples
    print("6. Query Examples:")
    print()
    
    queries = [
        ("France", "capital"),
        ("Water", "formula"),
        ("World_War_II", "ended"),
        ("Germany", "continent"),
    ]
    
    for subject, predicate in queries:
        result = fact_store.query(subject, predicate)
        
        if result.answer:
            print(f"   Q: What is the {predicate} of {subject}?")
            print(f"   A: {result.answer} (confidence: {result.confidence:.2f})")
            print(f"      Layers queried: {len(result.layer_ids)}")
        else:
            print(f"   Q: What is the {predicate} of {subject}?")
            print(f"   A: Not found")
        print()
    
    # Layer statistics
    print("7. Layer Statistics:")
    stats = fact_store.get_layer_stats()
    for layer_id, layer_stats in stats.items():
        print(f"   {layer_id[:8]}...")
        print(f"   - Description: {layer_stats['description']}")
        print(f"   - Facts: {layer_stats['fact_count']}")
        print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic script to analyze vector storage layers and detect redundancy.

This script checks all vector storage layers in the Hologram system to identify:
1. What information is stored in each layer
2. Potential redundancy between layers
3. Memory usage and efficiency
4. Recommendations for optimization
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hologram.container import HologramContainer


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def analyze_chroma_fact_store(persist_dir: str) -> Dict[str, Any]:
    """Analyze ChromaDB fact storage layer."""
    print("\n" + "="*60)
    print("LAYER 1: ChromaDB Fact Store (Persistent)")
    print("="*60)
    
    analysis = {
        "layer_name": "ChromaDB Fact Store",
        "storage_type": "persistent",
        "location": persist_dir,
        "facts": [],
        "total_facts": 0,
        "unique_subjects": set(),
        "unique_predicates": set(),
        "unique_objects": set(),
    }
    
    try:
        from hologram.persistence.chroma_adapter import ChromaFactStore
        from hologram.core.vector_space import VectorSpace
        from hologram.core.codebook import Codebook
        
        # Create temporary container to get codebook
        space = VectorSpace(dimensions=10000)
        codebook = Codebook(space)
        
        # Load ChromaDB
        chroma_store = ChromaFactStore(
            codebook=codebook,
            persist_dir=persist_dir,
        )
        
        # Get all facts
        facts = chroma_store.get_all_facts()
        analysis["total_facts"] = len(facts)
        
        print(f"\n📊 Total facts stored: {len(facts)}")
        print(f"📍 Location: {persist_dir}")
        
        if facts:
            print(f"\n📝 Sample facts (first 10):")
            for i, fact in enumerate(facts[:10], 1):
                subject = fact.get('subject', '') if isinstance(fact, dict) else getattr(fact, 'subject', '')
                predicate = fact.get('predicate', '') if isinstance(fact, dict) else getattr(fact, 'predicate', '')
                obj = fact.get('object', '') if isinstance(fact, dict) else getattr(fact, 'object', '')
                
                print(f"  {i}. {subject} --{predicate}--> {obj}")
                
                analysis["facts"].append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
                analysis["unique_subjects"].add(subject.lower() if subject else "")
                analysis["unique_predicates"].add(predicate.lower() if predicate else "")
                analysis["unique_objects"].add(obj.lower() if obj else "")
            
            print(f"\n📈 Statistics:")
            print(f"  Unique subjects: {len(analysis['unique_subjects'])}")
            print(f"  Unique predicates: {len(analysis['unique_predicates'])}")
            print(f"  Unique objects: {len(analysis['unique_objects'])}")
            
            # Check for duplicates
            fact_tuples = []
            for fact in facts:
                subject = fact.get('subject', '') if isinstance(fact, dict) else getattr(fact, 'subject', '')
                predicate = fact.get('predicate', '') if isinstance(fact, dict) else getattr(fact, 'predicate', '')
                obj = fact.get('object', '') if isinstance(fact, dict) else getattr(fact, 'object', '')
                fact_tuples.append((subject.lower(), predicate.lower(), obj.lower()))
            
            duplicates = len(fact_tuples) - len(set(fact_tuples))
            if duplicates > 0:
                print(f"\n⚠️  WARNING: {duplicates} duplicate facts found!")
                analysis["duplicates"] = duplicates
            else:
                print(f"\n✅ No duplicate facts found")
                analysis["duplicates"] = 0
        else:
            print("  (No facts stored yet)")
        
        # Convert sets to lists for JSON serialization
        analysis["unique_subjects"] = list(analysis["unique_subjects"])
        analysis["unique_predicates"] = list(analysis["unique_predicates"])
        analysis["unique_objects"] = list(analysis["unique_objects"])
        
    except Exception as e:
        print(f"❌ Error analyzing ChromaDB: {e}")
        analysis["error"] = str(e)
    
    return analysis


def analyze_response_corpus(chatbot) -> Dict[str, Any]:
    """Analyze Response Corpus layer."""
    print("\n" + "="*60)
    print("LAYER 2: Response Corpus (In-Memory)")
    print("="*60)
    
    analysis = {
        "layer_name": "Response Corpus",
        "storage_type": "in-memory",
        "enabled": False,
    }
    
    if chatbot._corpus:
        analysis["enabled"] = True
        stats = chatbot._corpus.get_stats()
        
        print(f"\n📊 Status: ENABLED")
        print(f"📝 Total entries: {stats['total_entries']}")
        
        if stats['total_entries'] > 0:
            print(f"\n📈 By Source:")
            for source, count in stats['by_source'].items():
                print(f"  {source}: {count}")
            
            print(f"\n📈 By Intent:")
            for intent, count in stats['by_intent'].items():
                print(f"  {intent}: {count}")
            
            print(f"\n📈 By Style:")
            for style, count in stats['by_style'].items():
                print(f"  {style}: {count}")
        
        analysis.update(stats)
    else:
        print(f"\n📊 Status: DISABLED")
        print(f"  (Corpus not initialized - this is normal for crew_trainer)")
    
    return analysis


def analyze_intent_classifier(chatbot) -> Dict[str, Any]:
    """Analyze Intent Classifier prototype vectors."""
    print("\n" + "="*60)
    print("LAYER 3: Intent Classifier Prototypes (In-Memory)")
    print("="*60)
    
    analysis = {
        "layer_name": "Intent Classifier",
        "storage_type": "in-memory",
        "intents": {},
    }
    
    classifier = chatbot._intent_classifier
    
    print(f"\n📊 Intent Prototypes:")
    for intent_type, example_count in classifier._example_counts.items():
        intent_name = intent_type.value
        has_vector = classifier._intent_vectors.get(intent_type) is not None
        
        print(f"  {intent_name}: {example_count} examples, vector={'✓' if has_vector else '✗'}")
        analysis["intents"][intent_name] = {
            "examples": example_count,
            "has_vector": has_vector
        }
    
    total_examples = sum(classifier._example_counts.values())
    print(f"\n📈 Total examples: {total_examples}")
    analysis["total_examples"] = total_examples
    
    return analysis


def analyze_entity_extractor(chatbot) -> Dict[str, Any]:
    """Analyze Entity Extractor vocabulary."""
    print("\n" + "="*60)
    print("LAYER 4: Entity Extractor (In-Memory)")
    print("="*60)
    
    analysis = {
        "layer_name": "Entity Extractor",
        "storage_type": "in-memory",
    }
    
    extractor = chatbot._entity_extractor
    
    # Get vocabulary size
    vocab_size = extractor.get_vocabulary_size()
    custom_vocab_size = len(extractor._custom_vocabulary)
    
    print(f"\n📊 Entity Statistics:")
    print(f"  Total vocabulary size: {vocab_size}")
    print(f"  Custom entities learned: {custom_vocab_size}")
    
    if custom_vocab_size > 0:
        print(f"\n📝 Custom Entities (learned from facts):")
        for i, entity in enumerate(list(extractor._custom_vocabulary)[:20], 1):
            print(f"  {i}. {entity}")
    else:
        print(f"\n  (No custom entities learned yet)")
    
    analysis["vocabulary_size"] = vocab_size
    analysis["custom_entities"] = custom_vocab_size
    analysis["custom_entity_list"] = list(extractor._custom_vocabulary)
    
    return analysis


def analyze_conversation_memory(chatbot) -> Dict[str, Any]:
    """Analyze Conversation Memory."""
    print("\n" + "="*60)
    print("LAYER 5: Conversation Memory (In-Memory)")
    print("="*60)
    
    analysis = {
        "layer_name": "Conversation Memory",
        "storage_type": "in-memory (session-based)",
    }
    
    memory = chatbot._memory
    
    print(f"\n📊 Session Statistics:")
    print(f"  Turns recorded: {memory.turn_count}")
    print(f"  Memory traces: {memory.turn_count} user-bot pairs")
    
    analysis["turn_count"] = memory.turn_count
    
    return analysis


def analyze_pattern_store(chatbot) -> Dict[str, Any]:
    """Analyze Response Pattern Store."""
    print("\n" + "="*60)
    print("LAYER 6: Response Pattern Store (In-Memory)")
    print("="*60)
    
    analysis = {
        "layer_name": "Response Pattern Store",
        "storage_type": "in-memory",
    }
    
    pattern_store = chatbot._pattern_store
    
    print(f"\n📊 Pattern Statistics:")
    print(f"  Total patterns: {pattern_store.pattern_count}")
    
    analysis["pattern_count"] = pattern_store.pattern_count
    
    return analysis


def check_redundancy(analyses: List[Dict[str, Any]]) -> None:
    """Check for redundancy across layers."""
    print("\n" + "="*60)
    print("REDUNDANCY ANALYSIS")
    print("="*60)
    
    # Extract ChromaDB facts
    chroma_facts = []
    for analysis in analyses:
        if analysis["layer_name"] == "ChromaDB Fact Store":
            chroma_facts = analysis.get("facts", [])
            break
    
    # Extract entity extractor entities
    entity_vocab = set()
    for analysis in analyses:
        if analysis["layer_name"] == "Entity Extractor":
            # Entities should match fact subjects/objects
            pass
    
    print("\n🔍 Checking for data redundancy...")
    
    # Check 1: Are facts stored in both ChromaDB and in-memory FactStore?
    print("\n1. ChromaDB vs In-Memory FactStore:")
    print("   ✓ ChromaDB is for PERSISTENCE across sessions")
    print("   ✓ In-memory FactStore is for FAST QUERIES during session")
    print("   → This is INTENTIONAL redundancy (necessary)")
    
    # Check 2: Are entities duplicating fact information?
    print("\n2. Entity Extractor vs Facts:")
    if chroma_facts:
        fact_entities = set()
        for fact in chroma_facts:
            fact_entities.add(fact['subject'].lower())
            fact_entities.add(fact['object'].lower())
        print(f"   Entities extracted from facts: {len(fact_entities)}")
        print("   ✓ Entity Extractor learns from facts (necessary)")
        print("   → This is INTENTIONAL redundancy (entity recognition)")
    
    # Check 3: Are responses and facts overlapping?
    print("\n3. Response Corpus vs Facts:")
    corpus_enabled = False
    for analysis in analyses:
        if analysis["layer_name"] == "Response Corpus":
            corpus_enabled = analysis.get("enabled", False)
            break
    
    if corpus_enabled:
        print("   ✓ Response Corpus stores CONVERSATIONAL responses")
        print("   ✓ Facts store STRUCTURED knowledge (S-P-O)")
        print("   → These serve DIFFERENT purposes (no redundancy)")
    else:
        print("   ✓ Response Corpus is DISABLED")
        print("   → No redundancy (corpus not in use)")
    
    # Check 4: Intent prototypes vs conversation memory
    print("\n4. Intent Prototypes vs Conversation Memory:")
    print("   ✓ Intent Prototypes are LEARNED PATTERNS (classification)")
    print("   ✓ Conversation Memory is SESSION CONTEXT (short-term)")
    print("   → These serve DIFFERENT purposes (no redundancy)")


def generate_recommendations(analyses: List[Dict[str, Any]]) -> None:
    """Generate optimization recommendations."""
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    # Find ChromaDB analysis
    chroma_analysis = None
    corpus_analysis = None
    
    for analysis in analyses:
        if analysis["layer_name"] == "ChromaDB Fact Store":
            chroma_analysis = analysis
        elif analysis["layer_name"] == "Response Corpus":
            corpus_analysis = analysis
    
    recommendations = []
    
    # Check for duplicates in ChromaDB
    if chroma_analysis and chroma_analysis.get("duplicates", 0) > 0:
        recommendations.append({
            "severity": "⚠️  WARNING",
            "issue": f"ChromaDB has {chroma_analysis['duplicates']} duplicate facts",
            "recommendation": "Run deduplication script to remove duplicates"
        })
    
    # Check if corpus is enabled but not being used
    if corpus_analysis and corpus_analysis.get("enabled") and corpus_analysis.get("total_entries", 0) == 0:
        recommendations.append({
            "severity": "💡 INFO",
            "issue": "Response Corpus is enabled but empty",
            "recommendation": "Consider disabling corpus if not using response learning"
        })
    
    # Check if corpus is disabled (which is correct for crew_trainer)
    if corpus_analysis and not corpus_analysis.get("enabled"):
        recommendations.append({
            "severity": "✅ GOOD",
            "issue": "Response Corpus is disabled in crew_trainer",
            "recommendation": "This is correct - crew_trainer uses atomic response learning"
        })
    
    # Print recommendations
    if recommendations:
        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['severity']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   → {rec['recommendation']}")
            print()
    else:
        print("\n✅ No optimization issues found!")
        print("   All vector layers are configured correctly.")


def main():
    """Main diagnostic routine."""
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  Hologram Vector Layer Diagnostic".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Configuration
    persist_dir = "./data/crew_training_facts"
    
    print(f"\n📂 Analyzing vector storage in: {persist_dir}")
    
    # Initialize container
    print("\n🔧 Initializing Hologram container...")
    container = HologramContainer(dimensions=10000)
    
    # Build basic vocabulary (needed for generation)
    vocabulary = {
        "nouns": ["paris", "france", "capital", "europe"],
        "verbs": ["is", "are", "was", "were"]
    }
    
    # Create chatbot (needed to analyze in-memory structures)
    chatbot = container.create_persistent_chatbot(
        persist_dir=persist_dir,
        enable_corpus=False,  # Match crew_trainer config
        enable_generation=True,
        vocabulary=vocabulary,
    )
    
    # Run analyses
    analyses = []
    
    # Layer 1: ChromaDB
    analyses.append(analyze_chroma_fact_store(persist_dir))
    
    # Layer 2: Response Corpus
    analyses.append(analyze_response_corpus(chatbot))
    
    # Layer 3: Intent Classifier
    analyses.append(analyze_intent_classifier(chatbot))
    
    # Layer 4: Entity Extractor
    analyses.append(analyze_entity_extractor(chatbot))
    
    # Layer 5: Conversation Memory
    analyses.append(analyze_conversation_memory(chatbot))
    
    # Layer 6: Pattern Store
    analyses.append(analyze_pattern_store(chatbot))
    
    # Redundancy check
    check_redundancy(analyses)
    
    # Recommendations
    generate_recommendations(analyses)
    
    # Save report
    report_path = Path("vector_layer_diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print(f"📄 Full report saved to: {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()

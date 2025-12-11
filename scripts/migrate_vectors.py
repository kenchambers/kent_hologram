#!/usr/bin/env python3
"""
Migration script to update ChromaDB vectors to SemanticCodebook.

This script:
1. Reads all facts and learned responses from the existing ChromaDB.
2. Clears the database.
3. Re-adds all items using the new SemanticCodebook (generating semantic vectors).

Run this once after upgrading to SemanticCodebook to fix the "lobotomy" issue.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hologram.container import HologramContainer
from hologram.persistence.chroma_adapter import ChromaFactStore, ChromaResponseCorpus

def migrate():
    print("Initializing Hologram with SemanticCodebook...")
    container = HologramContainer(dimensions=10000, use_semantic=True)
    codebook = container.codebook
    
    print(f"Codebook type: {type(codebook).__name__}")
    if type(codebook).__name__ != "SemanticCodebook":
        print("ERROR: SemanticCodebook not active. Please install sentence-transformers.")
        return

    # --- Migrate Facts ---
    print("\nMigrating Facts...")
    fact_store = ChromaFactStore(codebook, "./data/crew_training_facts")
    
    # Get all existing facts (metadata only)
    # We access the internal collection directly to get everything
    all_facts_data = fact_store._collection.get(include=["metadatas", "documents"])
    metadatas = all_facts_data.get("metadatas", [])
    
    if not metadatas:
        print("No facts found to migrate.")
    else:
        print(f"Found {len(metadatas)} facts. Re-indexing...")
        
        # Clear existing data
        fact_store.clear()
        
        # Re-add facts
        count = 0
        for meta in metadatas:
            try:
                fact_store.add_fact(
                    subject=meta["subject"],
                    predicate=meta["predicate"],
                    obj=meta["object"],
                    source=meta.get("source")
                )
                count += 1
                if count % 10 == 0:
                    print(f"  Processed {count}/{len(metadatas)} facts...")
            except Exception as e:
                print(f"  Error migrating fact {meta}: {e}")
                
        print(f"✓ Successfully migrated {count} facts.")

    # --- Migrate Response Corpus ---
    # The chatbot creates corpus with enable_corpus=True
    # The default path in container.create_persistent_chatbot is ./data/hologram_corpus?
    # Wait, in crew_trainer.py:
    # self.chatbot = self.container.create_persistent_chatbot(persist_dir=persist_dir, enable_corpus=True)
    # create_persistent_chatbot uses the SAME persist_dir for corpus?
    # No:
    # if enable_corpus:
    #     response_corpus = ChromaResponseCorpus(..., persist_dir=persist_dir)
    
    # So we should check the same directory.
    
    print("\nMigrating Response Corpus...")
    corpus = ChromaResponseCorpus(codebook, "./data/crew_training_facts") # Using same dir as crew_trainer
    
    all_corpus_data = corpus._collection.get(include=["metadatas", "documents"])
    metadatas = all_corpus_data.get("metadatas", [])
    
    if not metadatas:
        print("No responses found to migrate.")
    else:
        print(f"Found {len(metadatas)} responses. Re-indexing...")
        
        # Clear existing data
        corpus.clear()
        
        # Re-add responses
        count = 0
        for meta in metadatas:
            try:
                # We need context vector to add_response.
                # But we don't have the original context vector (it's incompatible anyway).
                # We can't perfectly reconstruct the context vector without the conversation history.
                # However, ChromaResponseCorpus stores the context vector as the embedding.
                # Since we are changing the vector space, the old context vector is useless.
                
                # Problem: We can't regenerate the context vector for the response because we don't have the conversation history that led to it.
                # The metadata only has response, intent, style.
                
                # Option A: Drop old responses (start fresh).
                # Option B: Create a synthetic context vector based on the intent/style?
                # Option C: Use the response itself as the context (auto-associative).
                
                # Let's use Option C as a best-effort recovery. 
                # We'll encode the response text itself as the "context". 
                # This means "when saying X, say X". It's not perfect but better than nothing.
                # Ideally, we would have stored the prompt, but we didn't.
                
                dummy_context = codebook.encode(meta["response"])
                
                corpus.add_response(
                    context_vector=dummy_context,
                    response=meta["response"],
                    intent=meta["intent"],
                    style=meta["style"],
                    source=meta.get("source", "migrated")
                )
                count += 1
                if count % 10 == 0:
                    print(f"  Processed {count}/{len(metadatas)} responses...")
            except Exception as e:
                print(f"  Error migrating response {meta}: {e}")
                
        print(f"✓ Successfully migrated {count} responses.")
        print("Note: Responses were migrated using auto-association (context = response) as history was lost.")

if __name__ == "__main__":
    migrate()

#!/usr/bin/env python3
"""
Complete demonstration of Hologram holographic memory system.

Shows the full system including:
1. Storing facts holographically
2. Querying with confidence scores
3. Refusal when uncertain ("I don't know")
4. Citation tracking
5. Persistence (save/load)
6. Bounded hallucination demonstration
7. Resonant Cavity Architecture
   - Resonator factorization
   - Style modulation
   - Disfluency injection
   - Constrained generation
8. Conversational Learning with ChromaDB Persistence (NEW)
   - HDC-based intent classification
   - Teaching facts via natural language
   - Fact persistence across sessions
"""

from pathlib import Path

from hologram import (
    CitationEnforcer,
    ConfidenceScorer,
    HologramContainer,
    RefusalPolicy,
    StateManager,
    # Resonant Cavity imports
    Resonator,
    ResonantGenerator,
    StyleType,
    SesameModulator,
)
from hologram.core.operations import Operations


def main():
    print("=" * 70)
    print("  HOLOGRAM: Complete System Demonstration")
    print("  Bentov-Style Holographic Memory with Resonant Cavity Architecture")
    print("=" * 70)
    print()

    # =========================================================================
    # SETUP
    # =========================================================================
    print("Initializing 10,000-dimensional holographic memory system...")
    container = HologramContainer(dimensions=10000)
    fact_store = container.create_fact_store()
    confidence_scorer = ConfidenceScorer()
    refusal_policy = RefusalPolicy(confidence_scorer)
    citation_enforcer = CitationEnforcer(fact_store)
    state_manager = StateManager()
    print(f"  {container}")
    print()

    # =========================================================================
    # PHASE 1: Store Facts with Citations
    # =========================================================================
    print("Phase 1: Storing facts with citations...")
    print("-" * 70)

    facts_to_store = [
        ("France", "capital", "Paris", "Wikipedia"),
        ("Germany", "capital", "Berlin", "Wikipedia"),
        ("Japan", "capital", "Tokyo", "Wikipedia"),
        ("Earth", "shape", "round", "NASA"),
        ("Water", "boiling_point", "100C", "Chemistry Textbook"),
        ("Python", "creator", "Guido", "Python.org"),
        # Additional facts for Resonant Cavity demo
        ("cat", "eats", "fish", "Nature"),
        ("dog", "chases", "cat", "Nature"),
        ("fish", "swims", "water", "Nature"),
    ]

    for subject, predicate, obj, source in facts_to_store:
        fact = fact_store.add_fact(subject, predicate, obj, source=source)
        print(f"  {citation_enforcer.format_citation(fact)}")

    print(f"\n{fact_store}")
    print()

    # =========================================================================
    # PHASE 2: Querying with Confidence
    # =========================================================================
    print("Phase 2: Querying with confidence scoring...")
    print("-" * 70)

    queries = [
        ("France", "capital", "should know"),
        ("Germany", "capital", "should know"),
        ("Mars", "color", "unknown - should refuse"),
    ]

    for subject, predicate, note in queries:
        print(f"\nQuery: {subject} --{predicate}--> ? ({note})")

        answer, confidence = fact_store.query(subject, predicate)

        refusal = refusal_policy.evaluate(answer, confidence)

        if refusal.should_refuse:
            print(f"  {refusal_policy.format_refusal(refusal)}")
        else:
            response = confidence_scorer.format_response(answer, confidence)
            print(f"  Answer: {response}")

            supporting_fact = citation_enforcer.find_supporting_fact(
                subject, predicate, answer
            )
            if supporting_fact:
                citation = citation_enforcer.format_citation(supporting_fact)
                print(f"    Citation: {citation}")

    print()

    # =========================================================================
    # PHASE 3: Demonstrate Bounded Hallucination
    # =========================================================================
    print("Phase 3: Bounded hallucination demonstration...")
    print("-" * 70)
    print("\nTesting queries on unknown topics:")

    unknown_queries = [
        ("Unicorn", "color"),
        ("Atlantis", "location"),
        ("DragonFruit", "taste"),
    ]

    for subject, predicate in unknown_queries:
        answer, confidence = fact_store.query(subject, predicate)
        print(f"\n  Query: {subject} --{predicate}--> ?")
        print(f"  Answer: {answer}")
        print(f"  Confidence: {confidence:.3f}")

        refusal = refusal_policy.evaluate(answer, confidence)
        if refusal.should_refuse:
            print("  CORRECTLY REFUSED (bounded hallucination working!)")
        else:
            print("  Should have refused (low confidence)")

    print()

    # =========================================================================
    # PHASE 4: Persistence
    # =========================================================================
    print("Phase 4: Testing persistence...")
    print("-" * 70)

    save_path = Path("./data/complete_demo")

    print(f"\nSaving to: {save_path}")
    state_manager.save(fact_store, save_path, description="Complete demo state")
    print("  Saved!")

    print("\nClearing memory (simulating new session)...")
    del fact_store
    print("  Memory cleared")

    print(f"\nLoading from: {save_path}")
    restored_fs = state_manager.load(save_path, validate_checksum=True)
    print(f"  Loaded: {restored_fs}")

    print("\nVerifying facts preserved...")
    restored_citation_enforcer = CitationEnforcer(restored_fs)
    answer, conf = restored_fs.query("France", "capital")
    fact = restored_citation_enforcer.find_supporting_fact("France", "capital", answer)

    if answer == "Paris" and conf > 0.15:
        print("  Facts correctly preserved!")
        print(f"     Answer: {answer} (confidence: {conf:.2%})")
        print(f"     {restored_citation_enforcer.format_citation(fact)}")
        print(f"\n  Note: Confidence of {conf:.2%} is expected for holographic storage")
        print(f"        with {restored_fs.fact_count} bundled facts (interference normal)")
    else:
        print("  Data corruption detected!")
        print(f"     Expected: Paris with confidence > 0.15")
        print(f"     Got: {answer} with confidence {conf:.2%}")

    print()

    # =========================================================================
    # PHASE 5: Resonant Cavity Architecture (NEW)
    # =========================================================================
    print("Phase 5: Resonant Cavity Architecture...")
    print("-" * 70)
    print("\nThe Resonant Cavity enables constrained generation via:")
    print("  - Resonator: Factorizes thoughts into (S, V, O)")
    print("  - TargetEncoder: Packages constraints")
    print("  - DivergenceCalculator: Verifies alignment")
    print("  - SesameModulator: Adds style and disfluency")

    # Create vocabulary from stored facts
    vocabulary = {
        "nouns": ["cat", "dog", "fish", "water", "bird", "mouse"],
        "verbs": ["eats", "chases", "swims", "flies", "catches"],
    }

    print(f"\nVocabulary: {len(vocabulary['nouns'])} nouns, {len(vocabulary['verbs'])} verbs")

    # Create Resonator
    resonator = container.create_resonator()
    print(f"\n  {resonator}")

    # Create a composite "thought" vector
    # Combining cat + eats to see what the Resonator extracts
    codebook = container.codebook
    cat_vec = codebook.encode("cat")
    eats_vec = codebook.encode("eats")
    fish_vec = codebook.encode("fish")

    # Create thought: (cat bound to SUBJECT) + (eats bound to VERB) + (fish bound to OBJECT)
    r_subj = codebook.get_role("SUBJECT")
    r_verb = codebook.get_role("VERB")
    r_obj = codebook.get_role("OBJECT")

    thought = Operations.bundle(
        Operations.bind(cat_vec, r_subj),
        Operations.bind(eats_vec, r_verb),
        Operations.bind(fish_vec, r_obj),
    )

    print("\n  Creating thought vector: 'cat eats fish'")
    print("  Running Resonator factorization...")

    result = resonator.resonate(thought, vocabulary["nouns"], vocabulary["verbs"])

    print(f"\n  Resonator Result:")
    print(f"    Subject: {result.subject_word} (confidence: {result.confidence.get('subject', 0):.3f})")
    print(f"    Verb: {result.verb_word} (confidence: {result.confidence.get('verb', 0):.3f})")
    print(f"    Object: {result.object_word} (confidence: {result.confidence.get('object', 0):.3f})")
    print(f"    Converged: {result.converged} in {result.iterations} iterations")

    if result.subject_word == "cat" and result.verb_word == "eats" and result.object_word == "fish":
        print("\n  SUCCESS: Resonator correctly factorized the thought!")
    else:
        print("\n  Note: Resonator found a different factorization (expected with noise)")

    print()

    # =========================================================================
    # PHASE 6: Style Modulation Demo
    # =========================================================================
    print("Phase 6: Style modulation demonstration...")
    print("-" * 70)

    sesame = container.create_sesame_modulator()
    print(f"\n  {sesame}")

    print("\n  Style vectors affect word selection:")
    for style in [StyleType.FORMAL, StyleType.CASUAL, StyleType.URGENT, StyleType.NEUTRAL]:
        style_vec = sesame.get_style_vector(style)
        norm = float(style_vec.norm().item())
        print(f"    {style.value:8s}: norm={norm:.2f} (zero norm = neutral)")

    print("\n  Disfluency injection thresholds:")
    for conf in [0.5, 0.35, 0.25, 0.15]:
        should_filler = sesame.should_inject_disfluency(conf)
        if should_filler:
            filler = sesame.select_filler(conf)
            print(f"    confidence={conf:.2f}: inject '{filler.value}'")
        else:
            print(f"    confidence={conf:.2f}: no filler needed")

    print()

    # =========================================================================
    # PHASE 7: Full Generation Demo
    # =========================================================================
    print("Phase 7: Full Resonant Generation...")
    print("-" * 70)

    generator = container.create_resonant_generator(vocabulary)
    print(f"\n  {generator}")

    print("\n  Generating from 'cat eats fish' thought vector...")
    gen_result = generator.generate(thought, style=StyleType.NEUTRAL)

    print(f"\n  Generated text: '{gen_result.text}'")
    print(f"  Tokens: {gen_result.tokens}")
    print(f"\n  Metrics:")
    print(f"    Total tokens: {gen_result.metrics.total_tokens}")
    print(f"    Accepted first try: {gen_result.metrics.accepted_first_try}")
    print(f"    Accepted with correction: {gen_result.metrics.accepted_with_correction}")
    print(f"    Rejected: {gen_result.metrics.rejected_tokens}")
    print(f"    Fillers injected: {gen_result.metrics.fillers_injected}")
    print(f"    Average similarity: {gen_result.metrics.average_similarity:.3f}")
    print(f"    Acceptance rate: {gen_result.metrics.acceptance_rate:.1%}")
    print(f"    Hallucination risk: {gen_result.metrics.hallucination_risk:.1%}")

    print("\n  Generation trace:")
    for trace in gen_result.trace:
        print(f"    {trace}")

    print()

    # =========================================================================
    # PHASE 8: Conversational Learning with Persistence
    # =========================================================================
    print("Phase 8: Conversational Learning with ChromaDB Persistence...")
    print("-" * 70)
    print("\nThe conversational chatbot learns from interactions using pure HDC:")
    print("  - Intent classification via example learning (no hardcoded keywords)")
    print("  - Teaching detection via TEACHING intent prototype")
    print("  - Fact persistence via ChromaDB with cosine distance")

    # Create persistent chatbot
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    persist_path = f"{temp_dir}/demo_facts"

    print(f"\n  Creating persistent chatbot ({persist_path})...")

    chatbot = container.create_persistent_chatbot(persist_path)
    greeting = chatbot.start_session()
    print(f"  Greeting: {greeting}")

    # Teach facts via natural language
    print("\n  Teaching facts via natural language:")

    teachings = [
        "the capital of Spain is Madrid",
        "the capital of Italy is Rome",
    ]

    for text in teachings:
        print(f"    User: \"{text}\"")
        response = chatbot.respond(text)
        print(f"    Bot: {response}")

    # Query taught facts
    print("\n  Querying taught facts:")

    queries = [
        "What is the capital of Spain?",
        "What is the capital of Italy?",
    ]

    for text in queries:
        print(f"    User: \"{text}\"")
        response = chatbot.respond(text)
        print(f"    Bot: {response}")

    chatbot.end_session()

    # Test persistence by creating new chatbot instance
    print("\n  Testing persistence (creating new chatbot instance)...")

    chatbot2 = container.create_persistent_chatbot(persist_path)
    chatbot2.start_session()

    print("    User: \"What is the capital of Spain?\"")
    response = chatbot2.respond("What is the capital of Spain?")
    print(f"    Bot: {response}")

    if "madrid" in response.lower():
        print("\n  SUCCESS: Facts persisted across sessions!")
    else:
        print("\n  Note: Fact not found (may need threshold tuning)")

    chatbot2.end_session()

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  Core Holographic Memory:")
    print("    Holographic storage via interference patterns")
    print("    Deterministic vector generation (same concept = same vector)")
    print("    Confidence-based responses")
    print("    Bounded hallucination (refuses when uncertain)")
    print("    Citation tracking for all facts")
    print("    Persistence with checksum validation")
    print("\n  Resonant Cavity Architecture:")
    print("    Resonator for thought factorization")
    print("    Style modulation (formal/casual/urgent)")
    print("    Disfluency injection (um, uh, ...)")
    print("    Constrained generation with verification")
    print("    Generation metrics and auditing")
    print("\n  Conversational Learning (NEW):")
    print("    HDC-based intent classification")
    print("    Teaching detection via TEACHING intent")
    print("    ChromaDB persistence for facts")
    print("    Natural language fact extraction")
    print()
    print("Try the interactive chat: python -m hologram.chat.interface")
    print()


if __name__ == "__main__":
    main()

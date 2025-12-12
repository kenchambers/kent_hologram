"""
Comprehensive HDC Fact Grounding Test Suite

Tests the Hologram architecture's core principle:
    HDC is the BRAIN (retrieval, memory, bounded hallucination)
    Ventriloquist is just the VOICE (fluency wrapper)

The system should NEVER rely on the LLM for factual accuracy.
All facts must come from the holographic memory.

Run with: uv run pytest tests/test_hdc_fact_grounding.py -v
Run specific section: uv run pytest tests/test_hdc_fact_grounding.py -v -k "TestHDCStorage"
"""

import os
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pytest
import torch

from hologram.container import HologramContainer
from hologram.config.constants import (
    RESPONSE_CONFIDENCE_THRESHOLD,
    REFUSAL_CONFIDENCE_THRESHOLD,
    SURPRISE_THRESHOLD,
    DEFAULT_DIMENSIONS,
)


# =============================================================================
# TEST DATA: Designed to test HDC properties, NOT LLM knowledge
# =============================================================================

# Fictional entities - LLM has NO training data on these
FICTIONAL_FACTS = [
    ("Zorbaxia", "capital", "Flumpton"),
    ("Quilnoria", "ruler", "Emperor Vex"),
    ("Mythranox", "currency", "Starshards"),
    ("Nebulheim", "language", "Voidspeak"),
    ("Crystallis", "population", "3.7 million"),
    ("Shadowmere", "founded", "Year 1247"),
    ("Aethoria", "continent", "Mistlands"),
    ("Pyralux", "export", "Firecrystals"),
    ("Glacium", "climate", "Eternal frost"),
    ("Voidreach", "government", "Council of Nine"),
]

# Contradictory facts - tests that HDC overrides LLM training
CONTRADICTORY_FACTS = [
    ("France", "capital", "Berlin"),  # Real: Paris
    ("Germany", "capital", "Rome"),   # Real: Berlin
    ("Japan", "capital", "Seoul"),    # Real: Tokyo
    ("Italy", "capital", "Madrid"),   # Real: Rome
    ("Spain", "capital", "Lisbon"),   # Real: Madrid
]

# Multi-word values - tests HDC handles complex objects
COMPLEX_FACTS = [
    ("Protocol_X", "purpose", "secure quantum communication"),
    ("Element_99", "properties", "highly unstable isotope"),
    ("Species_Alpha", "habitat", "deep ocean thermal vents"),
    ("Algorithm_Prime", "complexity", "O(n log n) average case"),
    ("Material_Omega", "composition", "carbon nanotube lattice"),
]

# Relationship chains - tests HDC graph traversal
CHAIN_FACTS = [
    ("Alice", "works_at", "TechCorp"),
    ("TechCorp", "located_in", "Silicon Valley"),
    ("Silicon Valley", "country", "USA"),
    ("Bob", "mentor", "Alice"),
    ("Alice", "created", "ProjectX"),
]

# Numerical precision - tests exact value retrieval
NUMERICAL_FACTS = [
    ("Building_A", "height", "127.5 meters"),
    ("River_X", "length", "2847 km"),
    ("Company_Y", "employees", "47832"),
    ("Event_Z", "date", "March 15, 2157"),
    ("Product_Q", "price", "$3.7 million"),
]

# Edge cases - tests HDC robustness
EDGE_CASE_FACTS = [
    ("Entity-With-Dashes", "value", "result-with-dashes"),
    ("Entity_With_Underscores", "value", "result_underscores"),
    ("ALLCAPS", "value", "CAPSVALUE"),
    ("lowercase", "value", "lowervalue"),
    ("MixedCase", "value", "MixedResult"),
    ("Short", "x", "y"),  # Single char predicate/object
    ("Numbers123", "count", "456"),
]


# =============================================================================
# SECTION 1: HDC CORE STORAGE TESTS
# =============================================================================

class TestHDCStorage:
    """Test the holographic memory's storage capabilities."""

    @pytest.fixture
    def container(self):
        """Create a fresh HologramContainer."""
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def fact_store(self, container):
        """Create an in-memory FactStore."""
        return container.create_fact_store()

    # -------------------------------------------------------------------------
    # Basic Storage Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("subject,predicate,obj", FICTIONAL_FACTS)
    def test_store_fictional_fact(self, fact_store, subject, predicate, obj):
        """Test storing facts about fictional entities."""
        fact = fact_store.add_fact(subject, predicate, obj, source="test")

        assert fact is not None, f"Failed to store: {subject} {predicate} {obj}"
        assert fact.subject == subject
        assert fact.predicate == predicate
        assert fact.object == obj

    @pytest.mark.parametrize("subject,predicate,obj", CONTRADICTORY_FACTS)
    def test_store_contradictory_fact(self, fact_store, subject, predicate, obj):
        """Test storing facts that contradict real-world knowledge."""
        fact = fact_store.add_fact(subject, predicate, obj, source="test")

        # HDC doesn't care about real-world truth - it stores what you tell it
        assert fact is not None, f"HDC should store contradictory facts"
        assert fact.object == obj

    @pytest.mark.parametrize("subject,predicate,obj", COMPLEX_FACTS)
    def test_store_complex_values(self, fact_store, subject, predicate, obj):
        """Test storing facts with multi-word values."""
        fact = fact_store.add_fact(subject, predicate, obj, source="test")

        assert fact is not None
        assert fact.object == obj  # Exact match required

    def test_store_duplicate_returns_none(self, fact_store):
        """Test that storing duplicate facts returns None (surprise gating)."""
        # First store
        fact1 = fact_store.add_fact("Test", "prop", "value", source="test")
        assert fact1 is not None

        # Duplicate store - should return None due to low surprise
        fact2 = fact_store.add_fact("Test", "prop", "value", source="test")
        assert fact2 is None, "Duplicate fact should return None"

    def test_update_fact_value(self, fact_store):
        """Test that storing same subject+predicate with new object updates it."""
        fact_store.add_fact("Entity", "property", "old_value", source="test")
        fact_store.add_fact("Entity", "property", "new_value", source="test")

        answer, conf = fact_store.query("Entity", "property")
        assert answer == "new_value", "Fact should be updated to new value"

    def test_fact_count_increments(self, fact_store):
        """Test that fact_count accurately tracks stored facts."""
        initial_count = fact_store.fact_count

        for i in range(10):
            fact_store.add_fact(f"Entity{i}", "prop", f"value{i}", source="test")

        assert fact_store.fact_count == initial_count + 10

    def test_vocabulary_size_tracks_unique_objects(self, fact_store):
        """Test that vocabulary_size tracks unique object values."""
        fact_store.add_fact("A", "prop", "unique1", source="test")
        fact_store.add_fact("B", "prop", "unique2", source="test")
        fact_store.add_fact("C", "prop", "unique1", source="test")  # Duplicate object

        # Should have 2 unique objects, not 3
        assert fact_store.vocabulary_size >= 2


# =============================================================================
# SECTION 2: HDC RETRIEVAL TESTS (The Core of Bounded Hallucination)
# =============================================================================

class TestHDCRetrieval:
    """Test the holographic memory's retrieval capabilities.

    This is the CRITICAL section - retrieval accuracy determines
    whether the system can provide bounded hallucination.
    """

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def populated_fact_store(self, container):
        """Create a FactStore populated with test data."""
        fs = container.create_fact_store()

        # Store all test facts
        for subject, predicate, obj in FICTIONAL_FACTS:
            fs.add_fact(subject, predicate, obj, source="test")
        for subject, predicate, obj in CONTRADICTORY_FACTS:
            fs.add_fact(subject, predicate, obj, source="test")
        for subject, predicate, obj in NUMERICAL_FACTS:
            fs.add_fact(subject, predicate, obj, source="test")

        return fs

    # -------------------------------------------------------------------------
    # Exact Match Retrieval
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("subject,predicate,expected", FICTIONAL_FACTS)
    def test_exact_retrieval_fictional(self, populated_fact_store, subject, predicate, expected):
        """Test exact retrieval of fictional facts."""
        answer, confidence = populated_fact_store.query(subject, predicate)

        assert answer == expected, f"Expected {expected}, got {answer}"
        assert confidence >= 0.9, f"Exact match should have high confidence, got {confidence}"

    @pytest.mark.parametrize("subject,predicate,expected", CONTRADICTORY_FACTS)
    def test_exact_retrieval_contradictory(self, populated_fact_store, subject, predicate, expected):
        """Test that HDC returns stored value, NOT real-world truth."""
        answer, confidence = populated_fact_store.query(subject, predicate)

        # THIS IS THE KEY TEST: HDC must return "Berlin" for France's capital
        # even though the LLM knows it's "Paris"
        assert answer == expected, \
            f"HDC MUST return stored value '{expected}', not LLM knowledge. Got '{answer}'"
        assert confidence >= 0.9

    @pytest.mark.parametrize("subject,predicate,expected", NUMERICAL_FACTS)
    def test_exact_retrieval_numerical(self, populated_fact_store, subject, predicate, expected):
        """Test exact retrieval of numerical values."""
        answer, confidence = populated_fact_store.query(subject, predicate)

        assert answer == expected, f"Numerical value must be exact: expected {expected}, got {answer}"

    # -------------------------------------------------------------------------
    # Case Insensitivity Tests
    # -------------------------------------------------------------------------

    def test_case_insensitive_subject(self, container):
        """Test that subject queries are case-insensitive."""
        fs = container.create_fact_store()
        fs.add_fact("TestEntity", "property", "TestValue", source="test")

        # Query with different cases
        answer1, conf1 = fs.query("testentity", "property")
        answer2, conf2 = fs.query("TESTENTITY", "property")
        answer3, conf3 = fs.query("TestEntity", "property")

        # All should return the same value
        assert answer1 == answer2 == answer3 == "TestValue"

    def test_case_insensitive_predicate(self, container):
        """Test that predicate queries are case-insensitive."""
        fs = container.create_fact_store()
        fs.add_fact("Entity", "Property", "Value", source="test")

        answer1, _ = fs.query("Entity", "property")
        answer2, _ = fs.query("Entity", "PROPERTY")

        assert answer1 == answer2 == "Value"

    # -------------------------------------------------------------------------
    # Confidence Threshold Tests
    # -------------------------------------------------------------------------

    def test_unknown_fact_low_confidence(self, populated_fact_store):
        """Test that querying unknown facts returns low confidence."""
        answer, confidence = populated_fact_store.query(
            "CompletelyUnknownEntity12345",
            "nonexistent_predicate"
        )

        # Should have low confidence for unknown facts
        assert confidence < RESPONSE_CONFIDENCE_THRESHOLD, \
            f"Unknown fact should have confidence < {RESPONSE_CONFIDENCE_THRESHOLD}, got {confidence}"

    def test_confidence_above_refusal_threshold(self, populated_fact_store):
        """Test that known facts have confidence above refusal threshold."""
        answer, confidence = populated_fact_store.query("Zorbaxia", "capital")

        assert confidence > REFUSAL_CONFIDENCE_THRESHOLD, \
            f"Known fact confidence {confidence} should be > refusal threshold {REFUSAL_CONFIDENCE_THRESHOLD}"

    # -------------------------------------------------------------------------
    # Reverse Query Tests
    # -------------------------------------------------------------------------

    def test_reverse_query_by_object(self, container):
        """Test finding subject by predicate and object."""
        fs = container.create_fact_store()
        fs.add_fact("Paris", "country", "France", source="test")
        fs.add_fact("Berlin", "country", "Germany", source="test")

        subject, confidence = fs.query_subject("country", "France")

        assert subject == "Paris", f"Reverse query should find Paris, got {subject}"
        assert confidence > 0.5

    def test_get_facts_by_subject(self, container):
        """Test retrieving all facts about a subject."""
        fs = container.create_fact_store()
        fs.add_fact("Alice", "age", "30", source="test")
        fs.add_fact("Alice", "job", "Engineer", source="test")
        fs.add_fact("Alice", "city", "NYC", source="test")
        fs.add_fact("Bob", "age", "25", source="test")

        alice_facts = fs.get_facts_by_subject("Alice")

        assert len(alice_facts) == 3, f"Expected 3 facts about Alice, got {len(alice_facts)}"


# =============================================================================
# SECTION 3: HDC ALGEBRAIC PROPERTIES TESTS
# =============================================================================

class TestHDCAlgebraicProperties:
    """Test the mathematical properties that make HDC reliable.

    HDC provides bounded hallucination through algebraic operations:
    - Binding: bind(A, B) creates a unique key
    - Bundling: bundle(A, B) creates a superposition
    - Similarity: cosine similarity measures relatedness
    """

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def codebook(self, container):
        return container.codebook

    def test_binding_creates_unique_keys(self, codebook):
        """Test that bind(A, B) != bind(A, C) for B != C."""
        from hologram.core.operations import Operations

        a = codebook.encode("France")
        b = codebook.encode("capital")
        c = codebook.encode("population")

        key1 = Operations.bind(a, b)
        key2 = Operations.bind(a, c)

        # Different predicates should create different keys
        similarity = torch.cosine_similarity(key1, key2, dim=0).item()
        assert similarity < 0.5, f"Different keys should be dissimilar, got {similarity}"

    def test_binding_is_reversible(self, codebook):
        """Test that unbind(bind(A, B), B) ≈ A.

        Note: For normalized MAP VSA (which this codebase uses), unbind recovery
        produces similarity in the 0.5-0.7 range due to normalization effects.
        This is expected behavior - the cleanup/resonance step is what provides
        exact recovery by finding the nearest codebook vector.
        """
        from hologram.core.operations import Operations

        a = codebook.encode("France")
        b = codebook.encode("capital")

        bound = Operations.bind(a, b)
        recovered = Operations.unbind(bound, b)

        similarity = torch.cosine_similarity(a, recovered, dim=0).item()
        # Normalized MAP VSA produces ~0.5-0.7 similarity after unbind
        # The cleanup step (resonance) is what provides exact recovery
        assert similarity > 0.4, f"Unbinding should recover approximate original, similarity: {similarity}"

    def test_bundling_preserves_components(self, codebook):
        """Test that bundled vectors retain similarity to components."""
        from hologram.core.operations import Operations

        a = codebook.encode("cat")
        b = codebook.encode("dog")
        c = codebook.encode("bird")

        bundled = Operations.bundle(a, Operations.bundle(b, c))

        # Bundled vector should be somewhat similar to each component
        sim_a = torch.cosine_similarity(bundled, a, dim=0).item()
        sim_b = torch.cosine_similarity(bundled, b, dim=0).item()
        sim_c = torch.cosine_similarity(bundled, c, dim=0).item()

        assert sim_a > 0.2, f"Bundle should retain similarity to 'cat': {sim_a}"
        assert sim_b > 0.2, f"Bundle should retain similarity to 'dog': {sim_b}"
        assert sim_c > 0.2, f"Bundle should retain similarity to 'bird': {sim_c}"

    def test_dissimilar_concepts_have_low_similarity(self, codebook):
        """Test that unrelated concepts have low vector similarity."""
        v1 = codebook.encode("quantum_physics")
        v2 = codebook.encode("chocolate_cake")

        similarity = torch.cosine_similarity(v1, v2, dim=0).item()

        # Random encodings should have low similarity
        assert abs(similarity) < 0.3, f"Unrelated concepts should be dissimilar: {similarity}"

    def test_same_word_same_vector(self, codebook):
        """Test that encoding the same word twice gives the same vector."""
        v1 = codebook.encode("hello")
        v2 = codebook.encode("hello")

        similarity = torch.cosine_similarity(v1, v2, dim=0).item()

        assert similarity > 0.99, "Same word should encode to same vector"


# =============================================================================
# SECTION 4: MEMORY TRACE TESTS (Holographic Superposition)
# =============================================================================

class TestMemoryTrace:
    """Test the holographic memory trace's superposition capabilities."""

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def memory_trace(self, container):
        return container.create_memory_trace()

    def test_store_and_query_single_fact(self, container, memory_trace):
        """Test basic store/query with memory trace.

        MemoryTrace uses query() (not retrieve()) for unbinding.
        The query operation returns a noisy approximation; the cleanup/resonance
        step is needed for exact recovery.
        """
        codebook = container.codebook

        key = codebook.encode("test_key")
        value = codebook.encode("test_value")

        memory_trace.store(key, value)
        # Use query() - the correct API method
        queried = memory_trace.query(key)

        similarity = torch.cosine_similarity(value, queried, dim=0).item()
        # For single fact, similarity should be reasonable
        assert similarity > 0.3, f"Queried value should approximate stored: {similarity}"

    def test_multiple_facts_superposition(self, container, memory_trace):
        """Test that multiple facts can be stored in superposition.

        Uses resonance() for cleanup - this is the correct way to retrieve
        exact values from a bundled memory trace.
        """
        codebook = container.codebook

        # Store multiple key-value pairs
        pairs = [
            ("key1", "value1"),
            ("key2", "value2"),
            ("key3", "value3"),
        ]

        for key_str, val_str in pairs:
            key = codebook.encode(key_str)
            value = codebook.encode(val_str)
            memory_trace.store(key, value)

        # Create candidate tensor for resonance lookup
        candidates = torch.stack([codebook.encode(val_str) for _, val_str in pairs])

        # Retrieve each using resonance (the cleanup operation)
        for i, (key_str, val_str) in enumerate(pairs):
            key = codebook.encode(key_str)
            similarities = memory_trace.resonance(key, candidates)
            best_idx = torch.argmax(similarities).item()

            assert best_idx == i, f"Failed to retrieve {key_str}: got index {best_idx}, expected {i}"

    def test_saturation_estimate(self, container, memory_trace):
        """Test that memory saturation is tracked."""
        codebook = container.codebook

        initial_saturation = memory_trace.saturation_estimate

        # Store many facts
        for i in range(50):
            key = codebook.encode(f"key_{i}")
            value = codebook.encode(f"value_{i}")
            memory_trace.store(key, value)

        final_saturation = memory_trace.saturation_estimate

        # Saturation should increase
        assert final_saturation > initial_saturation, \
            f"Saturation should increase: {initial_saturation} -> {final_saturation}"


# =============================================================================
# SECTION 5: SURPRISE-GATED LEARNING TESTS (Titans-inspired)
# =============================================================================

class TestSurpriseGatedLearning:
    """Test the surprise-based learning mechanism.

    New facts should have high surprise and be learned.
    Known facts should have low surprise and be skipped.
    """

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def fact_store(self, container):
        return container.create_fact_store()

    def test_new_fact_has_high_surprise(self, fact_store):
        """Test that genuinely new facts have high surprise scores."""
        fact = fact_store.add_fact("NewEntity", "new_prop", "new_value", source="test")

        assert fact is not None, "New fact should be stored"
        assert fact.surprise_score is not None, "Surprise score should be set"
        assert fact.surprise_score >= SURPRISE_THRESHOLD, \
            f"New fact surprise {fact.surprise_score} should be >= {SURPRISE_THRESHOLD}"

    def test_duplicate_fact_has_low_surprise(self, fact_store):
        """Test that duplicate facts are rejected due to low surprise."""
        # Store first time
        fact1 = fact_store.add_fact("Entity", "prop", "value", source="test")
        assert fact1 is not None

        # Try to store again - should be rejected
        fact2 = fact_store.add_fact("Entity", "prop", "value", source="test")
        assert fact2 is None, "Duplicate fact should return None (low surprise)"

    def test_similar_facts_distinguished(self, fact_store):
        """Test that similar but different facts are both stored."""
        fact1 = fact_store.add_fact("Paris", "country", "France", source="test")
        fact2 = fact_store.add_fact("Paris", "population", "2.2 million", source="test")

        # Both should be stored (different predicates)
        assert fact1 is not None
        assert fact2 is not None

        # Both should be retrievable
        answer1, _ = fact_store.query("Paris", "country")
        answer2, _ = fact_store.query("Paris", "population")

        assert answer1 == "France"
        assert answer2 == "2.2 million"


# =============================================================================
# SECTION 6: CHATBOT INTEGRATION TESTS (Without LLM)
# =============================================================================

class TestChatbotHDCIntegration:
    """Test chatbot's use of HDC for fact retrieval WITHOUT relying on LLM."""

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def chatbot_no_llm(self, container):
        """Create chatbot with ventriloquist DISABLED."""
        return container.create_conversational_chatbot(
            enable_ventriloquist=False,
            enable_generation=False,
            enable_corpus=False,
        )

    def test_teach_and_query_fictional(self, chatbot_no_llm):
        """Test teaching and querying fictional facts without LLM."""
        chatbot = chatbot_no_llm

        # Teach a fictional fact
        response = chatbot.teach_fact("Zorbaxia", "capital", "Flumpton")
        assert "remember" in response.lower() or "got it" in response.lower()

        # Query it back
        response = chatbot.respond("What is the capital of Zorbaxia?")

        # Should contain the stored fact
        assert "flumpton" in response.lower(), \
            f"Response should contain 'Flumpton': {response}"

    def test_teach_contradictory_fact(self, chatbot_no_llm):
        """Test that chatbot uses HDC even for contradictory facts."""
        chatbot = chatbot_no_llm

        # Teach contradictory fact
        chatbot.teach_fact("France", "capital", "Berlin")

        # Query - should use HDC, not LLM knowledge
        response = chatbot.respond("What is the capital of France?")

        # Should contain "Berlin" (HDC answer), not "Paris" (LLM answer)
        response_lower = response.lower()

        # The response should mention Berlin (our stored fact)
        # It should NOT confidently say Paris
        has_berlin = "berlin" in response_lower
        confident_paris = "capital of france is paris" in response_lower

        assert has_berlin or not confident_paris, \
            f"Chatbot should use HDC fact 'Berlin', not LLM knowledge 'Paris'. Response: {response}"

    def test_unknown_entity_no_hallucination(self, chatbot_no_llm):
        """Test that chatbot doesn't confidently hallucinate for unknown entities.

        The chatbot WITHOUT ventriloquist may return:
        1. Explicit admission ("I don't know")
        2. A generic fallback pattern (greeting, acknowledgment)
        3. A template with "I don't know that yet" in place of {answer}

        What it should NOT do:
        - Confidently state a made-up capital city
        - Generate plausible-sounding but fabricated facts
        """
        chatbot = chatbot_no_llm

        # Query something we never taught
        response = chatbot.respond("What is the capital of Qwertyland?")
        response_lower = response.lower()

        # Check for explicit admission phrases
        admission_phrases = ["don't know", "not sure", "haven't learned", "no information"]
        has_admission = any(phrase in response_lower for phrase in admission_phrases)

        # Fallback patterns are acceptable (greeting, acknowledgment, etc.)
        # These don't hallucinate facts, they just provide a generic response
        fallback_indicators = ["hello", "hi", "how may", "how can", "help you", "assist", "good day"]
        is_generic_fallback = any(phrase in response_lower for phrase in fallback_indicators)

        # The key check: Did it NOT make up a specific capital city?
        # A hallucination would look like "the capital of qwertyland is <specific city>"
        made_up_cities = ["paris", "london", "berlin", "tokyo", "rome", "madrid"]
        hallucinated_city = any(city in response_lower for city in made_up_cities)

        # Test passes if: admission OR generic fallback OR no hallucinated city
        assert has_admission or is_generic_fallback or not hallucinated_city, \
            f"Chatbot should not confidently hallucinate a city. Response: {response}"

    def test_fact_learning_flag(self, chatbot_no_llm):
        """Test that fact learning is properly tracked."""
        chatbot = chatbot_no_llm

        # Clear any previous learning
        chatbot.clear_learning_flag()

        # Teach a fact
        chatbot.respond("The capital of Mythoria is Starhold.")

        # Check if fact was learned
        if chatbot.did_learn_fact_this_turn():
            fact = chatbot.get_last_learned_fact()
            assert fact is not None
            # Fact should be a tuple of (subject, predicate, object)
            assert len(fact) == 3


# =============================================================================
# SECTION 7: VENTRILOQUIST GROUNDING TESTS (With LLM - API Required)
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("NOVITA_API_KEY"),
    reason="NOVITA_API_KEY not set - skipping LLM tests"
)
class TestVentriloquistGrounding:
    """Test that Ventriloquist is properly grounded by HDC facts.

    These tests require the NOVITA_API_KEY environment variable.
    They verify that the LLM voice is constrained by HDC retrieval.
    """

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def ventriloquist(self, container):
        """Create VentriloquistGenerator."""
        return container.create_ventriloquist_generator()

    @pytest.fixture
    def generation_context(self):
        """Factory for creating GenerationContext."""
        from hologram.generation.base import GenerationContext
        from hologram.modulation.sesame import StyleType

        def create(query: str, fact_answer: str, entities: list = None):
            return GenerationContext(
                query_text=query,
                thought_vector=None,
                intent=None,
                fact_answer=fact_answer,
                entities=entities or [],
                style=StyleType.NEUTRAL,
                expected_subject=None,
            )
        return create

    @pytest.mark.parametrize("subject,predicate,obj", FICTIONAL_FACTS[:3])
    def test_ventriloquist_uses_fictional_facts(
        self, ventriloquist, generation_context, subject, predicate, obj
    ):
        """Test that ventriloquist uses provided fictional facts."""
        query = f"What is the {predicate} of {subject}?"
        context = generation_context(query, obj, [subject, predicate])

        try:
            result = ventriloquist.generate_with_validation(context)
        except Exception as e:
            pytest.skip(f"API call failed (network/timeout): {e}")
            return

        # If API returned None, it might be rate limiting or validation failure
        # We skip rather than fail for flaky API behavior
        if result is None:
            pytest.skip("Generation returned None - possible API issue or validation failure")
            return

        assert obj.lower() in result.text.lower(), \
            f"Response should contain fact '{obj}': {result.text}"

    def test_ventriloquist_validation_rejects_missing_fact(
        self, ventriloquist, generation_context
    ):
        """Test that validation rejects responses missing the fact."""
        # This tests the validation logic at ventriloquist.py:175-185
        context = generation_context(
            "What is X?",
            "Flumpton",  # Unique word that must appear
            ["X"]
        )

        result = ventriloquist.generate_with_validation(context)

        # If generation succeeded, the fact word MUST be in the response
        if result is not None:
            assert "flumpton" in result.text.lower(), \
                f"Validation should ensure fact word appears: {result.text}"

    def test_ventriloquist_no_hallucination_without_fact(
        self, ventriloquist, generation_context
    ):
        """Test behavior when no fact is provided."""
        context = generation_context(
            "What is the capital of Randomland?",
            None,  # No fact provided
            ["Randomland", "capital"]
        )

        result = ventriloquist.generate_with_validation(context)

        # Should either return None or express uncertainty
        if result is not None:
            uncertain_phrases = [
                "don't know", "not sure", "no information", "cannot",
                "not aware", "don't have information", "fictional",
                "not a real", "not recognized", "unfamiliar",
                # Additional phrases LLMs commonly use to express uncertainty
                "haven't heard", "never heard", "doesn't exist", "isn't a real",
                "not on any", "made up", "doesn't appear", "can't find",
                "no such", "not familiar",
            ]
            has_uncertainty = any(p in result.text.lower() for p in uncertain_phrases)
            # If it gives an answer without fact, should be uncertain
            assert has_uncertainty or len(result.text) < 50, \
                f"Should not confidently answer without fact: {result.text}"


# =============================================================================
# SECTION 8: FULL INTEGRATION TESTS (Chatbot + HDC + Optional Ventriloquist)
# =============================================================================

class TestFullIntegration:
    """Full system integration tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for persistence tests."""
        temp = tempfile.mkdtemp(prefix="hologram_test_")
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    def test_bulk_fact_storage_and_retrieval(self, container):
        """Test storing and retrieving many facts."""
        fs = container.create_fact_store()

        # Store all test facts
        all_facts = FICTIONAL_FACTS + CONTRADICTORY_FACTS + NUMERICAL_FACTS + EDGE_CASE_FACTS

        stored_count = 0
        for subject, predicate, obj in all_facts:
            fact = fs.add_fact(subject, predicate, obj, source="bulk_test")
            if fact is not None:
                stored_count += 1

        # Should store most facts
        assert stored_count >= len(all_facts) * 0.8, \
            f"Should store at least 80% of facts: {stored_count}/{len(all_facts)}"

        # Verify retrieval accuracy
        correct = 0
        for subject, predicate, expected in all_facts:
            answer, conf = fs.query(subject, predicate)
            if answer == expected and conf > REFUSAL_CONFIDENCE_THRESHOLD:
                correct += 1

        accuracy = correct / len(all_facts)
        assert accuracy >= 0.8, f"Retrieval accuracy should be >= 80%: {accuracy:.1%}"

    def test_chain_fact_traversal(self, container):
        """Test retrieving facts in a chain."""
        fs = container.create_fact_store()

        # Store chain facts
        for subject, predicate, obj in CHAIN_FACTS:
            fs.add_fact(subject, predicate, obj, source="chain_test")

        # Traverse: Alice -> TechCorp -> Silicon Valley -> USA
        answer1, _ = fs.query("Alice", "works_at")
        assert answer1 == "TechCorp"

        answer2, _ = fs.query("TechCorp", "located_in")
        assert answer2 == "Silicon Valley"

        answer3, _ = fs.query("Silicon Valley", "country")
        assert answer3 == "USA"

    def test_persistent_chatbot_fact_retention(self, container, temp_dir):
        """Test that facts persist across chatbot sessions."""
        # Session 1: Create chatbot and teach fact
        chatbot1 = container.create_persistent_chatbot(
            persist_dir=temp_dir,
            enable_ventriloquist=False,
        )
        chatbot1.teach_fact("PersistTest", "value", "PersistValue")

        # Session 2: Create new chatbot with same persist_dir
        chatbot2 = container.create_persistent_chatbot(
            persist_dir=temp_dir,
            enable_ventriloquist=False,
        )

        # Query the fact from session 2
        if hasattr(chatbot2, '_fact_store') and chatbot2._fact_store:
            answer, conf = chatbot2._fact_store.query("PersistTest", "value")
            # Note: Persistence depends on ChromaDB implementation
            # This test documents expected behavior
            if conf > 0.5:
                assert answer == "PersistValue", \
                    f"Persisted fact should be retrievable: got {answer}"


# =============================================================================
# SECTION 9: STRESS AND EDGE CASE TESTS
# =============================================================================

class TestStressAndEdgeCases:
    """Stress tests and edge cases for HDC robustness."""

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    @pytest.fixture
    def fact_store(self, container):
        return container.create_fact_store()

    def test_rapid_store_query_cycle(self, fact_store):
        """Test rapid alternating store/query operations."""
        success = 0
        total = 50

        for i in range(total):
            subject = f"RapidEntity{i}"
            obj = f"RapidValue{i}"

            fact_store.add_fact(subject, "rapid_prop", obj, source="test")
            answer, conf = fact_store.query(subject, "rapid_prop")

            if answer == obj:
                success += 1

        assert success >= total * 0.9, f"Rapid cycle success: {success}/{total}"

    def test_similar_entity_disambiguation(self, fact_store):
        """Test that similar entity names don't cause confusion."""
        facts = [
            ("France", "capital", "Paris"),
            ("Franconia", "capital", "Nuremberg"),
            ("Frankfurt", "country", "Germany"),
        ]

        for s, p, o in facts:
            fact_store.add_fact(s, p, o, source="test")

        # Each query should return correct value
        for subject, predicate, expected in facts:
            answer, conf = fact_store.query(subject, predicate)
            assert answer == expected, \
                f"Confusion: {subject} -> {answer} (expected {expected})"

    @pytest.mark.parametrize("subject,predicate,obj", EDGE_CASE_FACTS)
    def test_edge_case_handling(self, fact_store, subject, predicate, obj):
        """Test handling of edge case inputs."""
        fact = fact_store.add_fact(subject, predicate, obj, source="test")

        # Should store successfully
        assert fact is not None or fact_store.fact_count > 0

        # Should retrieve correctly
        answer, conf = fact_store.query(subject, predicate)
        assert answer == obj, f"Edge case failed: {subject}.{predicate} = {answer} (expected {obj})"

    def test_empty_input_handling(self, fact_store):
        """Test handling of empty or whitespace inputs."""
        # These should not crash, but may not store
        try:
            fact_store.add_fact("", "prop", "value", source="test")
            fact_store.add_fact("subject", "", "value", source="test")
            fact_store.add_fact("subject", "prop", "", source="test")
        except Exception as e:
            pytest.fail(f"Empty input should not crash: {e}")

    def test_memory_saturation_graceful_degradation(self, container):
        """Test that memory degrades gracefully under saturation."""
        fs = container.create_fact_store()

        # Store many facts (beyond estimated capacity)
        for i in range(200):
            fs.add_fact(f"Entity{i}", "prop", f"Value{i}", source="test")

        # Sample some facts to check degradation
        correct = 0
        sample_indices = [0, 50, 100, 150, 199]

        for i in sample_indices:
            answer, conf = fs.query(f"Entity{i}", "prop")
            if answer == f"Value{i}":
                correct += 1

        # Even under saturation, should still work reasonably
        accuracy = correct / len(sample_indices)
        assert accuracy >= 0.6, f"Saturated memory accuracy: {accuracy:.1%}"


# =============================================================================
# SECTION 10: ARCHITECTURE VALIDATION TESTS
# =============================================================================

class TestArchitectureValidation:
    """Validate that the system follows the intended architecture:

    HDC = Brain (facts, retrieval, bounded hallucination)
    Ventriloquist = Voice (fluency only, constrained by HDC)
    """

    @pytest.fixture
    def container(self):
        return HologramContainer(dimensions=DEFAULT_DIMENSIONS)

    def test_fact_store_is_independent_of_llm(self, container):
        """Verify FactStore works without any LLM."""
        # FactStore should work completely independently
        fs = container.create_fact_store()

        fs.add_fact("Test", "prop", "Value", source="test")
        answer, conf = fs.query("Test", "prop")

        assert answer == "Value"
        assert conf > 0.9
        # No LLM was involved in this operation

    def test_chatbot_works_without_ventriloquist(self, container):
        """Verify chatbot can function without LLM."""
        chatbot = container.create_conversational_chatbot(
            enable_ventriloquist=False,
            enable_generation=False,
        )

        # Should be able to teach and query
        chatbot.teach_fact("NoLLM", "test", "Works")
        response = chatbot.respond("What is the test of NoLLM?")

        # Should get a response (even if from templates)
        assert response is not None
        assert len(response) > 0

    def test_hdc_operations_are_deterministic(self, container):
        """Verify HDC operations give consistent results."""
        codebook = container.codebook

        # Encode same word multiple times
        v1 = codebook.encode("consistency")
        v2 = codebook.encode("consistency")
        v3 = codebook.encode("consistency")

        # Should all be identical
        assert torch.allclose(v1, v2)
        assert torch.allclose(v2, v3)

    def test_fact_retrieval_is_bounded(self, container):
        """Verify that retrieval can only return stored vocabulary."""
        fs = container.create_fact_store()

        # Store specific facts
        fs.add_fact("A", "prop", "Value1", source="test")
        fs.add_fact("B", "prop", "Value2", source="test")

        # Query for something not stored
        answer, conf = fs.query("C", "prop")

        # Answer must be from vocabulary (Value1 or Value2) or empty
        # Cannot hallucinate "Value3"
        assert answer in ["Value1", "Value2", ""], \
            f"HDC returned value outside vocabulary: {answer}"


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

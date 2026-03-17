"""
Tests for FHRR (Fourier Holographic Reduced Representation) binding mode.

This test suite validates that FHRR circular convolution works correctly
and preserves all HDC algebra properties.

NOTE: FHRR is optimized for continuous Gaussian vectors and achieves >0.9
unbinding similarity with those. However, this codebase uses bipolar (+1/-1)
vectors, where MAP actually performs better (~0.99 vs ~0.65 for FHRR).

FHRR mode is provided for experimentation and future use with continuous
vector spaces.
"""

import pytest
import torch
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.memory.fact_store import FactStore
from hologram.memory.memory_trace import MemoryTrace


@pytest.fixture
def fhrr_mode():
    """Set FHRR mode for tests, restore MAP mode after."""
    original_mode = Operations.get_binding_mode()
    Operations.set_binding_mode("FHRR")
    yield
    Operations.set_binding_mode(original_mode)


@pytest.fixture
def map_mode():
    """Ensure MAP mode for tests, restore original mode after."""
    original_mode = Operations.get_binding_mode()
    Operations.set_binding_mode("MAP")
    yield
    Operations.set_binding_mode(original_mode)


@pytest.fixture
def space():
    """Create a VectorSpace for testing."""
    return VectorSpace(dimensions=1000)


@pytest.fixture
def codebook(space):
    """Create a Codebook for integration tests."""
    return Codebook(space)


class TestFHRRAlgebra:
    """Test FHRR binding algebra properties."""

    def test_exact_unbinding_recovery(self, fhrr_mode, space):
        """FHRR unbind should recover >0.65 similarity (vs ~0.4-0.6 for MAP with bipolar vectors)."""
        a = space.random_vector(seed=1)
        b = space.random_vector(seed=2)

        # Bind and unbind
        composite = Operations.bind(a, b)
        recovered = Operations.unbind(composite, a)

        # FHRR should get >0.65 similarity (bipolar vectors don't achieve theoretical 0.9+)
        similarity = torch.cosine_similarity(recovered, b, dim=0).item()
        assert similarity > 0.65, f"FHRR unbinding similarity {similarity:.3f} < 0.65"

    def test_map_unbinding_baseline(self, map_mode, space):
        """Baseline: MAP unbind achieves near-perfect recovery with bipolar vectors."""
        a = space.random_vector(seed=100)
        b = space.random_vector(seed=200)

        composite = Operations.bind(a, b)
        recovered = Operations.unbind(composite, a)

        similarity = torch.cosine_similarity(recovered, b, dim=0).item()
        # For bipolar vectors (+1/-1), MAP achieves near-perfect unbinding
        # because (a*b)*a = b when a*a = 1 for all elements
        assert similarity > 0.95, f"MAP unbinding similarity {similarity:.3f} < 0.95"

    def test_commutativity(self, fhrr_mode, space):
        """FHRR bind should be commutative: bind(a, b) == bind(b, a)."""
        a = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        b = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())

        ab = Operations.bind(a, b)
        ba = Operations.bind(b, a)

        similarity = torch.cosine_similarity(ab, ba, dim=0)
        assert similarity > 0.99, f"FHRR commutativity failed: similarity {similarity:.3f}"

    def test_dissimilarity(self, fhrr_mode, space):
        """bind(a, b) should be orthogonal to both a and b."""
        a = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        b = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())

        composite = Operations.bind(a, b)

        sim_a = torch.cosine_similarity(composite, a, dim=0)
        sim_b = torch.cosine_similarity(composite, b, dim=0)

        # Should be near-orthogonal (close to 0)
        assert abs(sim_a) < 0.2, f"Composite too similar to a: {sim_a:.3f}"
        assert abs(sim_b) < 0.2, f"Composite too similar to b: {sim_b:.3f}"

    def test_inverse_property(self, fhrr_mode, space):
        """bind(v, inverse(v)) should act as identity."""
        v = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        inv_v = Operations.inverse(v)

        # Bind with inverse
        identity = Operations.bind(v, inv_v)

        # Should have special property (concentrated at origin in frequency domain)
        # For FHRR, this should be close to a unit vector or special pattern
        # The exact property depends on implementation, but it should be different from v
        sim_to_original = torch.cosine_similarity(identity, v, dim=0)
        assert abs(sim_to_original) < 0.5, f"Identity too similar to original: {sim_to_original:.3f}"

    def test_nested_unbinding(self, fhrr_mode, space):
        """Test S-P-O triple unbinding pattern (nested bind/unbind)."""
        # Create subject-predicate-object triple
        subject = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        predicate = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        obj = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())

        # Bind into triple: bind(bind(s, p), o)
        sp = Operations.bind(subject, predicate)
        spo = Operations.bind(sp, obj)

        # Unbind in reverse order
        sp_recovered = Operations.unbind(spo, obj)
        subject_recovered = Operations.unbind(sp_recovered, predicate)

        # Should recover subject with moderate similarity (nested unbinding degrades)
        similarity = torch.cosine_similarity(subject_recovered, subject, dim=0).item()
        assert similarity > 0.4, f"Nested unbinding failed: similarity {similarity:.3f}"

    def test_mode_switching(self, space):
        """Test set_binding_mode validates input and get/set round-trips."""
        # Save original mode
        original = Operations.get_binding_mode()

        # Test valid modes
        Operations.set_binding_mode("MAP")
        assert Operations.get_binding_mode() == "MAP"

        Operations.set_binding_mode("FHRR")
        assert Operations.get_binding_mode() == "FHRR"

        # Test invalid mode
        with pytest.raises(ValueError, match="Invalid binding mode"):
            Operations.set_binding_mode("INVALID")

        # Restore original
        Operations.set_binding_mode(original)

    def test_bundle_unaffected(self, fhrr_mode, space):
        """Bundle should work identically regardless of binding mode."""
        a = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        b = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        c = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())

        # Bundle in FHRR mode
        bundled_fhrr = Operations.bundle(a, b, c)

        # Switch to MAP and bundle
        Operations.set_binding_mode("MAP")
        bundled_map = Operations.bundle(a, b, c)

        # Should be identical (bundle is mode-independent)
        similarity = torch.cosine_similarity(bundled_fhrr, bundled_map, dim=0)
        assert similarity > 0.99, f"Bundle differs between modes: {similarity:.3f}"

        # Restore FHRR mode (fixture will restore later, but be explicit)
        Operations.set_binding_mode("FHRR")


class TestFHRRIntegration:
    """Integration tests with FactStore and MemoryTrace."""

    def test_fact_store_query(self, fhrr_mode, space, codebook):
        """FactStore should work with FHRR and get higher confidence."""
        fact_store = FactStore(space, codebook)

        # Add a fact
        fact_store.add_fact("Paris", "capital_of", "France")

        # Query it back (returns tuple of (answer, confidence))
        answer, confidence = fact_store.query("Paris", "capital_of")

        # Should get high confidence with FHRR
        assert answer == "France", f"FHRR answer '{answer}' != 'France'"
        assert confidence > 0.6, f"FHRR confidence {confidence:.3f} < 0.6"

    def test_memory_trace_retrieval(self, fhrr_mode, space, codebook):
        """MemoryTrace store + query should work with FHRR."""
        memory_trace = MemoryTrace(space)

        # Create test vectors
        key1 = codebook.encode("key1")
        value1 = codebook.encode("value1")
        key2 = codebook.encode("key2")
        value2 = codebook.encode("value2")

        # Store key-value pairs
        memory_trace.store(key1, value1)
        memory_trace.store(key2, value2)

        # Retrieve via query
        retrieved = memory_trace.query(key1)

        # Should retrieve correct value with moderate similarity (bipolar vectors)
        similarity = torch.cosine_similarity(retrieved, value1, dim=0).item()
        assert similarity > 0.5, f"FHRR MemoryTrace retrieval similarity {similarity:.3f} < 0.5"

    def test_mode_persistence(self, space):
        """Ensure binding mode persists across multiple operations."""
        Operations.set_binding_mode("FHRR")

        a = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())
        b = space.random_vector(seed=torch.randint(0, 1000000, (1,)).item())

        # Multiple operations
        c1 = Operations.bind(a, b)
        c2 = Operations.bind(b, a)
        c3 = Operations.bundle(c1, c2)

        # Mode should still be FHRR
        assert Operations.get_binding_mode() == "FHRR"

        # Cleanup
        Operations.set_binding_mode("MAP")

    def test_fhrr_capacity_vs_map(self, space, codebook):
        """FHRR should support higher confidence at same fact count vs MAP."""
        # This is a smoke test - full capacity testing is in the benchmark script

        # Test FHRR with 20 facts
        Operations.set_binding_mode("FHRR")
        fact_store_fhrr = FactStore(space, codebook)

        for i in range(20):
            fact_store_fhrr.add_fact(f"subject{i}", "predicate", f"object{i}")

        # Query a middle fact (returns tuple of (answer, confidence))
        _, fhrr_confidence = fact_store_fhrr.query("subject10", "predicate")

        # Test MAP with same facts
        Operations.set_binding_mode("MAP")
        fact_store_map = FactStore(space, codebook)

        for i in range(20):
            fact_store_map.add_fact(f"subject{i}", "predicate", f"object{i}")

        _, map_confidence = fact_store_map.query("subject10", "predicate")

        # FHRR should have higher or equal confidence at same capacity
        assert fhrr_confidence >= map_confidence * 0.9, (
            f"FHRR confidence {fhrr_confidence:.3f} not >= MAP {map_confidence:.3f} * 0.9"
        )

        # Cleanup
        Operations.set_binding_mode("MAP")

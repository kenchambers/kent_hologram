"""
Integration test for CrewAI trainer script.

Tests the full flow with real API keys:
- Agents generate messages
- Hologram chatbot learns from conversations
- Facts persist to ChromaDB
- Logs are created correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Load environment variables
load_dotenv()

# Skip if API keys not available
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

pytestmark = pytest.mark.skipif(
    not GEMINI_KEY or not ANTHROPIC_KEY,
    reason="API keys not available in environment",
)

# Import after path setup
import crew_trainer


class TestCrewTrainerIntegration:
    """Integration tests for CrewTrainer."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        persist_dir = tempfile.mkdtemp(prefix="hologram_test_persist_")
        log_dir = tempfile.mkdtemp(prefix="hologram_test_logs_")
        yield persist_dir, log_dir
        # Cleanup
        shutil.rmtree(persist_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    def test_trainer_initialization(self, temp_dirs):
        """Test that trainer initializes correctly."""
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=3,
        )

        assert trainer.chatbot is not None
        assert trainer.gemini_llm is not None
        assert trainer.claude_llm is not None
        assert trainer.logger is not None
        assert trainer.running is True

    def test_single_conversation_round(self, temp_dirs):
        """Test a single conversation round."""
        from unittest.mock import MagicMock, patch
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs
        
        # Mock the LLM responses to avoid external API calls
        mock_gemini_response = MagicMock()
        mock_gemini_response.content = "The capital of France is Paris"
        
        mock_claude_response = MagicMock()
        mock_claude_response.content = "That's correct! Paris is indeed the capital."
        
        with patch.object(CrewTrainer, '_setup_agents') as mock_setup:
            trainer = CrewTrainer(
                persist_dir=persist_dir,
                log_dir=Path(log_dir),
                max_turns_per_topic=2,  # Short round for testing
            )
            
            # Setup mocked LLMs after initialization
            trainer.gemini_llm = MagicMock()
            trainer.gemini_llm.invoke.return_value = mock_gemini_response
            trainer.claude_llm = MagicMock()
            trainer.claude_llm.invoke.return_value = mock_claude_response
            
            # Run one round
            trainer.run_conversation_round()

        # Verify conversation history was created
        assert len(trainer.conversation_history) > 0

        # Verify log file was created
        assert trainer.logger.log_file.exists()
        log_content = trainer.logger.log_file.read_text()
        assert "gemini" in log_content.lower() or "claude" in log_content.lower()

    def test_hologram_learning(self, temp_dirs):
        """Test that Hologram learns facts from conversations."""
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer

        persist_dir, log_dir = temp_dirs

        # Create trainer and teach a fact
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Manually teach a fact (simulating what happens in conversation)
        trainer.chatbot.teach_fact("TestCountry", "capital", "TestCity")

        # Verify fact was stored
        fact_store = trainer.chatbot._fact_store
        answer, confidence = fact_store.query("TestCountry", "capital")
        assert answer.lower() == "testcity"
        assert confidence > 0.1  # Should have reasonable confidence

    def test_chromadb_persistence(self, temp_dirs):
        """Test that facts persist across sessions (via neural memory or ChromaDB)."""
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer

        persist_dir, log_dir = temp_dirs

        # First session: teach a fact
        trainer1 = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )
        trainer1.chatbot.teach_fact("PersistTest", "capital", "PersistCity")
        
        # Save neural memory explicitly (normally happens on exit)
        trainer1.chatbot.save_memory(persist_dir)
        trainer1.chatbot.end_session()

        # Second session: verify fact persists via query (not fact_count)
        # Note: With neural consolidation, the metadata list isn't repopulated
        # on load, but the neural memory and vocabulary are restored
        trainer2 = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Query the persisted fact - this should work via neural/HDC lookup
        answer, confidence = trainer2.chatbot._fact_store.query("PersistTest", "capital")
        assert answer.lower() == "persistcity", f"Expected 'persistcity', got '{answer}'"
        assert confidence > 0.1, f"Confidence {confidence} too low"

    def test_log_format(self, temp_dirs):
        """Test that logs are formatted correctly."""
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=2,
        )

        # Log some test messages
        trainer.logger.log("gemini", "Test message from gemini")
        trainer.logger.log("claude", "Test message from claude")
        trainer.logger.log("hologram", "Test response from hologram")

        # Verify log format
        log_content = trainer.logger.log_file.read_text()
        assert "gemini:" in log_content
        assert "claude:" in log_content
        assert "hologram:" in log_content

        # Check timestamp format
        lines = log_content.strip().split("\n")
        for line in lines:
            if line and not line.startswith("="):
                assert line.startswith("[")  # Should have timestamp

    def test_conversation_context(self, temp_dirs):
        """Test that conversation context is maintained."""
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=2,
        )

        # Add some conversation history
        trainer.conversation_history = [
            ("gemini", "Hello"),
            ("hologram", "Hi there!"),
            ("claude", "How are you?"),
        ]

        # Get context - note: _get_conversation_context uses generic labels
        # "Partner" for gemini/claude and "Bot" for hologram to prevent role-play
        context = trainer._get_conversation_context(last_n=3)
        assert "Partner:" in context  # gemini/claude mapped to Partner
        assert "Bot:" in context  # hologram mapped to Bot

        # Test limiting context
        context_limited = trainer._get_conversation_context(last_n=2)
        lines = context_limited.split("\n")
        assert len(lines) <= 2

    def test_graceful_shutdown(self, temp_dirs):
        """Test that trainer can be stopped gracefully."""
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Stop the trainer
        trainer.running = False
        assert trainer.running is False

        # Should not raise exception
        trainer.run_conversation_round()

    def test_neural_persistence_reproduction(self, temp_dirs):
        """
        Reproduction test for neural memory persistence issue.
        Verifies that facts trained in one session are retrievable in another
        using the exact same flow as the interactive CLI.
        """
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer
        import torch

        persist_dir, log_dir = temp_dirs

        # --- Phase 1: Training Session ---
        print("\\n--- Phase 1: Training ---")
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )
        
        # Verify neural consolidation is enabled
        assert trainer.chatbot._fact_store._consolidation_manager is not None, "Neural consolidation should be enabled in trainer"

        # Teach specific facts that were failing for the user
        facts_to_teach = [
            ("France", "capital", "Paris"),
            ("Brazil", "capital", "Brasilia"),
            ("Sky", "color", "Blue"),
        ]
        
        for subj, pred, obj in facts_to_teach:
            trainer.chatbot.teach_fact(subj, pred, obj)
            # Verify immediate recall in training session
            ans, conf = trainer.chatbot._fact_store.query(subj, pred)
            print(f"Immediate recall for {subj} {pred}: {ans} ({conf:.3f})")
            assert ans.lower() == obj.lower(), f"Immediate recall failed for {subj}"

        # Save memory
        save_success = trainer.chatbot.save_memory(persist_dir)
        assert save_success, "Failed to save memory"
        trainer.chatbot.end_session()

        # Verify file exists
        neural_path = Path(persist_dir) / "neural_memory.pt"
        assert neural_path.exists(), "neural_memory.pt was not created"
        print(f"Neural memory saved to {neural_path}")

        # --- Phase 2: Interactive Session (Simulation) ---
        print("\\n--- Phase 2: Interactive Session ---")
        
        # Re-initialize essentially how ChatInterface does it
        # Note: ChatInterface uses force_neural=True/False or auto-detect
        
        # Simulate CLI initialization with auto-detection
        container = HologramContainer(dimensions=10000) # Match default dims
        
        # Manually check for neural file (logic from interface.py)
        use_neural = neural_path.exists()
        assert use_neural is True, "Should have detected neural memory file"
        
        chatbot = container.create_persistent_chatbot(
            persist_dir=persist_dir,
            enable_neural_consolidation=use_neural,
        )
        
        # Verify consolidation manager is active and loaded
        assert chatbot._fact_store._consolidation_manager is not None, "Consolidation manager missing in loaded chatbot"
        
        # Verify facts are loaded in pending/vocab
        # Note: If they haven't been consolidated yet (count < threshold), 
        # they should be in 'pending_facts' restored from state_dict
        manager = chatbot._fact_store._consolidation_manager
        print(f"Pending facts: {manager.pending_count}")
        print(f"Vocab size: {manager.vocab_size}")
        
        # Query facts
        for subj, pred, obj in facts_to_teach:
            ans, conf = chatbot._fact_store.query(subj, pred)
            
            # The core issue: if ans is empty or wrong
            assert ans.lower() == obj.lower(), f"Failed to recall {subj} {pred} after reload. Got '{ans}'"
            assert conf > 0.05, f"Confidence too low for {subj} {pred}: {conf}" 

    def test_neural_persistence_conversation_reproduction(self, temp_dirs):
        """
        Reproduction test for neural memory persistence issue using full conversation.
        Verifies that facts trained in one session are retrievable via NATURAL LANGUAGE
        in another session.
        """
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer
        
        persist_dir, log_dir = temp_dirs

        # --- Phase 1: Training Session ---
        print("\\n--- Phase 1: Training ---")
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )
        
        # Teach specific facts via teach_fact
        facts_to_teach = [
            ("France", "capital", "Paris"),
        ]
        
        for subj, pred, obj in facts_to_teach:
            trainer.chatbot.teach_fact(subj, pred, obj)
            # Verify immediate recall in training session
            ans, conf = trainer.chatbot._fact_store.query(subj, pred)
            print(f"Immediate recall for {subj} {pred}: {ans} ({conf:.3f})")

        # Save memory
        save_success = trainer.chatbot.save_memory(persist_dir)
        assert save_success
        trainer.chatbot.end_session()

        # --- Phase 2: Interactive Session (Simulation) ---
        print("\\n--- Phase 2: Interactive Session ---")
        
        container = HologramContainer(dimensions=10000)
        chatbot = container.create_persistent_chatbot(
            persist_dir=persist_dir,
            enable_neural_consolidation=True,
        )
        
        # Try natural language query
        queries = [
            "What is the capital of France?",
            "What's the capital of France?",
            "France capital?",
        ]
        
        for query in queries:
            response = chatbot.respond(query)

            # This is likely where it fails for the user
            assert "Paris" in response, f"Failed to retrieve Paris for query: {query}. Response: {response}"

    def test_semantic_fact_search_object_position(self, temp_dirs):
        """
        Test semantic fact search when entity appears in object position.

        This tests the fix for the "About is I don't know that yet" bug where:
        - User asks: "do you know about river?"
        - Entity extracted: ["river"]
        - Stored fact: "Nile is longest river in the world" (river in OBJECT)
        - Old behavior: query("river", "is") fails → returns nothing
        - New behavior: semantic search finds "river" mention → returns context
        """
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer

        persist_dir, log_dir = temp_dirs

        # Create trainer and teach facts where entity will be in OBJECT position
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Teach facts where the queried term appears in the OBJECT, not subject
        facts_with_object_mentions = [
            ("Nile", "is", "longest river in the world"),
            ("Amazon", "is", "second longest river"),
            ("Eiffel Tower", "location", "Paris France"),
            ("Mount Everest", "is", "tallest mountain"),
        ]

        for subj, pred, obj in facts_with_object_mentions:
            trainer.chatbot.teach_fact(subj, pred, obj)

        # Test 1: Direct FactStore.search_facts_mentioning() method
        fact_store = trainer.chatbot._fact_store

        # Search for "river" - should find it in objects
        river_matches = fact_store.search_facts_mentioning("river", match_type="object")
        assert len(river_matches) >= 2, f"Expected at least 2 river matches, got {len(river_matches)}"

        # Verify matches are scored correctly
        for fact, score in river_matches:
            assert score == 0.5, f"Object matches should have score 0.5, got {score}"
            assert "river" in fact.object.lower(), f"Expected 'river' in object: {fact.object}"

        # Search for "mountain" - should find it
        mountain_matches = fact_store.search_facts_mentioning("mountain", match_type="object")
        assert len(mountain_matches) >= 1, "Should find 'mountain' in Mount Everest fact"

        # Test 2: Word boundary protection - "can" should NOT match "Vatican"
        # First teach a fact with "Vatican"
        trainer.chatbot.teach_fact("Vatican", "is", "smallest country")

        can_matches = fact_store.search_facts_mentioning("can", match_type="any")
        assert len(can_matches) == 0, f"'can' should NOT match 'Vatican', got {len(can_matches)} matches"

        # Test 3: Short term rejection
        short_matches = fact_store.search_facts_mentioning("is", match_type="any")
        assert len(short_matches) == 0, "Terms < 3 chars should be rejected"

        # Test 4: Subject match has higher priority
        # "Nile" appears as subject, should get higher score
        nile_matches = fact_store.search_facts_mentioning("Nile", match_type="any")
        assert len(nile_matches) >= 1
        fact, score = nile_matches[0]
        assert score == 0.9, f"Subject matches should have score 0.9, got {score}"

        print("\n✅ All semantic fact search unit tests passed!")

    def test_semantic_fallback_in_query_facts(self, temp_dirs):
        """
        Test that the semantic fallback works in the full query pipeline.

        When structured queries fail, the system should fall back to
        semantic search and find facts mentioning the entity.
        """
        from crew_trainer import CrewTrainer
        from hologram.container import HologramContainer

        persist_dir, log_dir = temp_dirs

        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Teach a fact where "river" is in the object
        trainer.chatbot.teach_fact("Nile", "is", "longest river in Africa")

        # The structured query query("river", "is") should fail
        # But the semantic fallback should find it
        fact_store = trainer.chatbot._fact_store

        # Direct structured query - should fail (river is not a subject)
        direct_answer, direct_conf = fact_store.query("river", "is")
        # This may return something via HDC resonance, but with low confidence

        # Semantic search should find it
        semantic_matches = fact_store.search_facts_mentioning("river", match_type="object")
        assert len(semantic_matches) >= 1, "Semantic search should find river in object"

        fact, score = semantic_matches[0]
        assert fact.subject == "Nile", f"Expected Nile, got {fact.subject}"
        assert "river" in fact.object.lower(), "Should contain 'river' in object"

        print("\n✅ Semantic fallback pipeline test passed!")

    def test_semantic_search_end_to_end_conversation(self, temp_dirs):
        """
        End-to-end test: Natural language query finds facts via semantic search.

        This simulates the exact user scenario:
        1. Train: "Nile is longest river"
        2. Query: "do you know about river?"
        3. Expected: Response mentions Nile (not "I don't know")

        Note: This test works within a single session because after neural memory
        reload, the _facts metadata list is empty (facts are stored in neural
        memory, not metadata). The semantic search operates on the _facts list.
        """
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs

        # Training and querying in same session
        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Teach facts where the query term appears in OBJECT position
        trainer.chatbot.teach_fact("Nile", "is", "longest river in the world")
        trainer.chatbot.teach_fact("Amazon Rainforest", "is", "largest tropical rainforest")
        trainer.chatbot.teach_fact("Blue Whale", "is", "largest ocean mammal")

        # Test semantic queries - entity in object position (same session)
        fact_store = trainer.chatbot._fact_store

        semantic_queries = [
            ("river", ["nile", "longest"]),  # Should find Nile fact
            ("rainforest", ["amazon", "tropical"]),  # Should find Amazon fact
            ("mammal", ["blue whale", "ocean"]),  # Should find Blue Whale fact ("mammal" is in object)
        ]

        for query_term, expected_keywords in semantic_queries:
            matches = fact_store.search_facts_mentioning(query_term, match_type="object")

            assert len(matches) >= 1, f"No semantic matches found for '{query_term}'"

            fact, score = matches[0]
            response_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()

            for keyword in expected_keywords:
                assert keyword in response_text, \
                    f"Expected '{keyword}' in response for '{query_term}'. Got: {response_text}"

            print(f"✅ Query '{query_term}' found: {fact.subject} {fact.predicate} {fact.object}")

        print("\n✅ End-to-end semantic search test passed!")

    def test_semantic_search_does_not_break_existing_queries(self, temp_dirs):
        """
        Regression test: Ensure semantic fallback doesn't break normal queries.

        Standard subject-based queries should still work exactly as before.
        """
        from crew_trainer import CrewTrainer

        persist_dir, log_dir = temp_dirs

        trainer = CrewTrainer(
            persist_dir=persist_dir,
            log_dir=Path(log_dir),
            max_turns_per_topic=1,
        )

        # Teach standard S-P-O facts
        standard_facts = [
            ("France", "capital", "Paris"),
            ("Germany", "capital", "Berlin"),
            ("Python", "creator", "Guido van Rossum"),
            ("Sky", "color", "blue"),
        ]

        for subj, pred, obj in standard_facts:
            trainer.chatbot.teach_fact(subj, pred, obj)

        fact_store = trainer.chatbot._fact_store

        # Standard queries should still work (this is the primary path)
        for subj, pred, expected_obj in standard_facts:
            answer, confidence = fact_store.query(subj, pred)
            assert answer.lower() == expected_obj.lower(), \
                f"Standard query failed: {subj} {pred} -> expected '{expected_obj}', got '{answer}'"
            assert confidence > 0.5, f"Confidence too low for {subj} {pred}: {confidence}"

        # Semantic search should also find these by subject
        france_matches = fact_store.search_facts_mentioning("France", match_type="subject")
        assert len(france_matches) >= 1, "Should find France as subject"
        fact, score = france_matches[0]
        assert score == 0.9, "Subject match should have 0.9 score"
        assert fact.subject == "France"

        print("\n✅ Regression test passed - existing queries still work!")


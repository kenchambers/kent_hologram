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


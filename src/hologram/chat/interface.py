#!/usr/bin/env python3
"""
Chat interface for Hologram holographic memory system.

Provides an interactive REPL with two modes:
1. Conversational mode (default): Natural language chat with learning
2. Command mode: Direct fact store/query via slash commands
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file before other imports

from pathlib import Path
from typing import Optional

from hologram.container import HologramContainer
from hologram.persistence.state_manager import StateManager
from hologram.retrieval.confidence import ConfidenceScorer
from hologram.safety.citation import CitationEnforcer
from hologram.safety.refusal import RefusalPolicy


class ChatInterface:
    """
    Interactive chat interface for holographic memory.

    Modes:
    - Conversational (default): Natural language with learning
    - Command mode: Use /commands for direct fact manipulation

    Commands:
    - /store <subject> <predicate> <object> [source] - Store a fact
    - /query <subject> <predicate> - Query a fact
    - /teach <subject> <predicate> <object> - Teach a fact conversationally
    - /mode - Toggle between conversational and command mode
    - /save <path> - Save memory to disk
    - /load <path> - Load memory from disk
    - /stats - Show memory statistics
    - /help - Show help
    - /exit - Exit

    Example conversational session:
        > Hello!
        Hello! How can I help you today?

        > What is the capital of France?
        The capital of France is Paris.
    """

    def __init__(
        self,
        dimensions: int = 10000,
        conversational: bool = True,
        persistent: bool = True,
        persist_dir: str = "./data/hologram_facts",
        force_neural: bool = False,
        enable_ventriloquist: bool = True,
    ):
        """
        Initialize chat interface.

        Args:
            dimensions: Hypervector dimensionality
            conversational: Enable conversational mode (default True)
            persistent: Use ChromaDB for fact persistence (default True)
            persist_dir: Directory for persistent storage
            force_neural: Force neural consolidation mode even if no file exists
            enable_ventriloquist: Enable the SLM-based 'ventriloquist' generator
        """
        self.force_neural = force_neural
        self.container = HologramContainer(dimensions=dimensions)
        self.persistent = persistent
        self.persist_dir = persist_dir
        self.enable_ventriloquist = enable_ventriloquist

        # Conversational mode with optional persistence
        self.conversational = conversational
        self.chatbot = None
        self.fact_store = None

        if conversational:
            if persistent:
                # Check if this directory was trained with neural consolidation
                neural_path = Path(persist_dir) / "neural_memory.pt"
                use_neural = self.force_neural or neural_path.exists()
                
                if use_neural:
                    print(f"  [Neural consolidation: {'loaded' if neural_path.exists() else 'enabled'}]")
                
                self.chatbot = self.container.create_persistent_chatbot(
                    persist_dir=persist_dir,
                    enable_ventriloquist=self.enable_ventriloquist,
                    ventriloquist_model="zai-org/glm-4.6v",
                    enable_neural_consolidation=use_neural,
                )
                self.fact_store = self.chatbot._fact_store
            else:
                # In-memory only
                self.fact_store = self.container.create_fact_store()
                self.chatbot = self.container.create_conversational_chatbot(
                    fact_store=self.fact_store
                )
        else:
            self.fact_store = self.container.create_fact_store()

        self.confidence_scorer = ConfidenceScorer()
        self.refusal_policy = RefusalPolicy(self.confidence_scorer)
        self.citation_enforcer = CitationEnforcer(self.fact_store)
        self.state_manager = StateManager()

    def start(self) -> None:
        """Start interactive REPL."""
        print("=" * 60)
        if self.conversational:
            print("  Hologram: Conversational Learning Chatbot")
            if self.persistent:
                print(f"  [Facts persist to: {self.persist_dir}]")
        else:
            print("  Hologram: Bentov-Style Holographic Memory")
        print("=" * 60)
        print()

        if self.conversational and self.chatbot:
            greeting = self.chatbot.start_session()
            print(greeting)
            print()
            print("(Type naturally or use /help for commands)")
        else:
            print("Quick start:")
            print("  France capital Paris     → Store a fact")
            print("  France capital           → Query a fact")
            print("  /help                    → All commands")
        print()

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                if self.conversational and self.chatbot:
                    self.chatbot.end_session()
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
            elif self.conversational and self.chatbot:
                # Use conversational chatbot
                response = self.chatbot.respond(user_input)
                print(f"\n{response}")
            else:
                # Fall back to command-style parsing
                self._handle_natural_input(user_input)

    def _handle_natural_input(self, user_input: str) -> None:
        """Handle natural language input (non-slash commands).

        Supports:
        - "subject predicate" -> query
        - "subject predicate object" -> store (no source)
        - "subject predicate object source" -> store with source
        """
        parts = user_input.split()

        if len(parts) == 2:
            # Looks like a query: "France capital"
            self._cmd_query(user_input)
        elif len(parts) >= 3:
            # Looks like a store: "France capital Paris" or "France capital Paris Wikipedia"
            print("(Interpreting as /store command)")
            self._cmd_store(user_input)
        else:
            print("Try: '<subject> <predicate>' to query")
            print("     '<subject> <predicate> <object>' to store")
            print("     '/help' for all commands")

    def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._show_help()
        elif cmd == "/exit":
            print("Goodbye!")
            exit(0)
        elif cmd == "/store":
            self._cmd_store(args)
        elif cmd == "/query":
            self._cmd_query(args)
        elif cmd == "/teach":
            self._cmd_teach(args)
        elif cmd == "/learned":
            self._cmd_learned()
        elif cmd == "/facts":
            self._cmd_facts()
        elif cmd == "/save":
            self._cmd_save(args)
        elif cmd == "/load":
            self._cmd_load(args)
        elif cmd == "/stats":
            self._cmd_stats()
        elif cmd == "/code":
            self._cmd_code(args)
        else:
            print(f"Unknown command: {cmd}. Use /help for commands.")

    def _show_help(self) -> None:
        """Show help message."""
        print("""
Conversational Mode:
  Just type naturally! The system learns from your interactions.

  Examples:
    Hello!                      → Greeting response
    What is the capital of France? → Query knowledge
    Thanks!                     → Farewell response

Commands:
  /teach <subject> <predicate> <object>
      Teach a fact conversationally
      Example: /teach France capital Paris

  /learned
      Show what the system has learned this session
      (facts, style inference, pattern strengths)

  /facts
      List all stored facts

  /store <subject> <predicate> <object> [source]
      Store a fact directly
      Example: /store France capital Paris Wikipedia

  /query <subject> <predicate>
      Query a fact with confidence scoring
      Example: /query France capital

  /save <path>
      Save memory to disk

  /load <path>
      Load memory from disk

  /stats
      Show memory statistics

  /help
      Show this help message

  /exit
      Exit the program
        """)

    def _cmd_store(self, args: str) -> None:
        """Handle /store command."""
        parts = args.split()
        if len(parts) < 3:
            print("Usage: /store <subject> <predicate> <object> [source]")
            return

        subject = parts[0]
        predicate = parts[1]
        obj = parts[2]
        source = parts[3] if len(parts) > 3 else None

        fact = self.fact_store.add_fact(subject, predicate, obj, source=source)
        citation = self.citation_enforcer.format_citation(fact)
        print(f"✓ Stored: {citation}")

    def _cmd_query(self, args: str) -> None:
        """Handle /query command."""
        parts = args.split()
        if len(parts) < 2:
            print("Usage: /query <subject> <predicate>")
            return

        subject = parts[0]
        predicate = parts[1]

        # Query holographic memory
        answer, confidence = self.fact_store.query(subject, predicate)

        # Evaluate refusal
        refusal = self.refusal_policy.evaluate(answer, confidence)

        if refusal.should_refuse:
            print(f"❌ {self.refusal_policy.format_refusal(refusal)}")
            return

        # Find supporting fact for citation
        supporting_fact = self.citation_enforcer.find_supporting_fact(
            subject, predicate, answer
        )

        # Format response with confidence
        response = self.confidence_scorer.format_response(
            answer, confidence, include_confidence=True
        )
        print(f"✓ Answer: {response}")

        # Show citation if available
        if supporting_fact:
            citation = self.citation_enforcer.format_citation(supporting_fact)
            print(f"  Citation: {citation}")

    def _cmd_save(self, args: str) -> None:
        """Handle /save command."""
        if not args:
            print("Usage: /save <path>")
            return

        path = Path(args)
        try:
            self.state_manager.save(self.fact_store, path, description="Chat session")
            print(f"✓ Saved to: {path}")
        except Exception as e:
            print(f"❌ Error saving: {e}")

    def _cmd_load(self, args: str) -> None:
        """Handle /load command."""
        if not args:
            print("Usage: /load <path>")
            return

        path = Path(args)
        try:
            self.fact_store = self.state_manager.load(path)
            self.citation_enforcer = CitationEnforcer(self.fact_store)
            print(f"✓ Loaded from: {path}")
            print(f"  {self.fact_store}")
        except Exception as e:
            print(f"❌ Error loading: {e}")

    def _cmd_teach(self, args: str) -> None:
        """Handle /teach command - teach a fact from natural language.

        Supports multiple formats:
        - /teach France capital Paris           (3 words: subject predicate object)
        - /teach France's capital is Paris      (X's Y is Z)
        - /teach the capital of France is Paris (the Y of X is Z)
        """
        args = args.strip()
        if not args:
            print("Usage: /teach <fact>")
            print("Examples:")
            print("  /teach France capital Paris")
            print("  /teach France's capital is Paris")
            print("  /teach the capital of France is Paris")
            return

        # Try to parse natural language patterns
        subject, predicate, obj = self._parse_fact(args)

        if not subject or not predicate or not obj:
            print(f"❌ Couldn't parse: {args}")
            print("Try: /teach <subject> <predicate> <object>")
            print("  or /teach X's Y is Z")
            print("  or /teach the Y of X is Z")
            return

        if self.conversational and self.chatbot:
            response = self.chatbot.teach_fact(subject, predicate, obj)
            print(f"\n{response}")
        else:
            self.fact_store.add_fact(subject, predicate, obj, source="taught")
            print(f"✓ Learned: {subject} {predicate} {obj}")

    def _parse_fact(self, text: str) -> tuple:
        """Parse natural language into (subject, predicate, object).

        Handles:
        - "France capital Paris" → (France, capital, Paris)
        - "France's capital is Paris" → (France, capital, Paris)
        - "the capital of France is Paris" → (France, capital, Paris)
        - "Germany is in Europe" → (Germany, is, Europe)
        """
        import re

        text = text.strip()
        words = text.split()

        # Pattern 1: "X's Y is Z" (e.g., "France's capital is Paris")
        match = re.match(r"(\w+)'s\s+(\w+)\s+is\s+(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2), match.group(3).strip()

        # Pattern 2: "the Y of X is Z" (e.g., "the capital of France is Paris")
        match = re.match(r"the\s+(\w+)\s+of\s+(\w+)\s+is\s+(.+)", text, re.IGNORECASE)
        if match:
            return match.group(2), match.group(1), match.group(3).strip()

        # Pattern 3: "X is Y" (e.g., "Paris is beautiful")
        match = re.match(r"(\w+)\s+is\s+(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1), "is", match.group(2).strip()

        # Pattern 4: Simple 3-word format "X Y Z"
        if len(words) == 3:
            return words[0], words[1], words[2]

        # Pattern 5: 3+ words, take first as subject, second as predicate, rest as object
        if len(words) >= 3:
            return words[0], words[1], " ".join(words[2:])

        return None, None, None

    def _cmd_learned(self) -> None:
        """Handle /learned command - show what system has learned."""
        print("\n" + "=" * 50)
        print("  What I've Learned")
        print("=" * 50)

        # Persistence status
        if self.persistent:
            print(f"\n💾 Persistence: ENABLED ({self.persist_dir})")
        else:
            print("\n💾 Persistence: DISABLED (in-memory only)")

        # Facts learned
        facts = self.fact_store.get_all_facts() if hasattr(self.fact_store, 'get_all_facts') else []
        print(f"\n📚 Facts Stored: {len(facts) if facts else self.fact_store.fact_count}")

        # Show chatbot learning if available
        if self.conversational and self.chatbot:
            stats = self.chatbot.get_session_stats()

            print(f"\n💬 Conversation Turns: {stats['turns']}")
            print(f"📝 Messages Observed: {stats['messages_observed']}")

            # Style inference
            print(f"\n🎨 Your Inferred Style: {stats['inferred_style'].upper()}")
            print(f"   Confidence: {stats['style_confidence']:.1%}")

            # Pattern info
            print(f"\n📋 Response Patterns: {stats['patterns_count']}")

            # Show intent classifier learning
            if hasattr(self.chatbot, '_intent_classifier'):
                counts = self.chatbot._intent_classifier.get_example_counts()
                print(f"\n🧠 Intent Examples Learned:")
                for intent, count in counts.items():
                    print(f"   {intent}: {count} examples")

        print()

    def _cmd_facts(self) -> None:
        """Handle /facts command - list all stored facts."""
        print("\n📚 Stored Facts:")
        print("-" * 40)

        if hasattr(self.fact_store, 'get_all_facts'):
            facts = self.fact_store.get_all_facts()
            if not facts:
                print("  (no facts stored yet)")
            else:
                for i, fact in enumerate(facts, 1):
                    source = f" [{fact.source}]" if fact.source else ""
                    print(f"  {i}. {fact.subject} {fact.predicate} {fact.object}{source}")
        else:
            print(f"  {self.fact_store.fact_count} facts stored")
            print("  (use /query to retrieve specific facts)")
        print()

    def _cmd_stats(self) -> None:
        """Handle /stats command."""
        print(f"\n{self.fact_store}")
        print(f"Confidence scorer: {self.confidence_scorer}")
        print(f"Citation enforcer: {self.citation_enforcer}")

        if self.conversational and self.chatbot:
            print(f"\nSession: {self.chatbot.get_session_stats()}")

    @property
    def code_generator(self):
        """Lazy-loaded code generator."""
        if not hasattr(self, '_code_generator') or self._code_generator is None:
            self._code_generator = self.container.create_code_generator(
                fact_store=self.fact_store
            )
        return self._code_generator

    def _cmd_code(self, args: str) -> None:
        """Handle /code command for code generation."""
        if not args:
            print("Usage: /code <issue description>")
            print("Example: /code Add input validation to process()")
            return

        from hologram.swe import SWETask
        task = SWETask(
            task_id="chat_task",
            repo="interactive",
            issue_text=args,
            code_before={},
            code_after={},
        )

        try:
            result = self.code_generator.generate(task)
            print(f"\nGenerated {len(result.patches)} patch(es):")
            for patch in result.patches:
                print(f"  {patch.file}: {patch.operation} at {patch.location}")
                print(f"    {patch.content[:60]}...")
            print(f"\nConfidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"Code generation failed: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hologram conversational chatbot interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directory
  uv run hologram
  
  # Query training facts
  uv run hologram --persist-dir ./data/crew_training_facts
  
  # Use custom directory
  uv run hologram --persist-dir ./data/my_facts
  
  # In-memory only (no persistence)
  uv run hologram --no-persist
        """
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/hologram_facts",
        help="Directory for ChromaDB fact persistence (default: ./data/hologram_facts)"
    )
    
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable persistence (in-memory only)"
    )
    
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Force neural consolidation mode (auto-detected if neural_memory.pt exists)"
    )

    parser.add_argument(
        "--no-ventriloquist",
        action="store_true",
        help="Disable the SLM-based ventriloquist generator (use direct HDC output)"
    )
    
    args = parser.parse_args()
    
    interface = ChatInterface(
        persist_dir=args.persist_dir,
        persistent=not args.no_persist,
        force_neural=args.neural,
        enable_ventriloquist=not args.no_ventriloquist,
    )
    interface.start()


if __name__ == "__main__":
    main()

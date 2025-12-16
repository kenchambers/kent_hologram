#!/usr/bin/env python3
"""
CrewAI-based training script for Hologram chatbot.

Continuously runs conversations between Gemini (topic starter), Claude (discussant),
and Hologram (HDC chatbot) to train the system through natural conversation.

All conversations are logged and facts persist via ChromaDB.
"""

import argparse
import os
import random
import signal
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangChain LLMs directly (we use these for actual conversation)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Error: langchain dependencies not installed. Run: uv sync")
    sys.exit(1)

# Optional: Import web search (gracefully handle if not installed)
# Try the new 'ddgs' package first, fall back to legacy 'duckduckgo_search'
try:
    from ddgs import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        WEB_SEARCH_AVAILABLE = True
    except ImportError:
        WEB_SEARCH_AVAILABLE = False
        print("Note: ddgs not installed. Web teaching mode unavailable.")
        print("To enable: pip install ddgs")

# Import Hologram components
from hologram.container import HologramContainer
from hologram.conversation.vocabulary import ConversationalVocabulary
from hologram.conversation.intent import IntentType
from hologram.modulation.sesame import StyleType
from hologram.generation.cadence_extractor import CadenceExtractor
from hologram.generation.cadence_memory import CadenceMemory
from hologram.config.constants import (
    QUESTION_START_WORDS,
    CONVERSATIONAL_MARKERS,
    TEACHING_PATTERNS
)


# System prompts - Quiz Master Training Mode
GEMINI_SYSTEM_PROMPT = """You are a teacher named Alex training a young AI named Hologram to speak naturally and understand nuance.

CORE RULES:
1. Teach ONE simple fact per message: "The capital of France is Paris."
2. After teaching, QUIZ the student: "What is the capital of France?"
3. If student answers correctly, say "Correct!" and MOVE TO A NEW TOPIC.
4. If wrong or unclear, repeat the fact simply: "The capital of France is Paris."
5. Keep sentences short (<12 words) and plain SVO structure.
6. IMPORTANT: Introduce DIFFERENT topics each round—avoid recent subjects.
7. You may use slash commands to teach directly when helpful:
   - /teach <subject> <predicate> <object>
   - /teach "<phrase>" implies "<intent>"
   - /teach "<word>" connotation "<nuance>"

LANGUAGE NUANCE (use commands when clearer):
- Pragmatics: capture implied intent (e.g., "Can you pass the salt?" → request).
- Connotation: capture shading (e.g., "enormous" connotation "high intensity, formal").
- Still keep messages brief and concrete; avoid stories or role-play.

TOPIC VARIETY:
- Capitals of countries (France, Japan, Brazil, Egypt, Kenya, Norway, etc.)
- Creators/inventors (Einstein, Shakespeare, Marie Curie, Tesla, etc.)
- Natural phenomena (sun, moon, ocean, mountains, seasons, etc.)
- Colors and properties (sky is blue, grass is green, etc.)
- Animals and their traits (whales are mammals, eagles fly, etc.)

EXAMPLES:
- "The sun is a star."
- "What is the sun?"
- "Correct! Water boils at 100 degrees."
- "What temperature does water boil at?"

CRITICAL: Only write YOUR OWN response. NEVER write dialogue for other speakers (Claude, Hologram, Bot, Student).
NEVER simulate a conversation. Just state your ONE fact or question and STOP.

Your goal is to teach diverse facts, pragmatic intent, and word shading, while testing understanding."""

CLAUDE_SYSTEM_PROMPT = """You are a teacher named Claude guiding a young AI named Hologram toward natural, nuanced conversation.

CORE RULES:
1. When Alex teaches a fact, reinforce it: "Yes, Paris is the capital of France."
2. Add ONE DIFFERENT related fact: "France is in Europe."
3. Quiz the student on both facts: "What is the capital of France? What continent is France in?"
4. Keep sentences short (<12 words) and simple SVO structure.
5. If student answers correctly multiple times, INTRODUCE A NEW TOPIC instead of repeating.
6. Use slash commands when concise teaching is clearer:
   - /teach <subject> <predicate> <object>
   - /teach "<phrase>" implies "<intent>"
   - /teach "<word>" connotation "<nuance>"
7. Focus on conversational choreography: when to clarify, acknowledge, or redirect briefly.

LANGUAGE NUANCE (prefer concise commands):
- Pragmatics: store implied intent (surface → intent).
- Connotation: store shading of a word (intensity, formality, tone).
- Keep output terse; avoid stories or role-play.

EXAMPLES:
- "Yes, the sun is a star. Stars produce light."
- "What is the sun? What do stars produce?"
- "Correct! Water boils at 100 degrees. Ice melts at 0 degrees."

CRITICAL: Only write YOUR OWN response. NEVER write dialogue for Alex, Gemini, Hologram, Bot, or Student.
NEVER simulate a multi-turn conversation. Just state your ONE response and STOP.

Your goal is to reinforce facts, add related knowledge, and model human-like flow with concise prompts and commands."""


class WebTeacher:
    """
    Fetches facts from the web and extracts them for training.
    
    This class searches the web for information on a given topic,
    uses an LLM to extract clean (Subject, Predicate, Object) triples,
    and bulk loads them into the FactStore for offline knowledge acquisition.
    """
    
    def __init__(self, api_key: Optional[str] = None, llm_provider: str = "anthropic"):
        """
        Initialize WebTeacher.
        
        Args:
            api_key: API key for LLM (default: from env vars)
            llm_provider: LLM provider for extraction ("anthropic" or "google")
        """
        if not WEB_SEARCH_AVAILABLE:
            raise RuntimeError(
                "WebTeacher requires duckduckgo-search. "
                "Install with: pip install duckduckgo-search"
            )
        
        self.llm_provider = llm_provider
        
        # Initialize LLM for fact extraction
        if llm_provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            self.llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                api_key=api_key,
                temperature=0.3,
            )
        elif llm_provider == "google":
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.3,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for information.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of dicts with 'title', 'body', 'href'
        """
        try:
            # New ddgs API doesn't use context manager
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            return results if isinstance(results, list) else list(results)
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
    
    def extract_facts(self, text: str, topic: str, mode: str = "general") -> List[Tuple[str, str, str]]:
        """
        Extract (Subject, Predicate, Object) triples from text using LLM.
        
        Args:
            text: Text to extract facts from
            topic: Topic context for extraction
            mode: Extraction mode ("general" or "code")
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        if mode == "code":
            # Code-specific extraction prompt
            extraction_prompt = f"""Extract technical facts from the following programming documentation about {topic}.

For each fact, identify:
- Subject: Function name, class name, algorithm, or pattern
- Predicate: Relationship (e.g., "signature", "returns", "complexity", "purpose", "implementation", "inherits")
- Object: The value/related entity

Output ONLY valid JSON in this exact format:
{{
  "facts": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}

Rules:
1. Extract API signatures, function purposes, complexity info
2. For design patterns, extract purpose and structure
3. For algorithms, extract time/space complexity
4. Use standard predicates: signature, returns, complexity, purpose, implementation, type
5. Keep code snippets concise (max 3 lines)

Text:
{text[:2000]}

JSON Output:"""
        else:
            # General extraction prompt
            extraction_prompt = f"""Extract factual statements from the following text about {topic}.

For each fact, identify:
- Subject: The main entity
- Predicate: The relationship/property
- Object: The value/related entity

Output ONLY valid JSON in this exact format:
{{
  "facts": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}

Rules:
1. Only extract clear, factual statements
2. Use simple predicates (e.g., "capital", "inventor", "location", "is")
3. Keep subjects and objects concise
4. Skip opinions, questions, or uncertain statements

Text:
{text[:2000]}

JSON Output:"""
        
        try:
            response = self.llm.invoke(extraction_prompt)
            content = response.content.strip()
            
            # Try to extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            facts = data.get("facts", [])
            
            # Convert to tuples
            fact_tuples = []
            for fact in facts:
                if all(k in fact for k in ["subject", "predicate", "object"]):
                    fact_tuples.append((
                        fact["subject"],
                        fact["predicate"],
                        fact["object"]
                    ))
            
            return fact_tuples
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Fact extraction failed: {e}")
            return []
    
    def teach_topic(
        self,
        topic: str,
        fact_store,
        max_results: int = 3,
        max_facts_per_result: int = 5,
        mode: str = "general"
    ) -> int:
        """
        Search web for topic, extract facts, and load into FactStore.
        
        Args:
            topic: Topic to search for
            fact_store: FactStore instance to populate
            max_results: Maximum web search results to process
            max_facts_per_result: Maximum facts to extract per result
            mode: Extraction mode ("general" or "code")
            
        Returns:
            Number of facts successfully added
        """
        mode_label = "Code" if mode == "code" else "General"
        print(f"\n[WebTeacher - {mode_label}] Searching web for: {topic}")
        
        # Search web
        results = self.search_web(topic, max_results=max_results)
        if not results:
            print(f"[WebTeacher - {mode_label}] No search results found")
            return 0
        
        print(f"[WebTeacher - {mode_label}] Found {len(results)} results")
        
        # Extract and store facts from each result
        total_facts_added = 0
        for i, result in enumerate(results):
            print(f"[WebTeacher - {mode_label}] Processing result {i+1}/{len(results)}: {result.get('title', 'N/A')[:50]}...")
            
            # Combine title and body for extraction
            text = f"{result.get('title', '')} {result.get('body', '')}"
            
            # Extract facts with specified mode
            facts = self.extract_facts(text, topic, mode=mode)
            print(f"[WebTeacher - {mode_label}] Extracted {len(facts)} facts")
            
            # Add facts to store (limit per result)
            for subject, predicate, obj in facts[:max_facts_per_result]:
                try:
                    # Note: ChromaFactStore doesn't support confidence parameter
                    fact_obj = fact_store.add_fact(
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        source=f"Web ({mode}): {result.get('href', 'unknown')[:50]}",
                    )
                    if fact_obj:
                        total_facts_added += 1
                        print(f"  ✓ {subject} --{predicate}--> {obj}")
                except Exception as e:
                    print(f"  ✗ Failed to add fact: {e}")
        
        print(f"[WebTeacher - {mode_label}] Successfully added {total_facts_added} facts to FactStore")
        return total_facts_added
    
    def teach_code_topics(
        self,
        topics: List[str],
        fact_store,
        max_results_per_topic: int = 3,
        max_facts_per_result: int = 5
    ) -> int:
        """
        Teach code-specific topics using specialized extraction.
        
        This is optimized for technical documentation, extracting:
        - API signatures
        - Function purposes and complexity
        - Design pattern implementations
        - Algorithm characteristics
        
        Args:
            topics: List of programming topics (e.g., ["Python list methods", "Binary Search algorithm"])
            fact_store: FactStore to populate
            max_results_per_topic: Web results per topic
            max_facts_per_result: Facts per result
            
        Returns:
            Total facts added
        """
        print("\n" + "="*70)
        print("CODE TEACHER MODE - Technical Documentation Extraction")
        print("="*70)
        
        total_facts = 0
        for topic in topics:
            facts_added = self.teach_topic(
                topic=topic,
                fact_store=fact_store,
                max_results=max_results_per_topic,
                max_facts_per_result=max_facts_per_result,
                mode="code"  # Use code-specific extraction
            )
            total_facts += facts_added
        
        print(f"\n[CodeTeacher] Total code facts added: {total_facts}")
        return total_facts


class ConversationLogger:
    """Logs conversations to a file."""

    # ANSI color codes
    COLORS = {
        "gemini": "\033[94m",  # Blue
        "claude": "\033[92m",  # Green
        "hologram": "\033[93m",  # Yellow
        "SYSTEM": "\033[96m",    # Cyan
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"
    }

    def __init__(self, log_dir: Path = Path("./conversation_logs")):
        """Initialize logger with timestamped session file."""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"session_{timestamp}.log"
        self.log_file.write_text("")  # Clear/create file

    def log(self, speaker: str, message: str):
        """Log a message from a speaker."""
        # Validate speaker to prevent mislabeling
        valid_speakers = {"gemini", "claude", "hologram", "SYSTEM", "ERROR"}
        if speaker not in valid_speakers:
            raise ValueError(
                f"Invalid speaker '{speaker}'. Must be one of {valid_speakers}"
            )
        
        # Normalize message: replace newlines with spaces to prevent log corruption
        # Multi-line responses from LLMs can break the timestamp format
        normalized_message = " ".join(message.split())
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Write to file (plain text)
        with self.log_file.open("a") as f:
            f.write(f"[{timestamp}] {speaker}: {normalized_message}\n")
            f.flush()  # Force write to disk immediately
            
        # Print to console (colored + spacing)
        color = self.COLORS.get(speaker, "")
        reset = self.COLORS["RESET"]
        
        # Add extra spacing for readability
        print(f"\n{color}{speaker.upper()}:{reset} {normalized_message}", flush=True)

    def log_separator(self):
        """Log a separator line."""
        with self.log_file.open("a") as f:
            f.write("\n" + "=" * 60 + "\n\n")
            f.flush()  # Force write to disk immediately


class CrewTrainer:
    """Main trainer orchestrating CrewAI agents and Hologram chatbot."""

    def __init__(
        self,
        persist_dir: str = "./data/crew_training_facts",
        log_dir: Path = Path("./conversation_logs"),
        max_turns_per_topic: int = 8,
        max_rounds: Optional[int] = None,
        consolidation_threshold: int = 10,  # Lower default for faster testing
    ):
        """
        Initialize trainer.

        Args:
            persist_dir: Directory for ChromaDB fact persistence
            log_dir: Directory for conversation logs
            max_turns_per_topic: Maximum conversation turns per topic
            max_rounds: Maximum number of conversation rounds (None for unlimited)
            consolidation_threshold: Number of facts before neural consolidation triggers (default: 10)
        """
        self.persist_dir = persist_dir  # Store for saving
        # Load API keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")

        # Initialize CrewAI agents early (needed for vocabulary generation)
        print("Initializing CrewAI agents...")
        self._setup_agents()

        # Initialize vocabulary for generation
        self.vocabulary = ConversationalVocabulary()

        # Initialize Hologram chatbot with generator enabled (not corpus)
        print("Initializing Hologram chatbot...")
        self.container = HologramContainer(dimensions=10000)
        
        # Build vocabulary from existing facts (if any)
        # Access fact store directly via ChromaDB
        vocab_dict = self._build_vocabulary_from_persist_dir(persist_dir)
        
        # Create chatbot with hybrid generation (HDC + SLM)
        # Enable Ventriloquist for fluent natural language output
        self.chatbot = self.container.create_persistent_chatbot(
            persist_dir=persist_dir,
            enable_corpus=False,  # Disable corpus - use generator instead
            enable_generation=True,  # Enable ResonantGenerator for factual questions
            vocabulary=vocab_dict,
            enable_ventriloquist=True,  # Enable VentriloquistGenerator for fluency
            ventriloquist_model="moonshotai/kimi-k2-thinking",  # Novita/Kimi model
            enable_neural_consolidation=True,  # Enable Neural Consolidation Layer
            consolidation_threshold=consolidation_threshold,  # Configurable threshold
        )
        self.chatbot.start_session()

        # Initialize logger
        self.logger = ConversationLogger(log_dir)

        # Configuration
        self.max_turns_per_topic = max_turns_per_topic
        self.max_rounds = max_rounds
        self.running = True

        # Conversation history
        self.conversation_history: List[Tuple[str, str]] = []
        
        # Learning statistics
        self.facts_learned_count = 0
        self.facts_learned: List[str] = []
        self.responses_learned_count = 0
        
        # NEW: Cadence extraction and memory
        self._cadence_extractor = CadenceExtractor(self.container._codebook)
        dimensions = self.container._space.dimensions
        self._cadence_memory = CadenceMemory(dimensions=dimensions)

        # Load existing cadence memory if available
        cadence_path = Path(persist_dir) / "cadence_memory.pt"
        if cadence_path.exists():
            try:
                cadence_state = torch.load(cadence_path, weights_only=False)
                self._cadence_memory.load_state_dict(cadence_state)
                print(f"Loaded cadence memory from {cadence_path} ({self._cadence_memory.pattern_count} patterns)")
            except Exception as e:
                print(f"Warning: Could not load cadence memory: {e}")
        
        # Quiz tracking
        self.quiz_questions_asked = 0
        self.quiz_answers_correct = 0
        self.quiz_answers_incorrect = 0
        self.last_taught_fact: Optional[str] = None  # Track what fact was just taught
        self.awaiting_feedback: bool = False  # Track if we are waiting for feedback on a quiz question

    def _build_vocabulary_from_persist_dir(self, persist_dir: str) -> dict:
        """
        Build vocabulary dictionary from persisted facts for Resonator.
        
        Args:
            persist_dir: Directory where facts are persisted
            
        Returns:
            Dict with 'nouns' and 'verbs' lists
        """
        nouns = set()
        verbs = {"is", "are", "was", "were", "has", "have", "can", "do", "does"}  # Base verbs
        
        try:
            # Try to load from neural memory first
            import torch
            from pathlib import Path
            neural_path = Path(persist_dir) / "neural_memory.pt"
            
            if neural_path.exists():
                # If neural memory exists, we might not have easy access to the exact words 
                # without instantiating the manager. For now, rely on base vocabulary 
                # or try to peek at value_vocab if stored in the pt file.
                state = torch.load(neural_path, weights_only=False)
                value_vocab = state.get("value_vocab", {})
                for word in value_vocab.keys():
                    if len(word) > 2:
                        nouns.add(word)
                print(f"Initialized vocabulary from neural memory: {len(nouns)} nouns")
            
            # Fallback/Complement: Try ChromaDB if it exists (legacy/dual mode)
            from hologram.persistence.chroma_adapter import ChromaFactStore
            
            chroma_store = ChromaFactStore(
                codebook=self.container._codebook,
                persist_dir=persist_dir,
            )
            facts = chroma_store.get_all_facts()
            
            for fact in facts:
                # Handle both dict and object-like facts
                subject = fact.get('subject', '') if isinstance(fact, dict) else getattr(fact, 'subject', '')
                obj = fact.get('object', '') if isinstance(fact, dict) else getattr(fact, 'object', '')
                predicate = fact.get('predicate', '') if isinstance(fact, dict) else getattr(fact, 'predicate', '')
                
                if subject:
                    # Split multi-word subjects/objects
                    for word in subject.split():
                        word_clean = word.strip().lower()
                        if len(word_clean) > 2:  # Skip very short words
                            nouns.add(word_clean)
                
                if obj:
                    for word in obj.split():
                        word_clean = word.strip().lower()
                        if len(word_clean) > 2:
                            nouns.add(word_clean)
                
                # Some predicates are verbs
                if predicate and predicate not in ["capital", "creator", "is", "are"]:
                    pred_clean = predicate.strip().lower()
                    if len(pred_clean) > 2:
                        verbs.add(pred_clean)
        except Exception as e:
            # If fact store doesn't exist or can't be accessed, start with empty vocab
            print(f"Note: Starting with base vocabulary (no existing facts found): {e}")
        
        # PRE-SEED vocabulary with useful starter words
        # This prevents the "vocabulary death spiral" where generator
        # is frozen with only "unknown" and can never learn
        if not nouns:
            # Starter vocabulary for cold start (empty database)
            # Use LLM to generate diverse seed vocabulary instead of hardcoding
            if hasattr(self, 'gemini_llm'):
                try:
                    print("Generating seed vocabulary from Gemini...")
                    prompt = (
                        "Generate a list of 100 diverse, common English nouns that would be useful "
                        "for a general knowledge chatbot. Include a mix of:\n"
                        "- Countries and cities\n"
                        "- Scientific concepts (e.g. atom, energy)\n"
                        "- Natural world (e.g. ocean, mountain)\n"
                        "- Everyday objects\n"
                        "- Abstract concepts (e.g. time, history)\n"
                        "Output ONLY the words separated by commas, no numbering or bullets."
                    )
                    response = self.gemini_llm.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(gemini_response)
                    words = [w.strip().lower() for w in content.split(',')]
                    valid_words = {w for w in words if len(w) > 2 and ' ' not in w} # Single words only
                    
                    if valid_words:
                        nouns.update(valid_words)
                        print(f"Generated {len(valid_words)} seed nouns.")
                    else:
                        raise ValueError("No valid words generated")
                except Exception as e:
                    print(f"Failed to generate vocabulary from LLM: {e}")
                    # Fallback to minimal set if LLM fails
                    nouns.update(["thing", "person", "place", "idea", "time", "world", "life"])
            else:
                 # Fallback if LLM not ready
                 nouns.update(["thing", "person", "place", "idea", "time", "world", "life"])

        # Add essential verbs for generating proper sentences
        verbs.update(["create", "invent", "discover", "build", "design", "develop"])

        return {
            "nouns": sorted(list(nouns)),
            "verbs": sorted(list(verbs))
        }
    
    def _update_vocabulary_from_fact(self, fact_string: str) -> None:
        """
        Update vocabulary from a newly learned fact string.
        
        Args:
            fact_string: Fact string like "sun is star" or "France capital Paris"
        """
        if not self.chatbot._generator:
            return  # No generator to update
        
        # Parse fact string (simple: split by spaces, assume S-P-O or S-O format)
        parts = fact_string.lower().split()
        if len(parts) >= 2:
            # Add words to vocabulary
            for word in parts:
                word_clean = word.strip()
                if len(word_clean) > 2:
                    # Add to vocabulary's noun/verb sets
                    self.vocabulary.learn_from_text(word_clean)
        
        # Note: Generator vocabulary is set at init time, so this is for future sessions
        # In a production system, we'd want to dynamically update the generator's vocab
    
    def _process_llm_command(self, message: str, speaker: str = "llm") -> bool:
        """
        Detect and execute slash commands issued by Gemini/Claude.
        
        Supported commands (single line):
        - /teach <subject> <predicate> <object>
        - /teach "<phrase>" implies "<intent>"
        - /teach "<word>" connotation "<nuance>"
        
        Returns:
            True if a command was processed (and message should NOT be sent to Hologram).
        """
        import re
        
        fact_store = getattr(self.chatbot, "_fact_store", None)
        if not fact_store:
            return False
        
        processed = False
        lines = message.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line.startswith("/teach"):
                continue
            
            # Log the raw command for audit in the training log
            self.logger.log("SYSTEM", f"COMMAND from {speaker}: {line}")
            
            # Pattern: /teach "<phrase>" implies "<intent>"
            m_implies = re.match(r'^/teach\s+"(.+?)"\s+implies\s+"(.+?)"\s*$', line, re.IGNORECASE)
            if m_implies:
                phrase, intent = m_implies.group(1).strip(), m_implies.group(2).strip()
                try:
                    fact_store.add_fact(subject=phrase, predicate="implies", obj=intent, source="crew_command")
                    self.logger.log("SYSTEM", f"✓ Stored pragmatic intent: {phrase} implies {intent}")
                    self.last_taught_fact = f"{phrase} implies {intent}"
                    processed = True
                except Exception as exc:
                    self.logger.log("ERROR", f"Failed to store pragmatic intent: {exc}")
                continue
            
            # Pattern: /teach "<word>" connotation "<nuance>"
            m_connotation = re.match(r'^/teach\s+"(.+?)"\s+connotation\s+"(.+?)"\s*$', line, re.IGNORECASE)
            if m_connotation:
                word, nuance = m_connotation.group(1).strip(), m_connotation.group(2).strip()
                try:
                    fact_store.add_fact(subject=word, predicate="connotation", obj=nuance, source="crew_command")
                    self.logger.log("SYSTEM", f"✓ Stored connotation: {word} connotation {nuance}")
                    self.last_taught_fact = f"{word} connotation {nuance}"
                    processed = True
                except Exception as exc:
                    self.logger.log("ERROR", f"Failed to store connotation: {exc}")
                continue
            
            # Fallback: /teach <subject> <predicate> <object...>
            # Try to extract clean S-P-O from various formats

            # First try: look for simple pattern "X is Y" or "X <predicate> Y"
            simple_match = re.match(
                r'^/teach\s+["\']?(\w+)["\']?\s+(?:is|capital|color|has|was)\s+["\']?(.+?)["\']?\.?\s*$',
                line, re.IGNORECASE
            )
            if simple_match:
                subject = simple_match.group(1).strip('"\'')
                obj = simple_match.group(2).strip('"\'.')
                # Extract predicate from line
                pred_match = re.search(r'\s+(is|capital|color|has|was)\s+', line, re.IGNORECASE)
                predicate = pred_match.group(1) if pred_match else "is"
                try:
                    confirmation = self.chatbot.teach_fact(subject, predicate, obj)
                    if confirmation is not None:
                        self.logger.log("SYSTEM", f"✓ Taught fact via command: {subject} {predicate} {obj}")
                        self.last_taught_fact = f"{subject} {predicate} {obj}"
                        processed = True
                except Exception as exc:
                    self.logger.log("ERROR", f"Failed to teach fact: {exc}")
                continue

            # Second try: handle "The X of Y is Z" pattern
            capital_match = re.match(
                r'^/teach\s+["\']?(?:The\s+)?capital\s+of\s+(\w+)\s+is\s+(\w+)["\']?\.?\s*$',
                line, re.IGNORECASE
            )
            if capital_match:
                subject = capital_match.group(1).strip('"\'')
                obj = capital_match.group(2).strip('"\'.')
                try:
                    confirmation = self.chatbot.teach_fact(subject, "capital", obj)
                    if confirmation is not None:
                        self.logger.log("SYSTEM", f"✓ Taught fact via command: {subject} capital {obj}")
                        self.last_taught_fact = f"{subject} capital {obj}"
                        processed = True
                except Exception as exc:
                    self.logger.log("ERROR", f"Failed to teach fact: {exc}")
                continue

            # Last fallback: simple space split with quote stripping
            parts = line.split()
            if len(parts) >= 4:
                subject = parts[1].strip('"\'')
                predicate = parts[2].strip('"\'')
                obj = " ".join(parts[3:]).strip('"\'.').strip()
                # Clean up common issues
                obj = re.sub(r'^the\s+', '', obj, flags=re.IGNORECASE)  # Remove leading "the"
                obj = re.sub(r'\s*of\s+\w+\.?$', '', obj)  # Remove trailing "of X"
                if obj:
                    try:
                        confirmation = self.chatbot.teach_fact(subject, predicate, obj)
                        if confirmation is not None:
                            self.logger.log("SYSTEM", f"✓ Taught fact via command: {subject} {predicate} {obj}")
                            self.last_taught_fact = f"{subject} {predicate} {obj}"
                            processed = True
                    except Exception as exc:
                        self.logger.log("ERROR", f"Failed to teach fact: {exc}")
                continue
        
        return processed
    
    def _build_vocabulary_from_facts(self, fact_store) -> dict:
        """
        Build vocabulary dictionary from FactStore for Resonator.
        
        Returns:
            Dict with 'nouns' and 'verbs' lists
        """
        nouns = set()
        verbs = {"is", "are", "was", "were", "has", "have", "can", "do", "does"}  # Base verbs
        
        try:
            if hasattr(fact_store, 'get_all_facts'):
                facts = fact_store.get_all_facts()
                for fact in facts:
                    # Extract subject and object as nouns
                    subject = getattr(fact, 'subject', '') or fact.get('subject', '') if isinstance(fact, dict) else ''
                    obj = getattr(fact, 'object', '') or fact.get('object', '') if isinstance(fact, dict) else ''
                    predicate = getattr(fact, 'predicate', '') or fact.get('predicate', '') if isinstance(fact, dict) else ''
                    
                    if subject:
                        # Split multi-word subjects/objects
                        for word in subject.split():
                            word_clean = word.strip().lower()
                            if len(word_clean) > 2:  # Skip very short words
                                nouns.add(word_clean)
                    
                    if obj:
                        for word in obj.split():
                            word_clean = word.strip().lower()
                            if len(word_clean) > 2:
                                nouns.add(word_clean)
                    
                    # Some predicates are verbs
                    if predicate and predicate not in ["capital", "creator", "is", "are"]:
                        pred_clean = predicate.strip().lower()
                        if len(pred_clean) > 2:
                            verbs.add(pred_clean)
        except Exception as e:
            # If fact store doesn't support get_all_facts, start with empty vocab
            print(f"Warning: Could not extract vocabulary from facts: {e}")
        
        # Add common conversational words
        nouns.update(["capital", "city", "country", "person", "thing", "place", "time"])
        
        return {
            "nouns": sorted(list(nouns)),
            "verbs": sorted(list(verbs))
        }
    
    def _setup_agents(self):
        """Set up CrewAI agents with LLMs."""
        # Use LangChain LLMs directly for more control
        # Gemini LLM (cheap model for topic generation)
        # Model name can be configured via GEMINI_MODEL env var
        # Common options: "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        gemini_llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=self.gemini_key,
            temperature=0.7,
        )

        # Claude LLM (cheap model for discussion)
        # Model name can be configured via ANTHROPIC_MODEL env var
        claude_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        claude_llm = ChatAnthropic(
            model=claude_model,
            anthropic_api_key=self.anthropic_key,
            temperature=0.7,
        )

        # Store LLMs for direct invocation
        # We use LangChain LLMs directly for the conversation loop
        self.gemini_llm = gemini_llm
        self.claude_llm = claude_llm
    
    def web_teach_topics(
        self,
        topics: List[str],
        max_results_per_topic: int = 3,
        max_facts_per_result: int = 5
    ) -> int:
        """
        Use WebTeacher to fetch facts from the web and populate FactStore.
        
        This is an offline training mode that rapidly fills the knowledge base
        with facts from the web, solving the "cold start" problem.
        
        Args:
            topics: List of topics to search for (e.g., ["Physics", "World Capitals"])
            max_results_per_topic: Web results to process per topic
            max_facts_per_result: Facts to extract per result
            
        Returns:
            Total number of facts added
            
        Example:
            >>> trainer.web_teach_topics(["World Capitals", "Famous Scientists"])
        """
        if not WEB_SEARCH_AVAILABLE:
            print("ERROR: Web teaching requires duckduckgo-search.")
            print("Install with: pip install duckduckgo-search")
            return 0
        
        # Initialize WebTeacher (use Claude for extraction)
        try:
            web_teacher = WebTeacher(llm_provider="anthropic")
        except Exception as e:
            print(f"ERROR: Failed to initialize WebTeacher: {e}")
            return 0
        
        # Get fact store from chatbot
        fact_store = self.chatbot._fact_store
        
        # Teach each topic
        total_facts = 0
        for topic in topics:
            facts_added = web_teacher.teach_topic(
                topic=topic,
                fact_store=fact_store,
                max_results=max_results_per_topic,
                max_facts_per_result=max_facts_per_result
            )
            total_facts += facts_added
            
            # Update learning statistics
            self.facts_learned_count += facts_added
        
        print(f"\n[WebTeacher] Total facts added across all topics: {total_facts}")
        return total_facts
    
    def web_teach_code_topics(
        self,
        topics: List[str],
        max_results_per_topic: int = 3,
        max_facts_per_result: int = 5
    ) -> int:
        """
        Use CodeTeacher mode to fetch code concepts from technical documentation.
        
        This is specialized for programming topics, extracting:
        - API signatures and return types
        - Algorithm complexity information
        - Design pattern implementations
        - Best practices and principles
        
        Args:
            topics: List of programming topics (e.g., ["Python dict methods", "Sorting algorithms"])
            max_results_per_topic: Web results to process per topic
            max_facts_per_result: Facts to extract per result
            
        Returns:
            Total number of code facts added
            
        Example:
            >>> trainer.web_teach_code_topics(["Python list comprehensions", "Binary search tree"])
        """
        if not WEB_SEARCH_AVAILABLE:
            print("ERROR: Code teaching requires duckduckgo-search.")
            print("Install with: pip install duckduckgo-search")
            return 0
        
        # Initialize WebTeacher (use Claude for extraction)
        try:
            web_teacher = WebTeacher(llm_provider="anthropic")
        except Exception as e:
            print(f"ERROR: Failed to initialize WebTeacher: {e}")
            return 0
        
        # Get fact store from chatbot
        fact_store = self.chatbot._fact_store
        
        # Teach code topics using specialized mode
        total_facts = web_teacher.teach_code_topics(
            topics=topics,
            fact_store=fact_store,
            max_results_per_topic=max_results_per_topic,
            max_facts_per_result=max_facts_per_result
        )
        
        # Update learning statistics
        self.facts_learned_count += total_facts
        
        print(f"\n[CodeTeacher] Total code facts added: {total_facts}")
        return total_facts

    def _get_conversation_context(self, last_n: int = 5) -> str:
        """Get recent conversation context as a string.
        
        Uses generic labels to prevent LLMs from role-playing as hologram.
        """
        if not self.conversation_history:
            return ""

        recent = self.conversation_history[-last_n:]
        context_lines = []
        for speaker, message in recent:
            # Use generic labels to prevent role-playing
            # Map: gemini/claude -> "Partner", hologram -> "Bot"
            if speaker == "hologram":
                label = "Bot"
            else:
                label = "Partner"
            context_lines.append(f"{label}: {message}")
        return "\n".join(context_lines)

    def _detect_fact_learning(self, response: str) -> Optional[str]:
        """
        Detect if the chatbot learned a fact this turn.

        NEW: Uses explicit learning protocol instead of string parsing.
        Queries chatbot.did_learn_fact_this_turn() for reliable detection.

        Returns the learned fact string (subject predicate object) if detected, None otherwise.
        """
        # NEW PROTOCOL: Use explicit learning flag
        if self.chatbot.did_learn_fact_this_turn():
            fact_tuple = self.chatbot.get_last_learned_fact()
            if fact_tuple:
                subject, predicate, obj = fact_tuple
                return f"{subject} {predicate} {obj}"

        # FALLBACK: Old string parsing (for backwards compatibility)
        response_lower = response.lower()
        if "got it" in response_lower and "i'll remember that" in response_lower:
            parts = response.split("remember that", 1)
            if len(parts) > 1:
                fact_part = parts[1].strip().rstrip(".")
                return fact_part

        return None

    def _detect_quiz_question(self, text: str) -> bool:
        """
        Detect if a message is a quiz question.
        
        Returns True if it looks like a quiz question (ends with ? and asks about a fact).
        """
        text_lower = text.lower().strip()
        
        # Must end with question mark
        if not text_lower.endswith("?"):
            return False
        
        return any(text_lower.startswith(qw) for qw in QUESTION_START_WORDS)
    
    def _detect_quiz_feedback(self, text: str) -> Optional[bool]:
        """
        Detect if a message is feedback on a quiz answer.
        
        Returns:
            True if correct, False if incorrect, None if not feedback
        """
        text_lower = text.lower().strip()
        
        # Correct feedback patterns
        correct_patterns = [
            "correct",
            "right",
            "yes",
            "that's right",
            "exactly",
            "good",
            "well done",
        ]
        
        # Incorrect feedback patterns
        incorrect_patterns = [
            "wrong",
            "incorrect",
            "no",
            "that's not",
            "not quite",
            "try again",
        ]
        
        # Check for correct feedback
        if any(pattern in text_lower for pattern in correct_patterns):
            return True
        
        # Check for incorrect feedback
        if any(pattern in text_lower for pattern in incorrect_patterns):
            return False
        
        return None
    
    def bulk_import_facts(
        self,
        facts: List[Tuple[str, str, str]],
        source: str = "bulk",
        confidence: float = 1.0,
    ) -> int:
        """
        Import a batch of facts directly into the chatbot's FactStore.

        Args:
            facts: List of (subject, predicate, object) tuples
            source: Source label for citation
            confidence: Learning rate / confidence for each fact

        Returns:
            Number of facts newly added (duplicates are skipped)
        """
        fact_store = getattr(self.chatbot, "_fact_store", None)
        if fact_store is None:
            self.logger.log("ERROR", "Chatbot has no fact store; cannot import facts.")
            return 0

        added = 0
        for subject, predicate, obj in facts:
            try:
                fact = fact_store.add_fact(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    source=source,
                    confidence=confidence,
                )
                if fact:
                    added += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.log("ERROR", f"Failed to add fact ({subject}, {predicate}, {obj}): {exc}")

        self.logger.log("SYSTEM", f"Bulk import complete: {added}/{len(facts)} added")
        return added
    
    def _is_teaching_statement(self, text: str) -> bool:
        """
        Detect if a message is a teaching statement (fact) vs conversational.
        
        Returns True if it looks like a fact statement, False if conversational.
        """
        text_lower = text.lower()
        
        # Check for conversational markers first
        if any(marker in text_lower for marker in CONVERSATIONAL_MARKERS):
            return False
        
        # Check for teaching patterns
        if any(pattern in text_lower for pattern in TEACHING_PATTERNS):
            return True
        
        # Check if it ends with ? (likely conversational question)
        if text.strip().endswith("?"):
            return False
        
        # Default: assume conversational
        return False

    def _process_llm_response_for_cadence(
        self, response: str, context_vec: Optional[torch.Tensor]
    ) -> None:
        """
        Process LLM response to extract both facts AND cadence patterns.

        Args:
            response: LLM response text
            context_vec: Context vector for cadence storage
        """
        if context_vec is None:
            return

        # Extract entities from response (simple: capitalized words)
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', response)
        
        # Extract cadence patterns
        try:
            cadence = self._cadence_extractor.extract_multi_sentence_cadence(
                response, entities
            )
            
            # Store each pattern
            for pattern in cadence.patterns:
                self._cadence_memory.store_cadence(context_vec, pattern)
            
            # Log cadence learning
            if cadence.transitions:
                transition_str = ", ".join([t.value for t in cadence.transitions])
                self.logger.log(
                    "SYSTEM",
                    f"  [Cadence] Learned {len(cadence.patterns)} patterns, "
                    f"transitions: [{transition_str}]"
                )
        except Exception as e:
            # Silently fail cadence extraction (non-critical)
            pass

    def _learn_response_from_llm(self, message: str, speaker: str) -> None:
        """
        Learn a response from Claude/Gemini if it's conversational (not teaching).
        
        Splits the response into atomic units (sentences) to enable
        compositional retrieval rather than paragraph replay.
        
        Args:
            message: The LLM's message
            speaker: "claude" or "gemini"
        """
        # Clean the message of any roleplay/prompt leakage artifacts
        for prefix in ["Human:", "Partner:", "AI:", "User:", "Assistant:"]:
            if prefix in message:
                if message.strip().startswith(prefix):
                    message = message.replace(prefix, "", 1).strip()
                else:
                    message = message.split(prefix)[0].strip()
        
        if not message:
            return

        # Skip if it's a teaching statement (those are handled by fact learning)
        if self._is_teaching_statement(message):
            return
        
        # CRITICAL: Use the chatbot's internal memory context vector
        context_vec = self.chatbot._memory.get_context_vector()
        
        # Classify intent (use chatbot's classifier)
        intent_result = self.chatbot._intent_classifier.classify(message)
        
        # Infer style
        style = self.chatbot._style_tracker.get_inferred_style()
        
        # NEW: ATOMIC DECOMPOSITION
        # Split message into sentences to store smaller, reusable shards
        import re
        # Split by punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', message)
        
        learned_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty or very short sentences (e.g. "Ok.")
            if len(sentence) < 5:
                continue
                
            # Learn each atomic sentence with the SAME context vector
            self.chatbot.learn_response(
                context_vector=context_vec,
                response=sentence,
                intent=intent_result.intent,
                style=style,
                source=speaker,
            )
            learned_count += 1
        
        if learned_count > 0:
            self.responses_learned_count += learned_count
            self.logger.log("SYSTEM", f"✓ Learned {learned_count} atomic responses from {speaker}")
            
            # NEW: Extract and store cadence patterns from LLM responses
            self._process_llm_response_for_cadence(message, context_vec)

    def run_conversation_round(self) -> None:
        """Run a single conversation round (topic + discussion)."""
        # Check if trainer is running before starting
        if not self.running:
            return
        
        self.logger.log_separator()
        self.logger.log("SYSTEM", f"Starting new conversation round")

        # Reset quiz state for new round
        self.awaiting_feedback = False

        # Step 1: Gemini starts a topic/teaching statement
        # Check if system is struggling and adjust teaching style
        simplify_instruction = ""
        if self.chatbot._metacognitive:
            trend = self.chatbot._metacognitive.state.get_confidence_trend()
            if trend < -0.1:
                simplify_instruction = "\nIMPORTANT: The student is struggling. Use the SIMPLEST possible fact format (e.g., 'The capital of France is Paris'). Avoid complex facts."
        
        # Encourage teaching statements that match the chatbot's TEACHING intent patterns
        # Build diversity instruction based on recently taught facts
        recent_topics = set()
        for fact in self.facts_learned[-5:]:  # Last 5 facts
            # Extract subject from fact string (first word typically)
            parts = fact.lower().split()
            if parts:
                recent_topics.add(parts[0])
        
        diversity_instruction = ""
        if recent_topics:
            topics_str = ", ".join(recent_topics)
            diversity_instruction = f"\n\nIMPORTANT: We recently discussed: {topics_str}. Please teach a DIFFERENT topic now (different country, person, or concept). Introduce variety!"

        topic_prompt = (
            f"{GEMINI_SYSTEM_PROMPT}\n\n"
            "Pick a RANDOM topic and teach ONE fact about it. Be creative and unpredictable!\n"
            "Choose from categories like: world capitals, famous inventors, science facts, "
            "animals, geography, history, art, music, sports, food, technology, or anything interesting.\n\n"
            "Use ONE of these exact formats:\n"
            "- 'The capital of [country] is [city]'\n"
            "- '[Person] invented/discovered [thing]'\n"
            "- '[Thing] is [property]'\n"
            "- 'The [property] of [thing] is [value]'\n\n"
            "IMPORTANT: State the fact directly without preamble like 'apparently' or 'I think'."
            f"{simplify_instruction}"
            f"{diversity_instruction}"
        )

        if self.conversation_history:
            topic_prompt += f"\n\nIMPORTANT: Only write YOUR response. Do NOT write dialogue for the Bot."
            # Don't show full conversation context to avoid topic fixation
            # topic_prompt += f"\n\nRecent conversation:\n{self._get_conversation_context()}"

        # Use LangChain LLM directly for more control
        gemini_response = self.gemini_llm.invoke(topic_prompt)
        gemini_message = gemini_response.content if hasattr(gemini_response, 'content') else str(gemini_response)
        self.logger.log("gemini", gemini_message)
        self.conversation_history.append(("gemini", gemini_message))
        
        # Learn vocabulary from Gemini's message
        self.vocabulary.learn_from_text(gemini_message)
        
        # Learn response if conversational
        self._learn_response_from_llm(gemini_message, "gemini")

        # Check if Gemini asked a quiz question (start of round)
        if self._detect_quiz_question(gemini_message):
            self.quiz_questions_asked += 1
            self.awaiting_feedback = True

        # Step 2: If Gemini issued a command, execute it and skip response; otherwise respond
        if self._process_llm_command(gemini_message, speaker="gemini"):
            ack = "Command received and stored."
            self.logger.log("hologram", ack)
            self.conversation_history.append(("hologram", ack))
            self.awaiting_feedback = False  # Command supersedes quiz
        else:
            # The chatbot automatically detects TEACHING intent and extracts facts
            hologram_response = self.chatbot.respond(gemini_message)
            
            # Check if a fact was learned (chatbot returns confirmation message)
            fact_learned = self._detect_fact_learning(hologram_response)
            if fact_learned:
                self.facts_learned_count += 1
                self.facts_learned.append(fact_learned)
                self.logger.log("SYSTEM", f"✓ Fact learned ({self.facts_learned_count} total): {fact_learned}")

            # Clear learning flag for next turn (explicit protocol)
            self.chatbot.clear_learning_flag()

            self.logger.log("hologram", hologram_response)
            self.conversation_history.append(("hologram", hologram_response))

        # Step 3-6: Discussion loop between Claude and Hologram
        turn = 0
        import random
        while turn < self.max_turns_per_topic and self.running:
            turn += 1
            
            # NEW: Dynamic interaction loop
            # Speakers: Claude, Gemini, Hologram (optional)
            
            # Randomly decide if Claude or Gemini speaks next (bias towards Claude as "discussant")
            # 70% Claude, 30% Gemini
            speaker = "claude" if random.random() < 0.7 else "gemini"
            
            # Context for next speaker
            last_msg = self.conversation_history[-1][1] if self.conversation_history else "Start discussion."
            
            prompt = ""
            if speaker == "claude":
                prompt = (
                    f"{CLAUDE_SYSTEM_PROMPT}\n\n"
                    f"Respond to the conversation naturally. Interact with Gemini (Alex) or the student.\n"
                    f"Last message: {last_msg}"
                )
                response = self.claude_llm.invoke(prompt)
            else:
                prompt = (
                    f"{GEMINI_SYSTEM_PROMPT}\n\n"
                    f"Respond to the conversation naturally. Interact with Claude or the student.\n"
                    f"Last message: {last_msg}"
                )
                response = self.gemini_llm.invoke(prompt)
                
            message = response.content if hasattr(response, 'content') else str(response)
            self.logger.log(speaker, message)
            self.conversation_history.append((speaker, message))
            
            # Learn vocabulary
            self.vocabulary.learn_from_text(message)
            self._learn_response_from_llm(message, speaker)
            
            # Check for quiz/commands
            is_command = self._process_llm_command(message, speaker=speaker)
            if is_command:
                ack = "Command received and stored."
                self.logger.log("hologram", ack)
                self.conversation_history.append(("hologram", ack))
                self.awaiting_feedback = False
                continue

            # Hologram Interaction Logic:
            # 1. Listen to everything (learn facts/vocabulary/context)
            # 2. Respond ONLY if:
            #    a) Directly addressed ("Hologram", "student", "?")
            #    b) High confidence response available (e.g. knows the answer)
            #    c) Random chance (20%) to keep conversation alive
            
            addressed_directly = "hologram" in message.lower() or "?" in message
            
            # Determine if we should speak
            should_speak = False
            
            if addressed_directly:
                should_speak = True
            elif random.random() < 0.2:  # Occasional chime-in
                should_speak = True
            
            # Always listen first
            self.chatbot.listen(message)
            
            # Check if we learned a fact while listening
            if self.chatbot.did_learn_fact_this_turn():
                fact_learned = self.chatbot.get_last_learned_fact()
                if fact_learned:
                    subject, predicate, obj = fact_learned
                    fact_str = f"{subject} {predicate} {obj}"
                    self.facts_learned_count += 1
                    self.facts_learned.append(fact_str)
                    self.last_taught_fact = fact_str
                    self.logger.log("SYSTEM", f"✓ Fact learned while listening: {fact_str}")
                    self._update_vocabulary_from_fact(fact_str)
                # If we learned a fact, we might want to acknowledge it even if not addressed
                if random.random() < 0.5:
                    should_speak = True
            self.chatbot.clear_learning_flag()

            if should_speak:
                # Actually generate a response
                # Note: We already 'listened', so context is updated.
                # But respond() expects to process the input again.
                # To avoid double-processing, we ideally should have separate 'generate' method.
                # For now, calling respond() again is safe (idempotent-ish), just a bit inefficient.
                # Optimization: In future refactor, split process/generate.
                hologram_response = self.chatbot.respond(message)
                self.logger.log("hologram", hologram_response)
                self.conversation_history.append(("hologram", hologram_response))
            else:
                # Log that we are listening
                # self.logger.log("hologram", "(listening...)")
                pass

            # Keep last 20 messages for context
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        self.logger.log_separator()

    def run_continuous(self):
        """Run continuous training loop with error recovery."""
        import time
        import traceback
        
        print("\n" + "=" * 60)
        print("  CrewAI Hologram Trainer - Continuous Mode")
        print("=" * 60)
        print(f"Logging to: {self.logger.log_file}")
        print(f"Facts persist to: ./data/crew_training_facts")
        if self.max_rounds:
            print(f"Max rounds: {self.max_rounds}")
        else:
            print("Max rounds: Unlimited (run until stopped)")
        print(f"Turns per topic: {self.max_turns_per_topic}")
        print("\nPress Ctrl+C to stop gracefully\n")

        round_num = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while self.running and (self.max_rounds is None or round_num < self.max_rounds):
                round_num += 1
                print(f"\n>>> Round {round_num} {'/' + str(self.max_rounds) if self.max_rounds else ''} <<<")
                
                try:
                    self.run_conversation_round()
                    consecutive_errors = 0  # Reset on success
                    
                    # Status report every 10 rounds
                    if round_num % 10 == 0:
                        self._print_status_report(round_num)
                        # Save vocabulary periodically
                        vocab_stats = self.vocabulary.get_stats()
                        self.logger.log("SYSTEM", f"Vocabulary: {vocab_stats['total_words']} words ({vocab_stats['nouns']} nouns, {vocab_stats['verbs']} verbs)")
                        
                        # Save neural memory periodically (don't force consolidation every time)
                        self.chatbot.save_memory(self.persist_dir, force_consolidation=False)
                        
                        # NEW: Consolidate and save cadence memory periodically
                        cadence_loss = self._cadence_memory.consolidate(epochs=30)
                        if cadence_loss > 0:
                            self.logger.log(
                                "SYSTEM",
                                f"  [Cadence Consolidation] Loss: {cadence_loss:.4f}, "
                                f"Patterns: {self._cadence_memory.pattern_count}"
                            )
                            # Save cadence memory periodically
                            cadence_state = self._cadence_memory.get_state_dict()
                            if cadence_state:
                                cadence_path = Path(self.persist_dir) / "cadence_memory.pt"
                                torch.save(cadence_state, cadence_path)

                except Exception as e:
                    consecutive_errors += 1
                    error_msg = f"Error in round {round_num}: {str(e)}"
                    print(f"\n⚠️  {error_msg}")
                    self.logger.log("ERROR", error_msg)
                    self.logger.log("ERROR", traceback.format_exc())
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n❌ Too many consecutive errors ({consecutive_errors}). Stopping...")
                        self.logger.log("SYSTEM", f"Stopped due to {consecutive_errors} consecutive errors")
                        break
                    
                    # Exponential backoff on errors
                    wait_time = min(60, 2 ** consecutive_errors)
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # Brief pause between successful rounds
                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\nStopping trainer...")
            self.running = False
        
        finally:
            # Always log final statistics
            self.logger.log("SYSTEM", f"Training stopped after {round_num} rounds")
            self.logger.log("SYSTEM", f"Total facts learned: {self.facts_learned_count}")
            self.logger.log("SYSTEM", f"Total responses learned: {self.responses_learned_count}")
            
            # Save neural memory on exit (with forced consolidation)
            if hasattr(self, 'chatbot'):
                # Force consolidation of any pending facts before saving
                self.chatbot.save_memory(self.persist_dir, force_consolidation=True)
                
                # NEW: Final cadence consolidation
                cadence_loss = self._cadence_memory.consolidate(epochs=50)
                if cadence_loss > 0:
                    self.logger.log(
                        "SYSTEM",
                        f"  [Final Cadence Consolidation] Loss: {cadence_loss:.4f}, "
                        f"Patterns: {self._cadence_memory.pattern_count}"
                    )

                # Save cadence memory to disk
                cadence_state = self._cadence_memory.get_state_dict()
                if cadence_state:
                    cadence_path = Path(self.persist_dir) / "cadence_memory.pt"
                    torch.save(cadence_state, cadence_path)
                    self.logger.log(
                        "SYSTEM",
                        f"💾 Saved cadence memory: {cadence_path} "
                        f"({self._cadence_memory.pattern_count} patterns)"
                    )
                    print(f"💾 Cadence memory saved: {cadence_path}")

                # Ensure worker is stopped
                self.chatbot.end_session()
            
            # Show final statistics
            self._print_status_report(round_num, final=True)
            print(f"\nConversation log saved to: {self.logger.log_file}")
            print(f"Facts persisted to ChromaDB at: ./data/crew_training_facts")
            print(f"Vocabulary stats: {self.vocabulary.get_stats()}")
    
    def _print_status_report(self, round_num: int, final: bool = False):
        """Print a status report of training progress."""
        title = "Final Training Summary" if final else f"Status Report - Round {round_num}"
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Rounds completed: {round_num}")
        print(f"  Facts learned (this session): {self.facts_learned_count}")
        print(f"  Responses learned (this session): {self.responses_learned_count}")
        if self.quiz_questions_asked > 0:
            accuracy = (self.quiz_answers_correct / self.quiz_questions_asked) * 100
            print(f"  Quiz performance: {self.quiz_answers_correct}/{self.quiz_questions_asked} correct ({accuracy:.1f}%)")
        print(f"  Total facts in store: {self.chatbot._fact_store.fact_count}")
        if self.chatbot._corpus:
            print(f"  Total responses in corpus: {self.chatbot._corpus.get_entry_count()}")
        vocab_stats = self.vocabulary.get_stats()
        print(f"  Vocabulary: {vocab_stats['total_words']} words ({vocab_stats['nouns']} nouns, {vocab_stats['verbs']} verbs)")
        print(f"  Conversation turns: {len(self.conversation_history)}")
        print(f"  (Note: Facts and responses are saved immediately upon learning)")
        if self.facts_learned:
            print(f"  Recent facts:")
            for fact in self.facts_learned[-3:]:
                print(f"    • {fact}")
        print(f"{'='*60}")


def signal_handler(sig, frame):
    """Handle SIGINT gracefully."""
    print("\n\nReceived interrupt signal. Stopping...")
    # sys.exit(0)
    raise KeyboardInterrupt


def main():
    """Main entry point with command-line arguments."""
    # Force unbuffered output for nohup/background runs
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    parser = argparse.ArgumentParser(
        description="CrewAI Hologram Trainer - Continuous training for Hologram chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unlimited rounds (until stopped)
  python crew_trainer.py
  
  # Run 100 rounds overnight
  python crew_trainer.py --max-rounds 100
  
  # Run with custom configuration
  python crew_trainer.py --max-rounds 50 --turns-per-topic 10
  
  # Use custom directories
  python crew_trainer.py --persist-dir ./my_facts --log-dir ./my_logs
  
  # Web teaching mode: populate knowledge base from web
  python crew_trainer.py --web-teach "World Capitals" "Famous Scientists" "Physics Basics"
  
  # Code teaching mode: populate with programming concepts
  python crew_trainer.py --web-teach-code "Python list methods" "Sorting algorithms" "Design patterns"
  
  # Combined teaching: general + code concepts
  python crew_trainer.py --web-teach "Physics" --web-teach-code "Binary search algorithm"
  
  # Web teaching with custom search parameters
  python crew_trainer.py --web-teach "Machine Learning" --web-results 5 --web-facts 10
  
  # Code teaching with custom parameters
  python crew_trainer.py --web-teach-code "Graph algorithms" "Dynamic programming" --web-results 5
  
  # Web teaching then conversational training
  python crew_trainer.py --web-teach "History" "Geography" --max-rounds 50
        """
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/crew_training_facts",
        help="Directory for ChromaDB fact persistence (default: ./data/crew_training_facts)"
    )
    
    parser.add_argument(
        "--log-dir",
        default="./conversation_logs",
        help="Directory for conversation logs (default: ./conversation_logs)"
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of conversation rounds (default: unlimited)"
    )
    
    parser.add_argument(
        "--turns-per-topic",
        type=int,
        default=8,
        help="Maximum conversation turns per topic (default: 8)"
    )
    
    parser.add_argument(
        "--web-teach",
        nargs="+",
        metavar="TOPIC",
        help="Web teaching mode: fetch facts from web for given topics (e.g., --web-teach 'World Capitals' 'Physics')"
    )
    
    parser.add_argument(
        "--web-teach-code",
        nargs="+",
        metavar="CODE_TOPIC",
        help="Code teaching mode: fetch programming concepts from technical docs (e.g., --web-teach-code 'Python list methods' 'Binary search')"
    )
    
    parser.add_argument(
        "--web-results",
        type=int,
        default=3,
        help="Number of web search results to process per topic (default: 3)"
    )
    
    parser.add_argument(
        "--web-facts",
        type=int,
        default=5,
        help="Maximum facts to extract per web result (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler
    # signal.signal(signal.SIGINT, signal_handler)

    # Create trainer
    trainer = CrewTrainer(
        persist_dir=args.persist_dir,
        log_dir=Path(args.log_dir),
        max_turns_per_topic=args.turns_per_topic,
        max_rounds=args.max_rounds,
    )

    # Check for web teaching mode
    if args.web_teach:
        print("\n" + "="*70)
        print("WEB TEACHING MODE - General Knowledge")
        print("="*70)
        print(f"Topics: {', '.join(args.web_teach)}")
        print(f"Results per topic: {args.web_results}")
        print(f"Facts per result: {args.web_facts}")
        print("="*70 + "\n")
        
        # Run web teaching
        total_facts = trainer.web_teach_topics(
            topics=args.web_teach,
            max_results_per_topic=args.web_results,
            max_facts_per_result=args.web_facts
        )
        
        print("\n" + "="*70)
        print(f"WEB TEACHING COMPLETE: {total_facts} facts added")
        print("="*70)
        
        # Ask if user wants to continue with conversational training
        response = input("\nContinue with conversational training? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Check for code teaching mode
    if args.web_teach_code:
        print("\n" + "="*70)
        print("CODE TEACHING MODE - Technical Documentation")
        print("="*70)
        print(f"Code Topics: {', '.join(args.web_teach_code)}")
        print(f"Results per topic: {args.web_results}")
        print(f"Facts per result: {args.web_facts}")
        print("="*70 + "\n")
        
        # Run code teaching
        total_facts = trainer.web_teach_code_topics(
            topics=args.web_teach_code,
            max_results_per_topic=args.web_results,
            max_facts_per_result=args.web_facts
        )
        
        print("\n" + "="*70)
        print(f"CODE TEACHING COMPLETE: {total_facts} code facts added")
        print("="*70)
        
        # Ask if user wants to continue
        if not args.web_teach:  # Only ask if we didn't already ask
            response = input("\nContinue with conversational training? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # Run conversational training
    trainer.run_continuous()


if __name__ == "__main__":
    main()

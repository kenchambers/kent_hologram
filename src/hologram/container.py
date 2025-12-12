"""
Dependency Injection Container for Hologram.

Manages shared dependencies and ensures all components use the same
VectorSpace for dimensional consistency.

Supports both the core holographic memory system and the Resonant Cavity
Architecture for constrained generation.
"""

from pathlib import Path
from typing import Dict, List, Optional

from hologram.config.constants import (
    CREATIVITY_TEMPERATURE,
    MAX_RESONATOR_ITERATIONS,
    STYLE_WEIGHT,
)
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.core.fractal import FractalSpace
from hologram.memory.fact_store import FactStore
from hologram.memory.memory_trace import MemoryTrace
from hologram.memory.sequence_encoder import SequenceEncoder


class HologramContainer:
    """
    Dependency injection container for Hologram system.

    Manages shared singletons (VectorSpace, Codebook) and provides
    factories for creating components that depend on them.

    This ensures all components use the same vector space, preventing
    dimensional mismatches.

    Attributes:
        _space: Shared VectorSpace configuration
        _codebook: Shared Codebook instance

    Example:
        >>> container = HologramContainer(dimensions=10000)
        >>> fact_store = container.create_fact_store()
        >>> encoder = container.create_sequence_encoder()
        >>> # Both use the same VectorSpace and Codebook
    """

    def __init__(self, dimensions: int = 10000, use_semantic: bool = False, use_fractal: bool = True):
        """
        Initialize container with shared dependencies.

        Args:
            dimensions: Hypervector dimensionality (default: 10000)
            use_semantic: If True, use SemanticCodebook if available (default: False)
                          NOTE: SemanticCodebook is DISABLED by default because its
                          random projection breaks HDC bind/bundle operations,
                          resulting in near-zero similarities for pattern matching.
            use_fractal: If True, use FractalSpace instead of VectorSpace (default: True)
                         FractalSpace provides holographic properties: any fragment
                         contains the whole concept (lower resolution).
        """
        # Shared singletons - use FractalSpace by default for holographic properties
        if use_fractal:
            self._space = FractalSpace(dimensions=dimensions)
        else:
            self._space = VectorSpace(dimensions=dimensions)
        
        # SemanticCodebook is disabled by default - it breaks HDC operations
        if use_semantic:
            try:
                from hologram.core.semantic_codebook import SemanticCodebook
                self._codebook = SemanticCodebook(self._space)
            except ImportError:
                # Fall back to regular Codebook if sentence-transformers not available
                self._codebook = Codebook(self._space)
        else:
            self._codebook = Codebook(self._space)

    @property
    def vector_space(self) -> VectorSpace:
        """Get the shared VectorSpace instance."""
        return self._space

    @property
    def codebook(self) -> Codebook:
        """Get the shared Codebook instance."""
        return self._codebook

    def create_fact_store(
        self,
        enable_neural_consolidation: bool = False,
        consolidation_threshold: int = 20,
    ) -> FactStore:
        """
        Create a new FactStore instance.

        Args:
            enable_neural_consolidation: If True, use Neural Consolidation layer
            consolidation_threshold: Facts before consolidation triggers

        Returns:
            FactStore using shared VectorSpace and Codebook
        """
        consolidation_manager = None
        if enable_neural_consolidation:
            consolidation_manager = self.create_consolidation_manager(
                threshold=consolidation_threshold
            )
            
        return FactStore(
            self._space, 
            self._codebook,
            consolidation_manager=consolidation_manager
        )

    def create_consolidation_manager(
        self,
        threshold: int = 20,
        decay_factor: float = 0.3,
        neural_hidden_dim: int = 256,
    ):
        """
        Create a ConsolidationManager for neural memory.

        Args:
            threshold: Number of pending facts to trigger consolidation
            decay_factor: HDC decay factor
            neural_hidden_dim: Neural network hidden dimension

        Returns:
            ConsolidationManager instance
        """
        from hologram.consolidation.manager import ConsolidationManager
        return ConsolidationManager(
            space=self._space,
            consolidation_threshold=threshold,
            decay_factor=decay_factor,
            neural_hidden_dim=neural_hidden_dim,
        )

    def create_hierarchical_fact_store(
        self,
        faiss_path: str = "/tmp/hologram_faiss",
        hot_confidence_threshold: float = 0.7,
    ):
        """
        Create a HierarchicalFactStore (hot HDC + cold FAISS).

        Args:
            faiss_path: Directory for FAISS persistence
            hot_confidence_threshold: Confidence threshold to keep queries in hot layer

        Returns:
            HierarchicalFactStore instance
        """
        from hologram.memory.fact_store import HierarchicalFactStore

        return HierarchicalFactStore(
            space=self._space,
            codebook=self._codebook,
            faiss_path=faiss_path,
            hot_confidence_threshold=hot_confidence_threshold,
        )

    def create_memory_trace(self) -> MemoryTrace:
        """
        Create a new MemoryTrace instance.

        Returns:
            MemoryTrace using shared VectorSpace
        """
        return MemoryTrace(self._space)

    def create_sequence_encoder(self, max_length: int = 1000) -> SequenceEncoder:
        """
        Create a new SequenceEncoder instance.

        Args:
            max_length: Maximum sequence length

        Returns:
            SequenceEncoder using shared Codebook
        """
        return SequenceEncoder(self._codebook, max_length=max_length)

    def create_faiss_adapter(self, path: str | Path):
        """
        Create a Faiss adapter for persistence.

        Args:
            path: Directory path for Faiss index storage

        Returns:
            FaissAdapter instance
        """
        from hologram.persistence.faiss_adapter import FaissAdapter
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return FaissAdapter(self._space.dimensions, str(path_obj))

    # =========================================================================
    # Resonant Cavity Architecture Components
    # =========================================================================

    def create_resonator(
        self,
        max_iterations: int = MAX_RESONATOR_ITERATIONS,
    ):
        """
        Create a Resonator for thought factorization.

        The Resonator decomposes composite thought vectors into
        (subject, verb, object) components using ALS.

        Args:
            max_iterations: Maximum ALS iterations

        Returns:
            Resonator using shared Codebook
        """
        from hologram.core.resonator import Resonator
        return Resonator(
            codebook=self._codebook,
            max_iterations=max_iterations,
        )

    def create_target_encoder(
        self,
        style_weight: float = STYLE_WEIGHT,
    ):
        """
        Create a TargetEncoder for constraint packaging.

        Args:
            style_weight: Weight for style injection

        Returns:
            TargetEncoder using shared Codebook
        """
        from hologram.cavity.target_encoder import TargetEncoder
        return TargetEncoder(
            codebook=self._codebook,
            style_weight=style_weight,
        )

    def create_re_encoder(self):
        """
        Create a ReEncoder for token projection.

        Returns:
            ReEncoder using shared Codebook
        """
        from hologram.cavity.re_encoder import ReEncoder
        return ReEncoder(codebook=self._codebook)

    def create_divergence_calculator(self):
        """
        Create a DivergenceCalculator for verification.

        Returns:
            DivergenceCalculator using shared Codebook
        """
        from hologram.cavity.divergence import DivergenceCalculator
        return DivergenceCalculator(codebook=self._codebook)

    def create_sesame_modulator(
        self,
        creativity: float = CREATIVITY_TEMPERATURE,
    ):
        """
        Create a SesameModulator for style and disfluency.

        Args:
            creativity: Creativity temperature

        Returns:
            SesameModulator using shared Codebook
        """
        from hologram.modulation.sesame import SesameModulator
        return SesameModulator(
            codebook=self._codebook,
            creativity_temperature=creativity,
        )

    def create_metacognitive_loop(
        self,
        max_retries: int = 2,
        retry_threshold: float = 0.3,
    ):
        """
        Create a MetacognitiveLoop for self-monitoring and adaptive behavior.

        The metacognitive layer maintains a persistent "self-state" vector that
        observes query confidence and rewires itself when stuck, enabling retry
        loops with modified query vectors.

        Args:
            max_retries: Maximum retry attempts (default: 2)
            retry_threshold: Confidence below which to retry (default: 0.3)

        Returns:
            MetacognitiveLoop using shared Codebook
        """
        from hologram.cognition.metacognition import MetacognitiveLoop
        return MetacognitiveLoop(
            codebook=self._codebook,
            max_retries=max_retries,
            retry_threshold=retry_threshold,
        )

    def create_ventriloquist_generator(
        self,
        api_key: Optional[str] = None,
        fluency_model: str = "moonshotai/kimi-k2-thinking",
        reasoning_model: str = "zai-org/glm-4.6v",
        temperature: float = 0.7,
        max_tokens: int = 256,
        enable_reasoning: bool = True,
    ):
        """
        Create a VentriloquistGenerator using Novita API (SLM for fluent output).

        The Ventriloquist acts as the "Voice Box" - it takes facts retrieved by
        the Hologram and generates fluent natural language responses.

        Args:
            api_key: Novita API key (default: from NOVITA_API_KEY env var)
            fluency_model: Model for fluent responses (default: moonshotai/kimi-k2-thinking)
            reasoning_model: Model for reasoning chains (default: zai-org/glm-4.6v)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens per response (default: 256)
            enable_reasoning: Enable chain-of-thought reasoning (default: True)

        Returns:
            VentriloquistGenerator instance
        """
        from hologram.generation.ventriloquist import VentriloquistGenerator
        return VentriloquistGenerator(
            api_key=api_key,
            fluency_model=fluency_model,
            reasoning_model=reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )

    def create_resonant_generator(
        self,
        vocabulary: Dict[str, List[str]],
        max_iterations: int = MAX_RESONATOR_ITERATIONS,
        style_weight: float = STYLE_WEIGHT,
        creativity: float = CREATIVITY_TEMPERATURE,
    ):
        """
        Create a full ResonantGenerator with all components.

        This is the main entry point for constrained generation.
        Creates and wires together all Resonant Cavity components.

        Args:
            vocabulary: Dict with "nouns" and "verbs" lists
            max_iterations: Resonator max iterations
            style_weight: TargetEncoder style weight
            creativity: SesameModulator creativity

        Returns:
            Fully configured ResonantGenerator

        Example:
            >>> vocab = {"nouns": ["cat", "dog"], "verbs": ["eats", "chases"]}
            >>> generator = container.create_resonant_generator(vocab)
            >>> result = generator.generate(thought_vector)
        """
        from hologram.generation.resonant_generator import ResonantGenerator

        return ResonantGenerator(
            resonator=self.create_resonator(max_iterations),
            target_encoder=self.create_target_encoder(style_weight),
            re_encoder=self.create_re_encoder(),
            divergence_calculator=self.create_divergence_calculator(),
            sesame_modulator=self.create_sesame_modulator(creativity),
            vocabulary=vocabulary,
        )

    # =========================================================================
    # Conversational Chatbot Components
    # =========================================================================

    def create_conversational_chatbot(
        self,
        fact_store: Optional[FactStore] = None,
        enable_corpus: bool = False,
        enable_generation: bool = False,
        vocabulary: Optional[Dict[str, List[str]]] = None,
        enable_metacognition: bool = True,
        enable_ventriloquist: bool = True,
        ventriloquist_model: str = "moonshotai/kimi-k2-thinking",
        ventriloquist_mode: str = "full",
    ):
        """
        Create a fully wired ConversationalChatbot.

        This is the main entry point for the learning conversational system.
        Creates and wires together all conversation components.

        Args:
            fact_store: Optional existing FactStore for knowledge
            enable_corpus: If True, create ResponseCorpus for learned responses
            enable_generation: If True, create ResonantGenerator for token-level generation
            vocabulary: Optional vocabulary dict for generator (required if enable_generation=True)
            enable_metacognition: If True, create MetacognitiveLoop for self-monitoring (default: True)
            enable_ventriloquist: If True, create VentriloquistGenerator for fluent SLM output (default: False)
            ventriloquist_model: Model identifier for Ventriloquist (default: moonshotai/kimi-k2-thinking)

        Returns:
            Configured ConversationalChatbot

        Example:
            >>> container = HologramContainer()
            >>> chatbot = container.create_conversational_chatbot()
            >>> response = chatbot.respond("Hello!")
        """
        from hologram.conversation.chatbot import ConversationalChatbot
        from hologram.conversation.entity import EntityExtractor
        from hologram.conversation.fast_intent import FastIntentClassifier
        from hologram.conversation.memory import ConversationMemory
        from hologram.conversation.patterns import ResponsePatternStore
        from hologram.conversation.selector import ResponseSelector
        from hologram.conversation.style_tracker import UserStyleTracker
        from hologram.conversation.corpus import ResponseCorpus

        # Create or use provided fact store
        fact_store = fact_store or self.create_fact_store()

        # Create all conversation components
        # Use FastIntentClassifier for accurate, fast intent classification
        intent_classifier = FastIntentClassifier()
        entity_extractor = EntityExtractor(self._codebook, fact_store)
        # Optional episodic store (FAISS) for long-context retrieval
        episodic_store = None
        try:
            episodic_store = self.create_faiss_adapter(Path("./data/episodic_memory"))
        except Exception:
            episodic_store = None

        conversation_memory = ConversationMemory(
            self._space,
            self._codebook,
            episodic_store=episodic_store,
        )
        pattern_store = ResponsePatternStore(self._space, self._codebook)
        sesame_modulator = self.create_sesame_modulator()
        style_tracker = UserStyleTracker(self._codebook, sesame_modulator)

        # Create optional corpus
        response_corpus = None
        if enable_corpus:
            response_corpus = ResponseCorpus(self._codebook)

        # Create optional generators
        resonant_generator = None
        if enable_generation:
            if not vocabulary:
                raise ValueError("vocabulary dict required when enable_generation=True")
            resonant_generator = self.create_resonant_generator(vocabulary)

        ventriloquist_generator = None
        if enable_ventriloquist:
            ventriloquist_generator = self.create_ventriloquist_generator(
                fluency_model=ventriloquist_model
            )

        # Create optional metacognitive loop
        metacognitive_loop = None
        if enable_metacognition:
            metacognitive_loop = self.create_metacognitive_loop()

        response_selector = ResponseSelector(
            pattern_store=pattern_store,
            conversation_memory=conversation_memory,
            fact_store=fact_store,
            codebook=self._codebook,
            response_corpus=response_corpus,
            resonant_generator=resonant_generator,
            ventriloquist_generator=ventriloquist_generator,
            voice_mode=ventriloquist_mode,
        )

        return ConversationalChatbot(
            intent_classifier=intent_classifier,
            entity_extractor=entity_extractor,
            response_selector=response_selector,
            sesame_modulator=sesame_modulator,
            conversation_memory=conversation_memory,
            style_tracker=style_tracker,
            pattern_store=pattern_store,
            fact_store=fact_store,
            codebook=self._codebook,
            response_corpus=response_corpus,
            resonant_generator=resonant_generator,
            metacognitive_loop=metacognitive_loop,
        )

    def create_chroma_fact_store(
        self,
        persist_dir: str = "./data/hologram_facts",
        collection_name: str = "facts",
        auto_recover: bool = True,
    ):
        """
        Create a ChromaDB-backed persistent fact store.

        Facts stored here persist across sessions.

        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
            auto_recover: Automatically recover from corrupted database

        Returns:
            ChromaFactStore instance
        """
        from hologram.persistence.chroma_adapter import ChromaFactStore

        return ChromaFactStore(
            codebook=self._codebook,
            persist_dir=persist_dir,
            collection_name=collection_name,
            auto_recover=auto_recover,
        )

    def create_persistent_chatbot(
        self,
        persist_dir: str = "./data/hologram_facts",
        enable_corpus: bool = False,  # Disabled by default - use generator instead
        enable_generation: bool = False,  # Still False by default, but can be enabled
        vocabulary: Optional[Dict[str, List[str]]] = None,
        enable_metacognition: bool = True,
        enable_ventriloquist: bool = False,
        ventriloquist_model: str = "moonshotai/kimi-k2-thinking",
        ventriloquist_mode: str = "full",
        enable_neural_consolidation: bool = False,
    ):
        """
        Create a ConversationalChatbot with persistent fact storage.

        Uses ChromaDB to persist facts across sessions. Optionally includes
        response corpus and token-level generation.

        Args:
            persist_dir: Directory for fact persistence
            enable_corpus: If True, create ResponseCorpus (default: True)
            enable_generation: If True, create ResonantGenerator (default: False)
            vocabulary: Optional vocabulary dict for generator (required if enable_generation=True)
            enable_metacognition: If True, create MetacognitiveLoop for self-monitoring (default: True)
            enable_ventriloquist: If True, create VentriloquistGenerator for fluent SLM output (default: False)
            ventriloquist_model: Model identifier for Ventriloquist (default: moonshotai/kimi-k2-thinking)
            enable_neural_consolidation: If True, use Neural Consolidation instead of Chroma (default: False)

        Returns:
            ConversationalChatbot with persistent storage

        Example:
            >>> container = HologramContainer()
            >>> chatbot = container.create_persistent_chatbot()
            >>> chatbot.teach_fact("France", "capital", "Paris")
            >>> # Exit and restart...
            >>> chatbot = container.create_persistent_chatbot()
            >>> chatbot.respond("What is the capital of France?")
            >>> # Returns "Paris" from persistent storage
        """
        from hologram.conversation.chatbot import ConversationalChatbot
        from hologram.conversation.entity import EntityExtractor
        from hologram.conversation.fast_intent import FastIntentClassifier
        from hologram.conversation.memory import ConversationMemory
        from hologram.conversation.patterns import ResponsePatternStore
        from hologram.conversation.selector import ResponseSelector
        from hologram.conversation.style_tracker import UserStyleTracker
        from hologram.persistence.chroma_adapter import ChromaFactStore, ChromaResponseCorpus

        # Create persistent fact store
        if enable_neural_consolidation:
            # Use Neural Consolidation (FactStore with Manager)
            # Persistence is handled by loading state dict if it exists
            import torch
            import os
            
            fact_store = self.create_fact_store(enable_neural_consolidation=True)
            
            # Try to load existing neural state
            neural_path = Path(persist_dir) / "neural_memory.pt"
            if neural_path.exists():
                try:
                    state = torch.load(neural_path)
                    if fact_store._consolidation_manager:
                        fact_store._consolidation_manager.load_state_dict(state)
                        print(f"Loaded neural memory from {neural_path}")
                except Exception as e:
                    print(f"Failed to load neural memory: {e}")
            
            # We also ensure the directory exists for saving later
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
        else:
            # Use ChromaDB (Standard Persistence)
            chroma_store = ChromaFactStore(
                codebook=self._codebook,
                persist_dir=persist_dir,
            )
            # Create wrapper that bridges ChromaFactStore to FactStore interface
            fact_store = self._create_chroma_bridge(chroma_store)

        # Create all conversation components
        # Use FastIntentClassifier for accurate, fast intent classification
        intent_classifier = FastIntentClassifier()
        entity_extractor = EntityExtractor(self._codebook, fact_store)
        episodic_store = None
        try:
            episodic_store = self.create_faiss_adapter(Path("./data/episodic_memory"))
        except Exception:
            episodic_store = None

        conversation_memory = ConversationMemory(
            self._space,
            self._codebook,
            episodic_store=episodic_store,
        )
        pattern_store = ResponsePatternStore(self._space, self._codebook)
        sesame_modulator = self.create_sesame_modulator()
        style_tracker = UserStyleTracker(self._codebook, sesame_modulator)

        # Create optional corpus
        response_corpus = None
        if enable_corpus:
            response_corpus = ChromaResponseCorpus(
                codebook=self._codebook,
                persist_dir=persist_dir,
            )

        # Create optional generators
        resonant_generator = None
        if enable_generation:
            if not vocabulary:
                raise ValueError("vocabulary dict required when enable_generation=True")
            resonant_generator = self.create_resonant_generator(vocabulary)

        ventriloquist_generator = None
        if enable_ventriloquist:
            ventriloquist_generator = self.create_ventriloquist_generator(
                fluency_model=ventriloquist_model
            )

        # Create optional metacognitive loop
        metacognitive_loop = None
        if enable_metacognition:
            metacognitive_loop = self.create_metacognitive_loop()

        response_selector = ResponseSelector(
            pattern_store=pattern_store,
            conversation_memory=conversation_memory,
            fact_store=fact_store,
            codebook=self._codebook,
            response_corpus=response_corpus,
            resonant_generator=resonant_generator,
            ventriloquist_generator=ventriloquist_generator,
            voice_mode=ventriloquist_mode,
        )

        return ConversationalChatbot(
            intent_classifier=intent_classifier,
            entity_extractor=entity_extractor,
            response_selector=response_selector,
            sesame_modulator=sesame_modulator,
            conversation_memory=conversation_memory,
            style_tracker=style_tracker,
            pattern_store=pattern_store,
            fact_store=fact_store,
            codebook=self._codebook,
            response_corpus=response_corpus,
            resonant_generator=resonant_generator,
            metacognitive_loop=metacognitive_loop,
        )

    def _create_chroma_bridge(self, chroma_store):
        """
        Create a bridge object that adapts ChromaFactStore to FactStore interface.

        This allows the chatbot to use ChromaDB for persistence while
        maintaining compatibility with the existing FactStore API.
        """
        from dataclasses import dataclass

        @dataclass
        class FactBridge:
            """Adapts ChromaFactStore to FactStore interface."""

            _chroma: object

            def _make_fact(self, f):
                """Create a fact-like object from a dict or values."""
                if isinstance(f, dict):
                    return type("Fact", (), {
                        "subject": f.get("subject", ""),
                        "predicate": f.get("predicate", ""),
                        "object": f.get("object", ""),
                        "source": f.get("source", ""),
                    })()
                return f

            def add_fact(self, subject, predicate, obj, source=None):
                # Returns None if duplicate, fact_id if new
                fact_id = self._chroma.add_fact(subject, predicate, obj, source)
                if fact_id is None:
                    # Duplicate - return None to match FactStore behavior
                    return None
                # Return a fact-like object for new facts
                return type("Fact", (), {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "source": source,
                })()

            def query(self, subject, predicate):
                return self._chroma.query(subject, predicate)

            def get_all_facts(self):
                facts = self._chroma.get_all_facts()
                # Convert dicts to fact-like objects
                return [self._make_fact(f) for f in facts]

            @property
            def fact_count(self):
                return self._chroma.fact_count

            @property
            def vocabulary_size(self):
                """Return count of unique objects in facts (vocabulary)."""
                facts = self._chroma.get_all_facts()
                unique_objects = set(f.get("object", "") for f in facts)
                return len(unique_objects)

            @property
            def _facts(self):
                """Provide _facts for compatibility with CitationEnforcer."""
                return self.get_all_facts()

            def get_facts_by_subject(self, subject):
                facts = self._chroma.query_by_metadata(subject=subject)
                return [self._make_fact(f) for f in facts]

            def get_facts_by_predicate(self, predicate):
                """Get facts by predicate for compatibility."""
                facts = self._chroma.query_by_metadata(predicate=predicate)
                return [self._make_fact(f) for f in facts]

            def __repr__(self):
                return f"FactBridge(chroma={self._chroma})"

        return FactBridge(_chroma=chroma_store)

    def __repr__(self) -> str:
        return (
            f"HologramContainer(dimensions={self._space.dimensions}, "
            f"codebook_cache={self._codebook.cache_size()})"
        )

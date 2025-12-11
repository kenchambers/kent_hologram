"""
Hologram: Bentov-Style Holographic Memory using Hyperdimensional Computing.

This package implements a memory system based on interference patterns in
high-dimensional vector spaces, providing bounded hallucination through
algebraic resonance rather than probabilistic prediction.

The system includes:
- Core holographic memory (FactStore, MemoryTrace)
- Safety mechanisms (RefusalPolicy, CitationEnforcer, ConfidenceScorer)
- Resonant Cavity Architecture for constrained generation
- Conversational Learning Chatbot using pure HDC architecture
"""

__version__ = "0.3.0"

# Core components
from hologram.chat import ChatInterface
from hologram.container import HologramContainer
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import Fact, FactStore
from hologram.persistence.state_manager import StateManager
from hologram.retrieval.confidence import ConfidenceScorer
from hologram.safety.citation import CitationEnforcer
from hologram.safety.refusal import RefusalPolicy

# Resonant Cavity Architecture
from hologram.core.resonator import Resonator, ResonatorResult
from hologram.cavity.target_encoder import TargetEncoder, TargetPackage
from hologram.cavity.re_encoder import ReEncoder
from hologram.cavity.divergence import (
    DivergenceCalculator,
    DivergenceResult,
    DivergenceAction,
)
from hologram.modulation.sesame import SesameModulator, StyleType, FillerType
from hologram.generation.resonant_generator import (
    ResonantGenerator,
    GenerationResult,
    GenerationMetrics,
)

# Conversational Learning Chatbot
from hologram.conversation.chatbot import ConversationalChatbot
from hologram.conversation.intent import IntentClassifier, IntentType, IntentResult
from hologram.conversation.entity import EntityExtractor, Entity
from hologram.conversation.memory import ConversationMemory, ConversationTurn
from hologram.conversation.patterns import ResponsePatternStore, ResponsePattern
from hologram.conversation.selector import ResponseSelector, ResponseCandidate
from hologram.conversation.style_tracker import UserStyleTracker

__all__ = [
    # Core
    "HologramContainer",
    "VectorSpace",
    "Fact",
    "FactStore",
    "StateManager",
    "ConfidenceScorer",
    "RefusalPolicy",
    "CitationEnforcer",
    "ChatInterface",
    # Resonant Cavity
    "Resonator",
    "ResonatorResult",
    "TargetEncoder",
    "TargetPackage",
    "ReEncoder",
    "DivergenceCalculator",
    "DivergenceResult",
    "DivergenceAction",
    "SesameModulator",
    "StyleType",
    "FillerType",
    "ResonantGenerator",
    "GenerationResult",
    "GenerationMetrics",
    # Conversational Chatbot
    "ConversationalChatbot",
    "IntentClassifier",
    "IntentType",
    "IntentResult",
    "EntityExtractor",
    "Entity",
    "ConversationMemory",
    "ConversationTurn",
    "ResponsePatternStore",
    "ResponsePattern",
    "ResponseSelector",
    "ResponseCandidate",
    "UserStyleTracker",
]

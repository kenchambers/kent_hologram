"""
Conversation package: Conversational chatbot using HDC architecture.

This package implements a learning conversational system that:
- Detects user intent via HDC similarity
- Extracts entities from vocabulary resonance
- Maintains conversation context holographically
- Learns from successful interactions (Hebbian)
- Adapts to user communication style

Components:
    - IntentClassifier: Intent detection via prototype vectors
    - EntityExtractor: Entity recognition via vocabulary matching
    - ConversationMemory: Session context as holographic trace
    - ResponsePatternStore: Response patterns with learning
    - UserStyleTracker: Style adaptation
    - ResponseSelector: Response selection orchestration
    - ConversationalChatbot: Main chatbot class
"""

from hologram.conversation.intent import IntentClassifier, IntentType, IntentResult
from hologram.conversation.entity import EntityExtractor, Entity
from hologram.conversation.memory import ConversationMemory, ConversationTurn
from hologram.conversation.patterns import ResponsePatternStore, ResponsePattern
from hologram.conversation.style_tracker import UserStyleTracker
from hologram.conversation.selector import ResponseSelector, ResponseCandidate
from hologram.conversation.chatbot import ConversationalChatbot
from hologram.conversation.corpus import ResponseCorpus, CorpusEntry
from hologram.conversation.vocabulary import ConversationalVocabulary

__all__ = [
    # Intent
    "IntentClassifier",
    "IntentType",
    "IntentResult",
    # Entity
    "EntityExtractor",
    "Entity",
    # Memory
    "ConversationMemory",
    "ConversationTurn",
    # Patterns
    "ResponsePatternStore",
    "ResponsePattern",
    # Style
    "UserStyleTracker",
    # Selection
    "ResponseSelector",
    "ResponseCandidate",
    # Corpus
    "ResponseCorpus",
    "CorpusEntry",
    # Vocabulary
    "ConversationalVocabulary",
    # Main
    "ConversationalChatbot",
]

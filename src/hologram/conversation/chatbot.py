"""
Main conversational chatbot orchestrating all components.

Provides the primary interface for natural conversation using
HDC-based intent detection, entity extraction, and response generation.
"""

from datetime import datetime
from typing import Optional

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.conversation.entity import Entity, EntityExtractor
from hologram.conversation.intent import IntentClassifier, IntentResult, IntentType
from hologram.conversation.memory import ConversationMemory, ConversationTurn
from hologram.conversation.patterns import ResponsePatternStore
from hologram.conversation.selector import ResponseCandidate, ResponseSelector
from hologram.conversation.style_tracker import UserStyleTracker
from hologram.conversation.corpus import ResponseCorpus
from hologram.memory.fact_store import FactStore
from hologram.modulation.sesame import SesameModulator, StyleType
from hologram.generation.resonant_generator import ResonantGenerator
from hologram.cognition.metacognition import MetacognitiveLoop
from hologram.config.constants import (
    INTENT_CONFIDENCE_THRESHOLD,
    QUESTION_START_WORDS,
    STOP_WORDS,
    SENTENCE_STARTERS,
    CONVERSATIONAL_MARKERS,
)


class ConversationalChatbot:
    """
    Main conversational chatbot using HDC architecture.

    Orchestrates all conversation components:
    - Intent classification
    - Entity extraction
    - Style tracking
    - Pattern matching
    - Response generation
    - Implicit learning

    Attributes:
        _intent_classifier: For detecting user intent
        _entity_extractor: For finding known entities
        _response_selector: For selecting responses
        _memory: Session conversation memory
        _style_tracker: For adapting to user style
        _pattern_store: For response patterns
        _fact_store: For fact-based answers
        _codebook: Shared Codebook

    Example:
        >>> chatbot = ConversationalChatbot(...)
        >>> greeting = chatbot.start_session()
        >>> response = chatbot.respond("Hello!")
        >>> response = chatbot.respond("What is the capital of France?")
    """

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        entity_extractor: EntityExtractor,
        response_selector: ResponseSelector,
        sesame_modulator: SesameModulator,
        conversation_memory: ConversationMemory,
        style_tracker: UserStyleTracker,
        pattern_store: ResponsePatternStore,
        fact_store: Optional[FactStore],
        codebook: Codebook,
        response_corpus: Optional[ResponseCorpus] = None,
        resonant_generator: Optional[ResonantGenerator] = None,
        metacognitive_loop: Optional[MetacognitiveLoop] = None,
    ):
        """
        Initialize conversational chatbot.

        Args:
            intent_classifier: Intent classification component
            entity_extractor: Entity extraction component
            response_selector: Response selection component
            sesame_modulator: Style modulation component
            conversation_memory: Conversation memory component
            style_tracker: User style tracking component
            pattern_store: Response pattern store
            fact_store: Fact store for knowledge
            codebook: Shared Codebook instance
            response_corpus: Optional response corpus for learned responses
            resonant_generator: Optional generator for token-level generation
            metacognitive_loop: Optional metacognitive loop for self-monitoring and retry
        """
        self._intent_classifier = intent_classifier
        self._entity_extractor = entity_extractor
        self._response_selector = response_selector
        self._sesame = sesame_modulator
        self._memory = conversation_memory
        self._style_tracker = style_tracker
        self._pattern_store = pattern_store
        self._fact_store = fact_store
        self._codebook = codebook
        self._corpus = response_corpus
        self._generator = resonant_generator
        self._metacognitive = metacognitive_loop

        self._last_candidate: Optional[ResponseCandidate] = None

        # Track learning events explicitly for training scripts
        self._last_learned_fact: Optional[tuple] = None  # (subject, predicate, object)
        self._fact_learned_this_turn: bool = False

        # Activity callback for streaming dashboard
        self._activity_callback = None

    def set_activity_callback(self, callback) -> None:
        """
        Set callback for streaming activity events to dashboard.

        Callback receives dict with: {"type": str, ...event_data}
        Event types: "intent", "fact", "thinking"
        """
        self._activity_callback = callback

    def _emit_activity(self, event_type: str, **data) -> None:
        """Emit activity event if callback is set."""
        if self._activity_callback:
            try:
                self._activity_callback({"type": event_type, **data})
            except Exception:
                pass  # Don't let callback errors break chat

    def respond(self, user_input: str) -> str:
        """
        Generate response to user input with continuous learning.

        Args:
            user_input: User's message

        Returns:
            Bot's response string
        """
        user_input = user_input.strip()
        if not user_input:
            return "I didn't catch that. Could you say something?"

        # 1. Classify intent
        intent = self._intent_classifier.classify(user_input)

        # Emit intent activity for dashboard
        self._emit_activity(
            "intent",
            intent=intent.intent.value,
            confidence=round(intent.confidence, 3)
        )

        # Handle commands specially
        if intent.intent == IntentType.COMMAND:
            return self._handle_command(user_input)

        # 2. Check if user is teaching a fact (high-confidence TEACHING intent)
        # Only learn if confidence is high AND it doesn't sound conversational
        is_conversational = any(marker in user_input.lower() for marker in CONVERSATIONAL_MARKERS)
        
        if not is_conversational:
            learned_fact = self._try_learn_from_statement(user_input, intent)
            if learned_fact:
                return learned_fact

        # 3. Extract entities
        entities = self._entity_extractor.extract(user_input)

        # 4. Update style tracker
        self._style_tracker.observe(user_input)
        inferred_style = self._style_tracker.get_inferred_style()

        # 5. Implicit learning from previous turn (Hebbian)
        self._implicit_learning(intent)

        # 6. Select response candidate
        # If metacognitive loop is enabled, it will observe confidence and potentially retry
        candidate = self._response_selector.select(
            intent=intent,
            entities=entities,
            text=user_input,
            style=inferred_style,
        )
        
        # 6b. Metacognitive observation and retry (if enabled)
        if self._metacognitive and candidate.confidence < self._metacognitive.retry_threshold:
            # Low confidence detected - update metacognitive state
            self._metacognitive.state.update_from_confidence(candidate.confidence)
            
            # Retry once with rewired state (if we haven't exceeded max retries)
            # The rewired state will modulate future queries through self_vector
            if self._metacognitive.state.mood.value in ["confused", "anxious"]:
                # Retry with modified query (metacognitive rewiring)
                # Note: The rewiring happens through self_vector which modulates
                # the context vector used in future queries
                retry_candidate = self._response_selector.select(
                    intent=intent,
                    entities=entities,
                    text=user_input,
                    style=inferred_style,
                )
                # Use retry result if it's better
                if retry_candidate.confidence > candidate.confidence:
                    candidate = retry_candidate
                    # Update metacognitive state with improved confidence
                    self._metacognitive.state.update_from_confidence(candidate.confidence)
        elif self._metacognitive:
            # Normal confidence - just observe
            self._metacognitive.state.update_from_confidence(candidate.confidence)

        # 7. Get response text
        response = candidate.filled_response

        # 8. Learn from successful interaction
        # If this is a conversational flow (STATEMENT, GREETING, FAREWELL),
        # strengthen the pattern we're about to use (optimistic learning)
        if intent.intent in {IntentType.STATEMENT, IntentType.GREETING, IntentType.FAREWELL}:
            self._pattern_store.strengthen_pattern(candidate.pattern.pattern_id)

        # 9. Record turn in memory
        self._record_turn(user_input, intent, entities, response, candidate)

        # 10. Store candidate for next turn's learning
        self._last_candidate = candidate

        return response

    def listen(self, user_input: str) -> None:
        """
        Listen to a message without responding (observational learning).
        
        This allows the chatbot to participate in group conversations by listening
        to other speakers, updating context, and learning facts/intents without
        interrupting every turn.
        
        Args:
            user_input: Message to observe
        """
        user_input = user_input.strip()
        if not user_input:
            return

        # 1. Classify intent (for implicit learning context)
        intent = self._intent_classifier.classify(user_input)

        # 2. Try to learn facts (if teaching/statement)
        is_conversational = any(marker in user_input.lower() for marker in CONVERSATIONAL_MARKERS)
        
        if not is_conversational:
            # We ignore the confirmation string since we're just listening
            self._try_learn_from_statement(user_input, intent)

        # 3. Extract entities (updates entity memory)
        entities = self._entity_extractor.extract(user_input)

        # 4. Update style tracker (adapt to group style)
        self._style_tracker.observe(user_input)
        
        # 5. Record turn in memory (updates context vector)
        # We record it as a turn where we didn't respond
        # Create a dummy candidate for recording
        dummy_pattern = self._pattern_store.get_patterns_for_intent(IntentType.UNKNOWN)[0]
        
        self._record_turn(
            user_input=user_input,
            intent=intent,
            entities=entities,
            response="",  # No response
            candidate=ResponseCandidate(
                pattern=dummy_pattern,
                filled_response="", 
                thought_vector=self._codebook._space.empty_vector(), 
                confidence=0.0
            )
        )

    def _try_learn_from_statement(self, text: str, intent: IntentResult) -> Optional[str]:
        """
        Try to learn a fact from a teaching statement using HDC.

        Uses fact extraction on individual sentences. Doesn't require TEACHING intent
        because mentor messages often include questions ("X is Y. What is X?")
        which would be classified as QUESTION overall.

        Returns confirmation message if learned AND no question present.
        Returns None if no fact found OR if message contains a question
        (so the response flow continues to answer the question).
        """
        # Filter out roleplay markers before processing
        # Skip messages with asterisk markers like "*nods*", "*leans back*"
        if "*" in text:
            return None

        # Skip purely conversational intents (greetings, farewells, commands)
        # These should never contain facts to learn
        if intent.intent in {IntentType.GREETING, IntentType.FAREWELL, IntentType.COMMAND}:
            return None

        # Check if message contains a question - if so, we should answer it
        # (after learning any facts)
        text_lower = text.lower()
        has_question = "?" in text or any(
            text_lower.strip().startswith(qw) or f" {qw} " in text_lower
            for qw in QUESTION_START_WORDS
        )

        # Extract fact structure from individual sentences
        # This handles "X is Y. What is X?" by parsing each sentence
        fact = self._extract_fact_structure(text)
        if fact:
            subject, predicate, obj = fact
            # Learn the fact (side-effect) - this will update vocabulary dynamically
            # Returns empty string if duplicate, confirmation if new
            confirmation = self.teach_fact(subject, predicate, obj)

            # ARCHITECTURAL FIX: If message contains BOTH teaching AND question,
            # learn the fact but DON'T return confirmation - let question-answering proceed
            # This enables Quiz Master mode: "X is Y. What is X?" → Learn AND Answer
            if has_question:
                # Fact learned (or duplicate), but continue to answer the question
                return None
            else:
                # Pure teaching statement - return confirmation (empty if duplicate)
                # Empty string means duplicate, which is fine - don't return early
                return confirmation if confirmation else None

        return None

    def _extract_fact_structure(self, text: str) -> Optional[tuple]:
        """
        Extract (subject, predicate, object) from text using HDC operations.

        Handles multiple sentence patterns:
        1. "the capital of France is Paris" → (France, capital, Paris)
        2. "France's capital is Paris" → (France, capital, Paris)
        3. "Ada Lovelace was the first programmer" → (Ada Lovelace, is, first programmer)
        4. "the sky is blue" → (sky, is, blue)
        5. "dogs are mammals" → (dogs, is, mammals)

        Now processes sentences individually and rejects overly long facts.

        Returns (subject, predicate, object) tuple or None.
        """
        import re
        
        # Split into sentences first (process only the first clean sentence)
        sentences = re.split(r'[.!?]+', text)
        
        # Try each sentence until we find a valid fact
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip sentences with roleplay markers
            if "*" in sentence:
                continue
            
            # Skip question sentences - they don't contain facts to learn
            if sentence.endswith("?") or sentence.lower().startswith(tuple(QUESTION_START_WORDS)):
                continue
            
            # Skip sentences that are too long (likely conversational, not factual)
            tokens = self._tokenize(sentence)
            if len(tokens) > 20:  # Reject facts longer than 20 words
                continue
            
            if len(tokens) < 3:
                continue

            # Find relation words
            relation_idx = None
            relation_word = None
            for i, token in enumerate(tokens):
                if token in {"is", "are", "was", "were"}:
                    relation_idx = i
                    relation_word = token
                    break

            if relation_idx is None or relation_idx < 1:
                continue

            before_relation = tokens[:relation_idx]
            after_relation = tokens[relation_idx + 1:]

            if not before_relation or not after_relation:
                continue

            subject = None
            predicate = "is"  # Default predicate
            obj = None

            # Pattern 1: "the X of Y is Z" → (Y, X, Z)
            if "of" in before_relation:
                of_idx = before_relation.index("of")
                if of_idx > 0 and of_idx < len(before_relation) - 1:
                    # Get predicate (word before "of")
                    pred_tokens = [t for t in before_relation[:of_idx] if t not in STOP_WORDS]
                    if pred_tokens:
                        predicate = pred_tokens[-1]  # Last meaningful word before "of"
                    # Get subject (words after "of")
                    subj_tokens = [t for t in before_relation[of_idx + 1:] if t not in STOP_WORDS]
                    if subj_tokens:
                        subject = " ".join(subj_tokens)
                    # Get object (words after relation) - limit to reasonable length
                    obj_tokens = [t for t in after_relation[:10] if t not in STOP_WORDS]  # Max 10 words
                    if obj_tokens:
                        obj = " ".join(obj_tokens)

            # Pattern 2: "X's Y is Z" → (X, Y, Z)
            if subject is None:
                for i, token in enumerate(before_relation):
                    if token.endswith("'s") or token.endswith("s'"):
                        subject = token.rstrip("'").rstrip("s'").rstrip("'s")
                        if not subject:
                            subject = token[:-2] if token.endswith("'s") else token[:-1]
                        # Predicate is words after possessive
                        pred_tokens = [t for t in before_relation[i + 1:] if t not in STOP_WORDS]
                        if pred_tokens:
                            predicate = pred_tokens[0]
                        break

            # Pattern 3: Check for proper noun(s) - use tokenized words, not regex on original
            # This avoids the bug where "Yes" at sentence start is detected as proper noun
            if subject is None:
                # Filter out sentence starters and stop words to find real content words
                meaningful_before = [t for t in before_relation if t not in STOP_WORDS and t not in SENTENCE_STARTERS]
                meaningful_after = [t for t in after_relation[:10] if t not in STOP_WORDS and t not in SENTENCE_STARTERS]
                
                # Use meaningful tokens directly instead of regex proper noun detection
                if meaningful_before:
                    subject = " ".join(meaningful_before)
                    if meaningful_after:
                        obj = " ".join(meaningful_after)

            # Pattern 4: Simple "X is Y" - last meaningful word before relation (fallback)
            if subject is None:
                meaningful_before = [t for t in before_relation if t not in STOP_WORDS and t not in SENTENCE_STARTERS]
                if meaningful_before:
                    subject = meaningful_before[-1]
                elif before_relation:
                    # Last resort: use last token even if it's a stop word
                    subject = before_relation[-1]

            # Extract object if not already set
            if obj is None:
                obj_tokens = [t for t in after_relation[:10] if t not in STOP_WORDS]  # Max 10 words
                if obj_tokens:
                    obj = " ".join(obj_tokens)
                elif after_relation:
                    obj = " ".join(after_relation[:10])  # Limit to first 10 words

            # Validate we got meaningful extractions
            if not subject or not obj:
                continue
            
            # Skip if subject or object is too generic or is a sentence starter
            generic_words = {"thing", "something", "it", "this", "that", "stuff", "one"}
            if subject.lower() in generic_words or obj.lower() in generic_words:
                continue
            if subject.lower() in SENTENCE_STARTERS:
                continue

            # Additional validation: reject if fact components are too long
            # (indicates we captured too much text)
            if len(subject.split()) > 5 or len(obj.split()) > 10:
                continue

            # Title case the subject for consistency
            subject = subject.title()
            obj = obj.strip()

            return (subject, predicate, obj)
        
        # No valid fact found in any sentence
        return None

    def _handle_command(self, user_input: str) -> str:
        """Handle slash commands (pass-through to existing interface)."""
        return f"[Command detected: {user_input}] Use the standard interface for commands."

    def _implicit_learning(self, current_intent: IntentResult) -> None:
        """
        Learn from conversation flow (implicit feedback).

        If user continues naturally (different intent or new question),
        strengthen the previous pattern. If user seems confused or
        repeats, don't strengthen.
        """
        if self._last_candidate is None:
            return

        last_turn = self._memory.get_last_turn()
        if last_turn is None:
            return

        # Detect if user is repeating/rephrasing (negative signal)
        is_repeating = self._detect_repetition(current_intent, last_turn)

        if is_repeating:
            # User confused - weaken the pattern
            self._pattern_store.weaken_pattern(self._last_candidate.pattern.pattern_id)
        else:
            # Conversation flowing - strengthen the pattern
            self._pattern_store.strengthen_pattern(
                self._last_candidate.pattern.pattern_id
            )

    def _detect_repetition(
        self, current_intent: IntentResult, last_turn: ConversationTurn
    ) -> bool:
        """
        Detect if user is repeating or rephrasing (sign of confusion).

        Returns:
            True if user seems to be repeating
        """
        # Check for "what?" or "huh?" type responses (confusion signal)
        if current_intent.intent == IntentType.UNKNOWN:
            return True

        # Same intent type in a row might indicate repetition
        if current_intent.intent == last_turn.intent:
            # Check if confidence is low (user might be rephrasing)
            if current_intent.confidence < 0.3:
                return True

        return False

    def _record_turn(
        self,
        user_input: str,
        intent: IntentResult,
        entities: list,
        response: str,
        candidate: ResponseCandidate,
    ) -> None:
        """Record the conversation turn in memory."""
        # Encode user input
        tokens = self._tokenize(user_input)
        if tokens:
            user_vecs = [self._codebook.encode(t) for t in tokens]
            user_vec = user_vecs[0]
            for v in user_vecs[1:]:
                user_vec = Operations.bundle(user_vec, v)
        else:
            # FIX: Use a semantic null vector instead of zero vector
            # Zero vectors cause "norm is zero" warnings in torchhd binding operations
            user_vec = self._codebook.encode("__EMPTY_INPUT__")

        # Encode response
        resp_tokens = self._tokenize(response)
        if resp_tokens:
            resp_vecs = [self._codebook.encode(t) for t in resp_tokens]
            resp_vec = resp_vecs[0]
            for v in resp_vecs[1:]:
                resp_vec = Operations.bundle(resp_vec, v)
        else:
            # FIX: Use a semantic null vector instead of zero vector
            resp_vec = self._codebook.encode("__EMPTY_RESPONSE__")

        turn = ConversationTurn(
            user_input=user_input,
            user_vector=user_vec,
            intent=intent.intent,
            entities=entities,
            response=response,
            response_vector=resp_vec,
            timestamp=datetime.now(),
            pattern_id=candidate.pattern.pattern_id,
        )

        self._memory.add_turn(turn)

    def _tokenize(self, text: str) -> list:
        """Simple tokenization."""
        import re

        text = re.sub(r"[^\w\s']", " ", text.lower())
        return [t for t in text.split() if t]

    def start_session(self) -> str:
        """
        Start new conversation session.

        Returns:
            Initial greeting message
        """
        self._memory.clear()
        self._style_tracker.reset()
        self._last_candidate = None

        # Start consolidation worker if using neural memory
        if self._fact_store and getattr(self._fact_store, "_consolidation_manager", None):
            self._fact_store._consolidation_manager.start_worker()

        # Return a greeting
        return "Hello! I'm Hologram, a learning chatbot. How can I help you today?"

    def end_session(self) -> None:
        """End conversation session and persist learned patterns."""
        # Stop consolidation worker if using neural memory
        if self._fact_store and getattr(self._fact_store, "_consolidation_manager", None):
            self._fact_store._consolidation_manager.stop_worker()
            
        # Future: persist pattern store to disk
        pass

    def teach_fact(self, subject: str, predicate: str, obj: str) -> str:
        """
        Teach a new fact to the chatbot.

        Args:
            subject: Fact subject
            predicate: Fact predicate
            obj: Fact object

        Returns:
            Confirmation message if new fact, empty string if duplicate
        """
        if self._fact_store:
            # Try to add fact - returns None if duplicate
            fact = self._fact_store.add_fact(subject, predicate, obj, source="conversation")
            
            # Only process if fact is new (not duplicate)
            if fact is not None:
                # Add to entity extractor vocabulary (HDC learning)
                self._entity_extractor.add_entity(subject)
                self._entity_extractor.add_entity(obj)

                # CRITICAL: Update generator vocabulary dynamically
                # This prevents vocabulary death spiral
                if self._generator:
                    # Extract words from fact and add to vocabulary
                    subject_words = subject.lower().split()
                    obj_words = obj.lower().split()
                    new_nouns = [w for w in subject_words + obj_words if len(w) > 2]
                    new_verbs = [predicate.lower()] if len(predicate) > 2 else []

                    self._generator.update_vocabulary({
                        "nouns": new_nouns,
                        "verbs": new_verbs
                    })

                # Track learning for training scripts (explicit protocol)
                # ONLY set flag if fact is actually new
                self._last_learned_fact = (subject, predicate, obj)
                self._fact_learned_this_turn = True

                # Emit fact activity for dashboard
                self._emit_activity(
                    "fact",
                    subject=subject,
                    predicate=predicate,
                    object=obj
                )

                return f"Got it! I'll remember that {subject} {predicate} {obj}."
            else:
                # Duplicate fact - don't set learning flag, return empty string
                # This prevents training script from counting duplicates
                return ""
        return "I can't store facts right now."

    def get_last_learned_fact(self) -> Optional[tuple]:
        """
        Get the last fact learned in this turn.

        Returns:
            (subject, predicate, object) tuple or None
        """
        return self._last_learned_fact

    def clear_learning_flag(self) -> None:
        """Clear the learning flag for next turn."""
        self._fact_learned_this_turn = False
        self._last_learned_fact = None

    def did_learn_fact_this_turn(self) -> bool:
        """Check if a fact was learned this turn."""
        return self._fact_learned_this_turn
    
    def learn_intent_from_context(self, user_input: str, correct_intent: IntentType) -> None:
        """
        Learn to recognize an intent from example (HDC learning).
        
        This allows the chatbot to improve its intent classification over time
        by adding examples to the intent classifier's prototype vectors.
        
        Args:
            user_input: The example input
            correct_intent: The correct intent for this input
        """
        self._intent_classifier.learn(user_input, correct_intent)

    def learn_response(
        self,
        context_vector: torch.Tensor,
        response: str,
        intent: IntentType,
        style: StyleType = StyleType.NEUTRAL,
        source: str = "learned",
    ) -> None:
        """
        Learn a response pattern from training.

        Stores a response with its context vector in the corpus for later retrieval.

        Args:
            context_vector: HDC vector representing the conversation context
            response: The response text to learn
            intent: Intent type for this response
            style: Style type
            source: Source identifier ("claude", "gemini", "learned")
        """
        if self._corpus:
            self._corpus.add_response(
                context_vector=context_vector,
                response=response,
                intent=intent,
                style=style,
                source=source,
            )

    def build_context_vector(self, conversation_history: list) -> torch.Tensor:
        """
        Build a context vector from conversation history.

        Args:
            conversation_history: List of (speaker, message) tuples

        Returns:
            HDC vector representing the conversation context
        """
        if not conversation_history:
            return self._codebook._space.empty_vector()

        # Encode recent messages
        context_vecs = []
        for speaker, message in conversation_history[-5:]:  # Last 5 messages
            tokens = self._tokenize(message)
            if tokens:
                msg_vecs = [self._codebook.encode(t) for t in tokens]
                msg_vec = msg_vecs[0]
                for v in msg_vecs[1:]:
                    msg_vec = Operations.bundle(msg_vec, v)
                # Encode speaker
                speaker_vec = self._codebook.encode(f"__SPEAKER_{speaker}__")
                combined = Operations.bind(msg_vec, speaker_vec)
                context_vecs.append(combined)

        if not context_vecs:
            # Return a deterministic "null context" vector instead of zeros
            return self._codebook._space.random_vector(0)

        # Bundle all context vectors
        result = context_vecs[0]
        for vec in context_vecs[1:]:
            result = Operations.bundle(result, vec)

        return result

    def get_session_stats(self) -> dict:
        """Get statistics about the current session."""
        stats = {
            "turns": self._memory.turn_count,
            "inferred_style": self._style_tracker.get_inferred_style().value,
            "style_confidence": self._style_tracker.get_style_confidence(),
            "patterns_count": self._pattern_store.pattern_count,
            "messages_observed": self._style_tracker.message_count,
        }
        if self._corpus:
            stats["corpus_entries"] = self._corpus.get_entry_count()
        if self._generator:
            stats["generator_enabled"] = True
        return stats

    def save_memory(self, persist_dir: str, force_consolidation: bool = True) -> bool:
        """
        Save memory state to disk (if using neural consolidation).
        
        Args:
            persist_dir: Directory to save to
            force_consolidation: If True, force consolidation of pending facts before saving
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self._fact_store:
            consolidation_manager = getattr(self._fact_store, "_consolidation_manager", None)
            
            # Force consolidation of pending facts before saving
            if force_consolidation and consolidation_manager:
                pending_before = consolidation_manager.pending_count
                if pending_before > 0:
                    print(f"  Consolidating {pending_before} pending facts before save...")
                    consolidated_before = consolidation_manager.total_consolidated
                    consolidation_manager.force_consolidation()
                    
                    # Wait for consolidation to complete (check if consolidation_count increased)
                    # Note: pending_facts is cleared immediately when scheduled, so we wait
                    # for the consolidation to actually complete by checking total_consolidated
                    import time
                    max_wait = 30.0  # 30 second timeout
                    wait_start = time.time()
                    while (consolidation_manager.total_consolidated == consolidated_before and 
                           (time.time() - wait_start) < max_wait):
                        time.sleep(0.5)
                    
                    if consolidation_manager.total_consolidated == consolidated_before:
                        print(f"  ⚠️  Warning: Consolidation may not have completed (timeout)")
                    else:
                        newly_consolidated = consolidation_manager.total_consolidated - consolidated_before
                        print(f"  ✓ Consolidation complete ({newly_consolidated} facts consolidated)")
            
            state = getattr(self._fact_store, "get_state_dict", lambda: None)()
            if state:
                import torch
                from pathlib import Path
                path = Path(persist_dir) / "neural_memory.pt"
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                torch.save(state, path)
                vocab_size = len(state.get("value_vocab", {}))
                consolidated = state.get("total_consolidated", 0)
                pending = len(state.get("pending_facts", []))
                print(f"💾 Saved neural memory: {path} ({vocab_size} vocab, {consolidated} consolidated, {pending} pending)")
                return True
        return False
                
    def __repr__(self) -> str:
        return f"ConversationalChatbot(turns={self._memory.turn_count})"

"""
Response selection and preparation.

Selects the best response based on intent, entities, context, and facts.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.conversation.entity import Entity
from hologram.conversation.intent import IntentResult, IntentType
from hologram.conversation.memory import ConversationMemory
from hologram.conversation.patterns import ResponsePattern, ResponsePatternStore
from hologram.conversation.corpus import ResponseCorpus
from hologram.memory.fact_store import FactStore
from hologram.modulation.sesame import StyleType
from hologram.generation.resonant_generator import ResonantGenerator
from hologram.generation.circuit_breaker import SimpleCircuitBreaker
from hologram.generation.base import GenerationContext


@dataclass
class ResponseCandidate:
    """A prepared response candidate."""

    pattern: ResponsePattern
    filled_response: str  # Template with slots filled
    thought_vector: torch.Tensor  # Vector for generation
    confidence: float
    fact_answer: Optional[str] = None  # Answer from fact store if applicable

    def __repr__(self) -> str:
        return f"ResponseCandidate('{self.filled_response[:30]}...', conf={self.confidence:.2f})"


class ResponseSelector:
    """
    Select and prepare response based on intent, entities, and context.

    Orchestrates pattern matching, fact retrieval, and template filling
    to produce the best response candidate.

    Attributes:
        _pattern_store: ResponsePatternStore for pattern matching
        _memory: ConversationMemory for context
        _fact_store: FactStore for fact lookup
        _codebook: Codebook for encoding

    Example:
        >>> selector = ResponseSelector(patterns, memory, facts, codebook)
        >>> candidate = selector.select(intent_result, entities, text)
        >>> print(candidate.filled_response)
    """

    def __init__(
        self,
        pattern_store: ResponsePatternStore,
        conversation_memory: ConversationMemory,
        fact_store: Optional[FactStore],
        codebook: Codebook,
        response_corpus: Optional[ResponseCorpus] = None,
        resonant_generator: Optional[ResonantGenerator] = None,
        ventriloquist_generator: Optional[object] = None,  # VentriloquistGenerator
    ):
        """
        Initialize response selector.

        Args:
            pattern_store: Pattern store for matching
            conversation_memory: Memory for context
            fact_store: Fact store for knowledge lookup
            codebook: Shared Codebook instance
            response_corpus: Optional corpus for learned responses
            resonant_generator: Optional HDC generator for factual questions
            ventriloquist_generator: Optional SLM generator for fluent conversation
        """
        self._pattern_store = pattern_store
        self._memory = conversation_memory
        self._fact_store = fact_store
        self._codebook = codebook
        self._corpus = response_corpus
        self._generator = resonant_generator
        self._ventriloquist = ventriloquist_generator
        
        # Initialize circuit breaker for generation failure detection
        self._circuit_breaker = SimpleCircuitBreaker(
            failure_threshold=3,
            window_size=10,
            cooldown_seconds=60.0
        )

    def select(
        self,
        intent: IntentResult,
        entities: List[Entity],
        text: str,
        style: Optional[StyleType] = None,
    ) -> ResponseCandidate:
        """
        Select best response for the given input.

        Args:
            intent: Classified intent result
            entities: Extracted entities
            text: Original user input
            style: Optional preferred style

        Returns:
            Best ResponseCandidate
        """
        # Get context from memory
        context_vec = self._memory.get_context_vector()

        # Get entity canonical forms
        entity_names = [e.canonical_form for e in entities]

        # Check for follow-up context (e.g., "And Germany?" after capital question)
        entity_names = self._enrich_with_context(entity_names, text)

        # Step 1: Try to answer from fact store if it's a question
        # CRITICAL FIX: Also check for questions in TEACHING/STATEMENT intents (mixed intent)
        # "The capital is Paris. What is the capital?" -> TEACHING but contains question
        is_question = intent.intent == IntentType.QUESTION or "?" in text
        
        fact_answer = None
        fact_confidence = 0.0
        if is_question and self._fact_store:
            result = self._query_facts(entity_names, text)
            if result:
                fact_answer, fact_confidence = result

        # Step 2: Hybrid generation routing
        # - Factual questions with high confidence: Use ResonantGenerator (HDC-native)
        # - Conversational or low confidence: Use VentriloquistGenerator (SLM for fluency)
        generated_response = None
        has_facts = self._fact_store and self._fact_store.fact_count > 0
        confidence_threshold = 0.5  # Minimum confidence for HDC generation
        is_factual_question = is_question and fact_answer is not None

        # Try generation if we have facts or if ventriloquist is available for conversation
        if (fact_answer and has_facts) or (self._ventriloquist and not is_question):
            # Build GenerationContext for unified interface
            thought_vec = None
            if fact_answer:
                thought_vec = self._create_thought_vector(intent.intent, entity_names, fact_answer)
            expected_subject = self._extract_expected_subject(entity_names)
            
            context = GenerationContext(
                query_text=text,
                thought_vector=thought_vec,
                intent=intent.intent,
                fact_answer=fact_answer,
                entities=entity_names,
                style=style or StyleType.NEUTRAL,
                expected_subject=expected_subject,
            )
            
            # Hybrid routing: Prioritize Ventriloquist for fluency + grounding
            if self._ventriloquist:
                # Always prefer Ventriloquist if available (provides both fluency and fact grounding)
                generated_response = self._generate_response_with_context(context, use_ventriloquist=True)
            elif is_factual_question and fact_confidence >= confidence_threshold and self._generator:
                # Fallback to HDC generator (bounded hallucination, but robotic output)
                generated_response = self._generate_response_with_context(context)

        # Step 3: Check ResponseCorpus for learned responses (deprioritized)
        corpus_response = None
        corpus_rejected = False  # Track if we found but rejected a response
        
        if self._corpus and not generated_response:  # Skip corpus if generation succeeded
            # IMPORTANT: Query vector MUST match what was used during learning!
            # Learning uses build_context_vector() which bundles conversation history.
            # So we use the same context_vec from memory, not thought_vector.
            if context_vec is not None:
                corpus_matches = self._corpus.retrieve(context_vec, intent.intent, style, top_k=3)
                # Lower threshold - HDC bundled vectors have lower similarity due to interference
                if corpus_matches and corpus_matches[0][1] > 0.32:
                    # Deduplication: Don't repeat recent messages
                    # Get last 5 messages from memory
                    recent_messages = self._memory.get_recent_messages(5)
                    
                    for match_response, score in corpus_matches:
                         if score < 0.32:
                             break
                             
                         # Check if this exact response was used recently (by anyone)
                         # IMPORTANT: Check against current 'text' (user input) too, as it might
                         # not be in memory yet if we are in the middle of processing the turn!
                         # Use Jaccard overlap for fuzzy matching
                         is_duplicate = False
                         check_texts = recent_messages + [text]
                         
                         # Check strict substring first (fast)
                         if any(match_response.strip().lower() in msg.strip().lower() or 
                                msg.strip().lower() in match_response.strip().lower() 
                                for msg in check_texts):
                             is_duplicate = True
                         
                         # Check semantic overlap (robust)
                         if not is_duplicate:
                             for msg in check_texts:
                                 if self._check_overlap(match_response, msg, threshold=0.6):
                                     is_duplicate = True
                                     break
                         
                         if not is_duplicate:
                             corpus_response = match_response
                             break
                         else:
                             corpus_rejected = True  # We found something relevant but it was a duplicate

        # Step 4: Match patterns (if generation and corpus didn't have a good match)
        matches = []
        if not generated_response and not corpus_response:
            matches = self._pattern_store.match(
                intent=intent.intent,
                entities=entity_names,
                context_vec=context_vec,
                style=style,
            )
            
            # If we rejected a corpus match (meaning we understood the context but didn't want to repeat),
            # inject VARIED continuation patterns to ensure diversity.
            # The repetition penalty will naturally rotate through them.
            if corpus_rejected:
                continuation_ids = [
                    "statement_curious",      # "Interesting! Tell me more about that."
                    "listening_continue",     # "Go on, I'm listening."
                    "statement_agree",        # "That makes sense. What else?"
                    "listening_encourage",    # "Tell me more!"
                    "statement_thoughtful",   # "That's a good point..."
                    "listening_understand",   # "I think I understand. What else?"
                    "statement_engaged",      # "Oh, that's fascinating! Go on."
                    "listening_absorb",       # "Okay, I'm taking that in."
                    "statement_reflective",   # "Hmm, that gives me something to think about."
                    "listening_process",      # "That's new to me. Keep going."
                    "statement_followup",     # "That's cool. What made you think of that?"
                ]
                
                # Add all continuation patterns as candidates with base score
                # The repetition penalty will handle variety
                existing_ids = {p.pattern_id for p, _ in matches}
                for pattern_id in continuation_ids:
                    if pattern_id not in existing_ids:
                        pattern = self._pattern_store.get_pattern(pattern_id)
                        if pattern:
                            # Give them competitive scores so they can win
                            matches.append((pattern, pattern.strength * 0.8))
        
        # DEBUG: Uncomment to see intent matching details
        # print(f"DEBUG: Intent={intent.intent}, Found {len(matches)} matches")

        # Apply repetition penalty and fact-based adjustments
        if matches:
            # Get recently used pattern IDs from memory
            recent_pattern_ids = self._memory.get_recent_patterns(lookback=5)
            # print(f"DEBUG: Recent patterns: {recent_pattern_ids}")
            
            adjusted_matches = []
            for pattern, score in matches:
                # 1. Fact availability adjustment
                has_answer_slot = "{answer}" in pattern.response_template
                if fact_answer and has_answer_slot:
                    # We have an answer, strongly prefer patterns that show it
                    score *= 2.0
                elif not fact_answer and has_answer_slot:
                    # We DON'T have an answer, heavily penalize patterns that need one
                    score *= 0.01
                elif not fact_answer and not has_answer_slot:
                    # We don't have an answer BUT this pattern doesn't need one - boost it!
                    score *= 1.5
                
                # 2. Repetition penalty (Habituation)
                # If pattern was used recently, drastically reduce its score
                if pattern.pattern_id in recent_pattern_ids:
                    # Penalize more for very recent usage
                    recency_index = recent_pattern_ids.index(pattern.pattern_id) if pattern.pattern_id in recent_pattern_ids else -1
                    if recency_index == 0: # Used in very last turn
                         score *= 0.05
                    elif recency_index == 1: # Used 2 turns ago
                         score *= 0.1
                    else:
                         score *= 0.3
                
                adjusted_matches.append((pattern, score))
            
            # Re-sort matches
            matches = sorted(adjusted_matches, key=lambda x: x[1], reverse=True)
            # DEBUG: Uncomment to see pattern scoring
            # print(f"DEBUG: Top 3 matches:")
            # for p, s in matches[:3]:
            #     print(f"  - {p.pattern_id}: {s:.4f}")

        # Select best response: Priority order: Generated > FactAnswer > Corpus > Patterns > Fallback
        if generated_response:
            # Use fact-based generated response (highest priority)
            best_pattern = self._get_fallback_pattern(intent.intent)
            filled_response = generated_response
            score = 0.8  # High confidence for fact-based generation
        elif fact_answer:
            # We have a fact answer - use it directly with an appropriate template
            # This handles the case where HDC pattern matching fails but we found the answer
            best_pattern = self._get_answer_pattern(intent.intent, entity_names)
            filled_response = self._fill_template(
                best_pattern.response_template,
                entities,
                fact_answer,
                entity_names,
            )
            score = 0.75  # High confidence for direct fact answer
        elif corpus_response:
            # Use learned response from corpus (deprioritized)
            best_pattern = self._get_fallback_pattern(intent.intent)  # Dummy pattern for structure
            filled_response = corpus_response
            score = 0.6  # Lower confidence than generation
        elif matches:
            # Use pattern template
            best_pattern, score = matches[0]
            # Fill template
            filled_response = self._fill_template(
                best_pattern.response_template,
                entities,
                fact_answer,
                entity_names,
            )
        else:
            # Ultimate fallback: use pattern templates (NOT generator)
            # Generator without facts produces garbage - only use it with fact_answer
            best_pattern = self._get_fallback_pattern(intent.intent)
            filled_response = self._fill_template(
                best_pattern.response_template,
                entities,
                fact_answer,
                entity_names,
            )
            score = 0.2

        # Create thought vector for potential generation
        thought_vec = self._create_thought_vector(
            intent.intent, entity_names, fact_answer
        )

        return ResponseCandidate(
            pattern=best_pattern,
            filled_response=filled_response,
            thought_vector=thought_vec,
            confidence=score,
            fact_answer=fact_answer,
        )

    def _enrich_with_context(self, entity_names: List[str], text: str) -> List[str]:
        """
        Enrich entities with context from previous turns.

        Handles cases like "And Germany?" where predicate is implied.
        """
        # Get topic entities from memory
        topic_entities = self._memory.get_topic_entities()

        # Get last intent/entities
        last_entities = self._memory.get_last_entities()

        # If current input is short and has new entity, inherit predicate
        words = text.lower().split()
        short_input = len(words) <= 4

        if short_input and last_entities:
            # Check if any last entity predicates should be inherited
            for last_e in last_entities:
                # Add predicates from last turn as context
                if last_e.canonical_form not in entity_names:
                    # This might be the implied predicate
                    if last_e.entity_type == "known_concept":
                        entity_names.append(last_e.canonical_form)

        return entity_names

    def _query_facts(self, entity_names: List[str], text: str) -> Optional[tuple[str, float]]:
        """
        Query fact store for an answer.

        Args:
            entity_names: Entities to query about
            text: Original question text

        Returns:
            (answer, confidence) tuple if found, None otherwise
        """
        if not self._fact_store or not entity_names:
            return None

        # Helper to try multiple case variants
        # Returns (answer, confidence) tuple
        def try_query(subject: str, predicate: str) -> Optional[tuple[str, float]]:
            # Try original, capitalized, and lowercase
            best_answer = None
            best_conf = 0.0
            for variant in [subject, subject.capitalize(), subject.lower()]:
                answer, conf = self._fact_store.query(variant, predicate)
                if conf > 0.1 and conf > best_conf:
                    best_answer = answer
                    best_conf = conf
            if best_answer:
                return (best_answer, best_conf)
            return None

        # Try to parse question structure
        # Common patterns: "What is X?", "What is the Y of X?"
        text_lower = text.lower()

        # Look for "capital of X" pattern
        # Skip question words when looking for subject
        question_words = {"what", "who", "where", "when", "why", "how", "which", "whose"}
        if "capital" in text_lower:
            for entity in entity_names:
                entity_lower = entity.lower()
                # Skip question words and the predicate word itself
                if entity_lower in question_words or entity_lower == "capital":
                    continue
                result = try_query(entity, "capital")
                if result:
                    return result

        # Look for "X is" pattern (subject query)
        # Skip question words and common predicate words
        skip_words = question_words | {"is", "are", "was", "were", "capital", "creator", "color", "shape"}
        for entity in entity_names:
            entity_lower = entity.lower()
            if entity_lower in skip_words:
                continue
            # Try common predicates
            for predicate in ["is", "capital", "creator", "color", "shape"]:
                result = try_query(entity, predicate)
                if result:
                    return result

        # Try with context from last turn
        last_entities = self._memory.get_last_entities()
        for last_e in last_entities:
            for entity in entity_names:
                entity_lower = entity.lower()
                if entity_lower in skip_words:
                    continue
                # Try using last entity's predicate
                result = try_query(entity, last_e.canonical_form)
                if result:
                    return result

        return None

    def _fill_template(
        self,
        template: str,
        entities: List[Entity],
        fact_answer: Optional[str],
        entity_names: List[str],
    ) -> str:
        """
        Fill template slots with values.

        Args:
            template: Response template with {slots}
            entities: Extracted entities
            fact_answer: Answer from fact store
            entity_names: Entity canonical forms

        Returns:
            Filled template string
        """
        result = template

        # Fill {answer} slot
        if "{answer}" in result:
            if fact_answer:
                result = result.replace("{answer}", fact_answer)
            else:
                # Use a better fallback when we don't know the answer
                result = result.replace("{answer}", "I don't know that yet")

        # Fill {entity} slot with first entity
        if "{entity}" in result:
            if entity_names:
                # Use the most relevant entity (not a predicate word)
                found_suitable = False
                for name in entity_names:
                    if name.lower() not in ["what", "who", "where", "when", "capital", "is"]:
                        result = result.replace("{entity}", name.capitalize())
                        found_suitable = True
                        break

                # If no suitable entity found, use generic fallback
                if not found_suitable:
                    result = result.replace("{entity}", "that")
            else:
                result = result.replace("{entity}", "that")

        # Fill {subject} and {object} if present
        if "{subject}" in result and entity_names:
            # Find first suitable entity (same logic as {entity})
            skip_words = ["what", "who", "where", "when", "capital", "is"]
            subject_filled = False
            for name in entity_names:
                if name.lower() not in skip_words:
                    result = result.replace("{subject}", name.capitalize())
                    subject_filled = True
                    break
            # If no suitable subject, use fallback
            if not subject_filled:
                result = result.replace("{subject}", "that")

        if "{object}" in result and fact_answer:
            result = result.replace("{object}", fact_answer)

        return result

    def _get_answer_pattern(self, intent: IntentType, entity_names: List[str]) -> ResponsePattern:
        """Get a pattern that uses {answer} slot for fact-based responses."""
        patterns = self._pattern_store.get_patterns_for_intent(intent)
        
        # Look for patterns with {answer} slot that match entity requirements
        for p in patterns:
            if "{answer}" in p.response_template:
                # Check if entity pattern matches
                if not p.entity_pattern:
                    # No entity requirement - good fallback
                    return p
                # Check if any entity matches
                entity_match = any(
                    e.lower() in [x.lower() for x in entity_names]
                    for e in p.entity_pattern
                )
                if entity_match:
                    return p
        
        # Fallback: create a simple answer pattern
        # Choose template based on entity context
        if "capital" in [e.lower() for e in entity_names]:
            template = "The capital of {entity} is {answer}."
        elif "who" in [e.lower() for e in entity_names]:
            template = "{entity} is {answer}."
        elif "where" in [e.lower() for e in entity_names]:
            template = "{entity} is located in {answer}."
        else:
            template = "{answer}"
        
        return ResponsePattern(
            pattern_id="dynamic_answer",
            intent=intent,
            entity_pattern=[],
            response_template=template,
            style=StyleType.NEUTRAL,
        )

    def _get_fallback_pattern(self, intent: IntentType) -> ResponsePattern:
        """Get a fallback pattern for the intent."""
        patterns = self._pattern_store.get_patterns_for_intent(intent)
        if not patterns:
             # Ultimate fallback
            unknown_patterns = self._pattern_store.get_patterns_for_intent(IntentType.UNKNOWN)
            if unknown_patterns:
                return unknown_patterns[0]

            # Create minimal pattern
            return ResponsePattern(
                pattern_id="fallback",
                intent=IntentType.UNKNOWN,
                entity_pattern=[],
                response_template="I'm not sure how to respond to that.",
                style=StyleType.NEUTRAL,
            )

        # Prefer patterns that don't require an answer/slots if we are falling back
        # Look for specific fallback patterns first
        for pid in ["no_information", "unknown_rephrase", "statement_acknowledge", "greeting_simple"]:
            for p in patterns:
                if p.pattern_id == pid:
                    return p
        
        # Look for patterns without slots
        for p in patterns:
            if "{" not in p.response_template:
                return p

        # Default to first available
        return patterns[0]

    def _create_thought_vector(
        self,
        intent: IntentType,
        entity_names: List[str],
        fact_answer: Optional[str],
    ) -> torch.Tensor:
        """
        Create structured thought vector for generation.
        
        Constructs thought vector using role-based encoding:
        Thought = (Subject * SUBJECT_ROLE) + (Predicate * PREDICATE_ROLE) + (Object * OBJECT_ROLE)
        
        This matches the structure expected by ResonantGenerator's resonator.
        """
        # Get role vectors
        role_subject = self._codebook.get_role("SUBJECT")
        role_verb = self._codebook.get_role("VERB")
        role_object = self._codebook.get_role("OBJECT")
        
        # Extract Subject, Predicate (Verb), and Object
        subject_str = None
        predicate_str = "is"  # Default predicate
        object_str = fact_answer if fact_answer else None
        
        # Parse entities to identify subject and predicate
        if entity_names:
            # Words to skip when finding subject (same as _extract_expected_subject)
            skip_words = {
                # Question words
                "what", "who", "where", "when", "why", "how", "which",
                # Predicate words  
                "capital", "is", "are", "was", "were", "creator", "color", "shape",
                "located", "used", "currency", "language", "continent", "country",
                "city", "region", "hemisphere", "population", "largest", "smallest",
                # Common conversation starters (look like proper nouns but aren't)
                "yes", "no", "okay", "ok", "sure", "let", "try", "again", "correct",
                "incorrect", "right", "wrong", "good", "excellent", "great", "nice",
                "i", "you", "we", "they", "he", "she", "it", "this", "that",
                "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
            }
            predicate_words = {"capital", "is", "creator", "color", "shape", "currency", "language"}
            
            # Subject is the first proper noun (capitalized) that isn't in skip_words
            for entity in entity_names:
                entity_lower = entity.lower()
                if entity_lower in skip_words or len(entity_lower) < 3:
                    # But check if it's a predicate word to capture
                    if entity_lower in predicate_words:
                        predicate_str = entity_lower
                    continue
                # Found a valid subject candidate
                if entity[0].isupper() and subject_str is None:
                    subject_str = entity
                    break
            
            # Fallback: if no capitalized subject found, use first non-skip entity
            if subject_str is None:
                for entity in entity_names:
                    entity_lower = entity.lower()
                    if entity_lower not in skip_words and len(entity_lower) >= 3:
                        subject_str = entity
                        break
        
        # If we have fact_answer, construct structured vector
        if fact_answer and subject_str:
            # Encode components
            subject_vec = self._codebook.encode(subject_str)
            predicate_vec = self._codebook.encode(predicate_str)
            object_vec = self._codebook.encode(object_str)
            
            # Bind with roles: (Token * Role)
            subject_role_vec = Operations.bind(subject_vec, role_subject)
            predicate_role_vec = Operations.bind(predicate_vec, role_verb)
            object_role_vec = Operations.bind(object_vec, role_object)
            
            # Bundle components: Thought = Subject + Predicate + Object
            thought = Operations.bundle(subject_role_vec, predicate_role_vec, object_role_vec)
            return thought
        
        # Fallback: if no fact_answer, use intent + entities (less structured)
        intent_vec = self._codebook.encode(f"__INTENT_{intent.value}__")
        if entity_names:
            entity_vecs = [self._codebook.encode(e) for e in entity_names]
            entity_vec = entity_vecs[0]
            for v in entity_vecs[1:]:
                entity_vec = Operations.bundle(entity_vec, v)
            return Operations.bind(intent_vec, entity_vec)
        
        return intent_vec

    def _check_overlap(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """
        Check if two texts have significant word overlap.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Jaccard similarity threshold (0.0 to 1.0)
            
        Returns:
            True if overlap exceeds threshold
        """
        def tokenize(t):
            # Simple word tokenization, removing short words
            return set(w.lower() for w in t.split() if len(w) > 3)
            
        s1 = tokenize(text1)
        s2 = tokenize(text2)
        
        if not s1 or not s2:
            return False
            
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        
        if union == 0:
            return False
            
        return (intersection / union) > threshold

    def _generate_response_with_context(
        self,
        context: GenerationContext,
        use_ventriloquist: bool = False,
    ) -> Optional[str]:
        """
        Generate response using either ResonantGenerator or VentriloquistGenerator.
        
        Args:
            context: GenerationContext with all information
            use_ventriloquist: If True, use SLM generator; if False, use HDC generator
            
        Returns:
            Generated text or None if generation fails
        """
        # CHECK CIRCUIT BREAKER FIRST - if tripped, skip generation
        if self._circuit_breaker.is_open():
            return None  # Fall back to templates

        generator = self._ventriloquist if use_ventriloquist else self._generator
        if not generator:
            return None

        try:
            # Generate with validation using unified interface
            result = generator.generate_with_validation(
                context=context,
                max_tokens=10 if not use_ventriloquist else 256
            )

            # Return generated text if validation passed
            if result and result.text and len(result.text.strip()) > 0:
                self._circuit_breaker.record(failed=False)  # SUCCESS
                return result.text.strip()
        except Exception:
            # Generation failed - return None to fall back to templates
            pass

        # Record failure (either exception or validation failed)
        self._circuit_breaker.record(failed=True)  # FAILURE
        return None
    
    def _generate_response(
        self,
        intent: IntentType,
        entity_names: List[str],
        fact_answer: Optional[str],
        style: Optional[StyleType],
    ) -> Optional[str]:
        """
        Legacy method for backward compatibility.
        
        Builds GenerationContext and calls _generate_response_with_context.
        """
        thought_vec = self._create_thought_vector(intent, entity_names, fact_answer)
        expected_subject = self._extract_expected_subject(entity_names)
        
        context = GenerationContext(
            query_text="",  # Legacy: no query text available
            thought_vector=thought_vec,
            intent=intent,
            fact_answer=fact_answer,
            entities=entity_names,
            style=style or StyleType.NEUTRAL,
            expected_subject=expected_subject,
        )
        
        return self._generate_response_with_context(context, use_ventriloquist=False)
    
    def _extract_expected_subject(self, entity_names: List[str]) -> Optional[str]:
        """
        Extract the expected subject from entity names for validation.
        
        Returns the first proper noun (capitalized entity) that isn't a question word
        or predicate word. This should match what _create_thought_vector uses.
        """
        if not entity_names:
            return None
        
        # Words to skip (lowercase for comparison) - same logic as _create_thought_vector
        skip_words = {
            # Question words
            "what", "who", "where", "when", "why", "how", "which",
            # Predicate words  
            "capital", "is", "are", "was", "were", "creator", "color", "shape",
            "located", "used", "currency", "language", "continent", "country",
            "city", "region", "hemisphere", "population", "largest", "smallest",
            # Common conversation starters (CRITICAL: these look like proper nouns but aren't)
            "yes", "no", "okay", "ok", "sure", "let", "try", "again", "correct",
            "incorrect", "right", "wrong", "good", "excellent", "great", "nice",
            "i", "you", "we", "they", "he", "she", "it", "this", "that",
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
        }
        
        for entity in entity_names:
            entity_lower = entity.lower()
            # Skip if in stop words OR if it's too short to be meaningful
            if entity_lower in skip_words or len(entity_lower) < 3:
                continue
            # Only accept proper nouns (capitalized) that are likely country/place names
            if entity[0].isupper():
                return entity
        
        # Fallback: return first non-stop-word entity
        for entity in entity_names:
            entity_lower = entity.lower()
            if entity_lower not in skip_words and len(entity_lower) >= 3:
                return entity
        
        return None

    def __repr__(self) -> str:
        corpus_info = f", corpus={self._corpus.get_entry_count() if self._corpus else 0}" if self._corpus else ""
        generator_info = ", generator=enabled" if self._generator else ""
        return f"ResponseSelector(patterns={self._pattern_store.pattern_count}{corpus_info}{generator_info})"

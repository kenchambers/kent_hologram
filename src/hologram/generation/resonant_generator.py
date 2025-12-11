"""
ResonantGenerator: Main orchestration loop for constrained generation.

The ResonantGenerator combines all Resonant Cavity components into a
unified token generation pipeline:
1. Resonator - factorizes thought into (subject, verb, object)
2. TargetEncoder - packages constraints
3. ReEncoder - projects generated tokens back to HDC
4. DivergenceCalculator - verifies alignment
5. SesameModulator - adds style and disfluency

This implements the closed-loop verification that prevents hallucination
while enabling creative combinations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

import torch

from hologram.cavity.divergence import (
    DivergenceAction,
    DivergenceCalculator,
    DivergenceResult,
)
from hologram.cavity.re_encoder import ReEncoder
from hologram.cavity.target_encoder import TargetEncoder, TargetPackage
from hologram.config.constants import CANDIDATE_K, MAX_GENERATION_TOKENS
from hologram.core.codebook import Codebook
from hologram.core.resonator import Resonator, ResonatorResult
from hologram.core.similarity import Similarity
from hologram.modulation.sesame import FillerType, SesameModulator, StyleType
from hologram.generation.jazz import JazzTemplate, StructureType
from hologram.generation.dreamer import Dreamer
from hologram.generation.base import GenerationContext


@dataclass
class GenerationTrace:
    """Single step in generation trace.

    Attributes:
        token: Generated token text
        type: "TOKEN" or "FILLER"
        similarity: Alignment score with target
        action: Divergence action taken
        role: Current grammatical role
        reason: Optional explanation
    """
    token: str
    type: str
    similarity: float
    action: DivergenceAction
    role: str
    reason: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.type}] {self.token} ({self.action.value}, sim={self.similarity:.3f})"


@dataclass
class GenerationMetrics:
    """Metrics for monitoring and auditing.

    Attributes:
        total_tokens: Total tokens generated
        accepted_first_try: Tokens accepted without correction
        accepted_with_correction: Tokens accepted with correction
        rejected_tokens: Tokens rejected (resampled)
        fillers_injected: Number of disfluency tokens
        average_similarity: Mean alignment score
        convergence_iterations: Resonator iterations
        fallback_count: Times fallback was used
    """
    total_tokens: int = 0
    accepted_first_try: int = 0
    accepted_with_correction: int = 0
    rejected_tokens: int = 0
    fillers_injected: int = 0
    average_similarity: float = 0.0
    convergence_iterations: int = 0
    fallback_count: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Percentage of tokens accepted on first try."""
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_first_try / self.total_tokens

    @property
    def hallucination_risk(self) -> float:
        """Indicator of potential hallucination (fallback usage)."""
        if self.total_tokens == 0:
            return 0.0
        return self.fallback_count / self.total_tokens


@dataclass
class GenerationResult:
    """Complete generation output.

    Attributes:
        text: Final generated text
        tokens: Token sequence (without fillers for clean output)
        trace: Full generation trace for debugging
        divergence_history: All divergence calculations
        metrics: Performance metrics
        resonator_result: Original resonator output
    """
    text: str
    tokens: List[str]
    trace: List[GenerationTrace]
    divergence_history: List[DivergenceResult]
    metrics: GenerationMetrics
    resonator_result: Optional[ResonatorResult] = None

    def __str__(self) -> str:
        return f'GenerationResult("{self.text}", acceptance={self.metrics.acceptance_rate:.1%})'


class ResonantGenerator:
    """
    Main generation loop orchestrating resonant cavity.

    Combines Resonator, TargetEncoder, ReEncoder, DivergenceCalculator,
    and SesameModulator into a unified token generation pipeline.

    The generation process:
    1. Stage 1 (Holographic Reasoning): Resonator factorizes thought
    2. Stage 2 (Constrained Generation): Token-by-token with verification

    Attributes:
        _resonator: Resonator for thought factorization
        _target_encoder: TargetEncoder for constraint packaging
        _re_encoder: ReEncoder for token projection
        _divergence: DivergenceCalculator for verification
        _sesame: SesameModulator for style
        _vocabulary: Word vocabularies {"nouns": [...], "verbs": [...]}

    Example:
        >>> generator = container.create_resonant_generator(vocabulary)
        >>> result = generator.generate(thought_vector, StyleType.FORMAL)
        >>> print(result.text)
    """

    def __init__(
        self,
        resonator: Resonator,
        target_encoder: TargetEncoder,
        re_encoder: ReEncoder,
        divergence_calculator: DivergenceCalculator,
        sesame_modulator: SesameModulator,
        vocabulary: Dict[str, List[str]],
    ):
        """
        Initialize generator with all components.

        Args:
            resonator: Resonator for factorization
            target_encoder: TargetEncoder for packaging
            re_encoder: ReEncoder for projection
            divergence_calculator: DivergenceCalculator for verification
            sesame_modulator: SesameModulator for style
            vocabulary: Dict with "nouns" and "verbs" lists
        """
        self._resonator = resonator
        self._target_encoder = target_encoder
        self._re_encoder = re_encoder
        self._divergence = divergence_calculator
        self._sesame = sesame_modulator
        self._vocabulary = vocabulary

        # Initialize Dreamer for creative exploration
        self._dreamer = Dreamer(
            resonator=self._resonator,
            confidence_threshold=0.4, # Start dreaming if confidence < 0.4
        )

        # Pre-compile regex patterns for performance (hot path optimization)
        self._REPETITION_PATTERN = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
        self._GARBAGE_PATTERNS = [
            re.compile(r'\b(unknown|algorithm)\s+(do|does|have|has|are|is)\s+', re.IGNORECASE),
            re.compile(r'\b(\w+)\s+(do|does|have|has|are|is)\s+\1\b', re.IGNORECASE),
            re.compile(r'\b(people|person)\s+(do|does|have|has)\s+', re.IGNORECASE),
        ]

    def update_vocabulary(self, new_words: Dict[str, List[str]]) -> None:
        """
        Update vocabulary dynamically after initialization.

        This allows the generator to learn new words as facts are taught,
        preventing the vocabulary death spiral where generator is frozen
        with only "unknown" words.

        Args:
            new_words: Dict with "nouns" and/or "verbs" lists to add
        """
        for category in ["nouns", "verbs"]:
            if category in new_words and new_words[category]:
                # Add new words
                self._vocabulary[category].extend(new_words[category])
                # Deduplicate while preserving order (important for indexing)
                seen = set()
                deduped = []
                for word in self._vocabulary[category]:
                    if word not in seen:
                        seen.add(word)
                        deduped.append(word)
                self._vocabulary[category] = deduped

    def generate(
        self,
        thought: torch.Tensor,
        style: StyleType = StyleType.NEUTRAL,
        max_tokens: int = MAX_GENERATION_TOKENS,
        structure: Optional[JazzTemplate] = None,
    ) -> GenerationResult:
        """
        Main generation entry point.

        Stage 1: Holographic Reasoning (Resonator + TargetEncoder)
        Stage 2: Constrained Generation (token-by-token verification)

        Args:
            thought: Composite thought vector to express
            style: StyleType for output modulation
            max_tokens: Maximum tokens to generate
            structure: Optional JazzTemplate for structural modulation

        Returns:
            GenerationResult with text, trace, and metrics
        """
        # Clear divergence history for fresh session
        self._divergence.clear_history()

        # Stage 1: Holographic Reasoning
        resonator_result = self._resonator.resonate(
            thought,
            self._vocabulary["nouns"],
            self._vocabulary["verbs"],
        )

        style_vector = self._sesame.get_style_vector(style)
        target = self._target_encoder.encode(resonator_result, style_vector)

        # Stage 2: Constrained Generation
        output_tokens: List[str] = []
        trace: List[GenerationTrace] = []
        metrics = GenerationMetrics(convergence_iterations=resonator_result.iterations)

        # Generate for each role in grammar
        for role in target.grammar:
            # Get vocabulary for this role
            if role == "VERB":
                role_vocab = self._vocabulary["verbs"]
            else:
                role_vocab = self._vocabulary["nouns"]

            # Generate candidates
            candidates = self._generate_candidates(
                target, role, role_vocab, style_vector, structure
            )

            # Evaluate and select best
            selected_token, selected_trace = self._evaluate_candidates(
                candidates, output_tokens, target, role, metrics, structure
            )

            # Check for disfluency injection
            if selected_trace.similarity < self._sesame._creativity:
                if self._sesame.should_inject_disfluency(selected_trace.similarity):
                    filler = self._sesame.select_filler(selected_trace.similarity)
                    filler_trace = GenerationTrace(
                        token=filler.value,
                        type="FILLER",
                        similarity=selected_trace.similarity,
                        action=DivergenceAction.ACCEPT,
                        role=role,
                        reason=f"Low confidence ({selected_trace.similarity:.2f})"
                    )
                    trace.append(filler_trace)
                    metrics.fillers_injected += 1

            # Add selected token
            output_tokens.append(selected_token)
            trace.append(selected_trace)

        # Calculate average similarity
        if trace:
            token_traces = [t for t in trace if t.type == "TOKEN"]
            if token_traces:
                metrics.average_similarity = sum(t.similarity for t in token_traces) / len(token_traces)

        # Construct final text
        text = " ".join(output_tokens)

        return GenerationResult(
            text=text,
            tokens=output_tokens,
            trace=trace,
            divergence_history=self._divergence.divergence_history,
            metrics=metrics,
            resonator_result=resonator_result,
        )
    
    def generate_with_validation(
        self,
        thought: Optional[torch.Tensor] = None,
        context: Optional[GenerationContext] = None,
        fact_answer: Optional[str] = None,
        expected_subject: Optional[str] = None,
        style: StyleType = StyleType.NEUTRAL,
        max_tokens: int = MAX_GENERATION_TOKENS,
        structure: Optional[JazzTemplate] = None,
    ) -> Optional[GenerationResult]:
        """
        Generate with output validation.
        
        Wrapper around generate() that validates output before returning.
        Returns None if validation fails (forcing fallback to templates).
        
        Supports both legacy signature (thought vector) and new signature (GenerationContext).
        
        Args:
            thought: Composite thought vector to express (legacy)
            context: GenerationContext with all information (new)
            fact_answer: Expected answer from fact store (for validation, legacy)
            expected_subject: Expected subject (e.g., "Australia") for validation (legacy)
            style: StyleType for output modulation (legacy)
            max_tokens: Maximum tokens to generate
            structure: Optional JazzTemplate for structural modulation
            
        Returns:
            GenerationResult if validation passes, None otherwise
        """
        # Support both legacy and new signatures
        if context is not None:
            # New signature: extract from context
            thought = context.thought_vector
            if thought is None:
                return None  # Can't generate without thought vector
            fact_answer = context.fact_answer
            expected_subject = context.expected_subject
            style = context.style
        elif thought is None:
            return None  # Need either context or thought
        
        result = self.generate(thought, style, max_tokens, structure)
        
        # Validate output with expected subject to catch garbage like "day is canberra"
        if not self.validate_output(result.text, fact_answer, expected_subject):
            return None
        
        return result

    def generate_from_words(
        self,
        subject: str,
        verb: str,
        obj: str,
        style: StyleType = StyleType.NEUTRAL,
    ) -> GenerationResult:
        """
        Generate from explicit S-V-O words.

        Useful for testing or when you have the desired output
        and want to verify generation matches.

        Args:
            subject: Subject word
            verb: Verb word
            obj: Object word
            style: StyleType for modulation

        Returns:
            GenerationResult
        """
        style_vector = self._sesame.get_style_vector(style)
        target = self._target_encoder.encode_from_words(
            subject, verb, obj, style_vector
        )

        # Simplified generation - just verify the words
        output_tokens = [subject, verb, obj]
        trace: List[GenerationTrace] = []

        for role, token in zip(["SUBJECT", "VERB", "OBJECT"], output_tokens):
            generated = self._re_encoder.encode_with_roles(subject, verb, obj)
            result = self._divergence.calculate(
                target.target_tensor, generated, role, target.confidence_map
            )
            trace.append(GenerationTrace(
                token=token,
                type="TOKEN",
                similarity=result.similarity,
                action=result.action,
                role=role,
            ))

        text = " ".join(output_tokens)
        metrics = GenerationMetrics(
            total_tokens=3,
            accepted_first_try=3,
            average_similarity=sum(t.similarity for t in trace) / 3,
        )

        return GenerationResult(
            text=text,
            tokens=output_tokens,
            trace=trace,
            divergence_history=self._divergence.divergence_history,
            metrics=metrics,
        )

    def _generate_candidates(
        self,
        target: TargetPackage,
        role: str,
        vocabulary: List[str],
        style_vector: torch.Tensor,
        structure: Optional[JazzTemplate] = None,
        k: int = CANDIDATE_K,
    ) -> List[tuple[str, float]]:
        """
        Generate candidate tokens for a role position.

        Uses modulated cleanup to get top-k candidates.

        Args:
            target: Target package with constraints
            role: Current role ("SUBJECT", "VERB", "OBJECT")
            vocabulary: Candidate words for this role
            style_vector: Style modulation vector
            k: Number of candidates to return

        Returns:
            List of (word, score) tuples sorted by score descending
        """
        # Get role-specific proposal from target
        codebook = self._re_encoder._codebook
        role_vec = codebook.get_role(role)
        proposal = self._re_encoder._ops.unbind(target.target_tensor, role_vec)

        # Get scores for all vocabulary words
        vocab_vectors = codebook.encode_batch(vocabulary)
        
        # If structure is provided, bind vocabulary vectors with structure
        # This ensures candidates match the structural template
        if structure is not None:
            struct_vec = structure.get_structure_vector(role)
            # Apply structure to each vocabulary vector
            structured_vocab = []
            for v in vocab_vectors:
                structured_vocab.append(structure.apply_structure(v, role))
            vocab_vectors = torch.stack(structured_vocab)
            # Also apply structure to proposal for fair comparison
            proposal = structure.apply_structure(proposal, role)
        
        semantic_scores = Similarity.cosine_batch(proposal, vocab_vectors)
        style_scores = Similarity.cosine_batch(style_vector, vocab_vectors)

        # Combined scores
        combined = semantic_scores + (0.2 * style_scores)

        # Get top-k
        top_k = torch.topk(combined, min(k, len(vocabulary)))
        candidates = [
            (vocabulary[idx], float(score))
            for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist())
        ]

        return candidates

    def _evaluate_candidates(
        self,
        candidates: List[tuple[str, float]],
        context: List[str],
        target: TargetPackage,
        role: str,
        metrics: GenerationMetrics,
        structure: Optional[JazzTemplate] = None,
    ) -> tuple[str, GenerationTrace]:
        """
        Evaluate candidates through divergence check.

        Args:
            candidates: List of (word, score) tuples
            context: Already generated tokens
            target: Target package
            role: Current role
            metrics: Metrics to update
            structure: Optional JazzTemplate (for applying structure to generated vectors)

        Returns:
            Tuple of (selected word, trace entry)
        """
        best_candidate = None
        best_result = None
        best_score = -float("inf")

        for word, candidate_score in candidates:
            # Re-encode with this candidate
            if role == "SUBJECT":
                generated = self._re_encoder.encode_with_roles(
                    word, target.grammar[1] if len(context) > 0 else "unknown",
                    target.grammar[2] if len(context) > 1 else "unknown"
                )
            elif role == "VERB":
                generated = self._re_encoder.encode_with_roles(
                    context[0] if context else "unknown", word, "unknown"
                )
            else:  # OBJECT
                generated = self._re_encoder.encode_with_roles(
                    context[0] if len(context) > 0 else "unknown",
                    context[1] if len(context) > 1 else "unknown",
                    word
                )
            
            # Apply structure to generated vector if provided
            # Note: Target is already structured via _apply_structure_cleanly
            if structure is not None:
                # Build structured component cleanly
                # Component = (Token * Structure) * Role
                word_vec = self._re_encoder.encode_token(word)
                role_vec = self._re_encoder._codebook.get_role(role)
                struct_vec = structure.get_structure_vector(role)
                
                # Standard component (what encode_with_roles produced): (Token * Role)
                standard_component = self._re_encoder._ops.bind(word_vec, role_vec)
                
                # Structured component: ((Token * Structure) * Role)
                structured_token = self._re_encoder._ops.bind(word_vec, struct_vec)
                structured_component = self._re_encoder._ops.bind(structured_token, role_vec)
                
                # Swap components: Generated - Standard + Structured
                generated = generated - standard_component + structured_component

            # Check divergence
            result = self._divergence.calculate(
                target.target_tensor, generated, role, target.confidence_map
            )

            # Track metrics
            metrics.total_tokens += 1

            if result.action == DivergenceAction.REJECT:
                metrics.rejected_tokens += 1
                continue  # Try next candidate

            # Score surviving candidates
            combined_score = result.similarity * candidate_score
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = word
                best_result = result

                if result.action == DivergenceAction.ACCEPT:
                    metrics.accepted_first_try += 1
                else:
                    metrics.accepted_with_correction += 1

        # Fallback if all rejected
        if best_candidate is None:
            metrics.fallback_count += 1
            best_candidate = candidates[0][0]  # Use highest scoring candidate
            generated = self._re_encoder.encode_token(best_candidate)
            best_result = self._divergence.calculate(
                target.target_tensor, generated, role, target.confidence_map
            )

        trace_entry = GenerationTrace(
            token=best_candidate,
            type="TOKEN",
            similarity=best_result.similarity,
            action=best_result.action,
            role=role,
        )

        return best_candidate, trace_entry

    def validate_output(
        self, 
        text: str, 
        fact_answer: Optional[str] = None,
        expected_subject: Optional[str] = None
    ) -> bool:
        """
        Validate generated output for safety and correctness.
        
        Checks:
        1. Subject Coherence: First word should be a proper noun or match expected subject
        2. Fact Consistency: If fact_answer provided, it MUST appear in text
        3. Garbage Detection: Detect repetitive patterns and nonsense
        
        Args:
            text: Generated text to validate
            fact_answer: Expected answer from fact store (if applicable)
            expected_subject: Expected subject word (if applicable)
            
        Returns:
            True if validation passes, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check 0: Subject Coherence (CRITICAL - prevents garbage like "day is canberra")
        if len(words) >= 3:  # At least "subject verb object"
            first_word = words[0]
            
            # Known garbage subjects that should NEVER start a response
            garbage_subjects = {
                "day", "time", "year", "way", "thing", "fact", "system", "process",
                "method", "world", "used", "located", "software", "computer", "program",
                "language", "data", "code", "person", "people", "human", "scientist",
                "inventor", "creator", "founder", "author", "artist", "ocean", "mountain",
                "light", "energy", "matter", "atom", "molecule", "algorithm",
            }
            
            if first_word in garbage_subjects:
                return False
            
            # If expected_subject provided, check it matches
            if expected_subject:
                expected_lower = expected_subject.lower()
                # Allow partial match (e.g., "france" matches "France")
                if first_word != expected_lower and expected_lower not in text_lower:
                    return False
        
        # Check 1: Fact Consistency
        if fact_answer:
            fact_lower = fact_answer.lower()
            # Check if fact answer appears in text (allowing for word boundaries)
            # Split fact_answer into words and check if all key words appear
            fact_words = [w for w in fact_lower.split() if len(w) > 2]  # Skip short words
            if fact_words:
                # At least 50% of key words must appear
                matches = sum(1 for word in fact_words if word in text_lower)
                if matches < len(fact_words) * 0.5:
                    return False
            else:
                # Single word or short answer - must appear exactly
                if fact_lower not in text_lower:
                    return False
        
        # Check 2: Garbage Detection
        # Pattern 1: Immediate word repetition (e.g., "star star", "are are")
        if self._REPETITION_PATTERN.search(text):
            return False
        
        # Pattern 2: Known garbage patterns (using pre-compiled patterns)
        for pattern in self._GARBAGE_PATTERNS:
            if pattern.search(text):
                return False
        
        # Pattern 3: Check for repeated words (more than 2 occurrences of same word)
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > 2:
                    return False
        
        return True

    def __repr__(self) -> str:
        return (
            f"ResonantGenerator(vocab_nouns={len(self._vocabulary.get('nouns', []))}, "
            f"vocab_verbs={len(self._vocabulary.get('verbs', []))})"
        )

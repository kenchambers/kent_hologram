"""
SearchVerifier: Verify transformation candidates against training pairs.

This module implements the "verification" stage of search+verify ARC solving.
Candidates are only accepted if they produce correct outputs for ALL training pairs.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from hologram.arc.types import Grid, ARCTask, TrainingPair
from hologram.arc.transform_resonator import TransformResult
from hologram.arc.executor import TransformationExecutor
from hologram.arc.detector import ObjectDetector


@dataclass
class VerificationResult:
    """Result of verifying a transformation candidate."""
    passed: bool
    score: float  # Fraction of training pairs that matched (0.0-1.0)
    matched_pairs: int
    total_pairs: int
    transform: TransformResult
    pair_breakdown: List[bool] = None  # Which pairs matched [True, True, False]

    def __post_init__(self):
        """Initialize pair_breakdown if not provided."""
        if self.pair_breakdown is None:
            self.pair_breakdown = []


@dataclass
class VerificationStats:
    """
    Statistics from verifying multiple candidates.

    Used by the metacognitive loop to adapt search parameters
    based on partial success signals.

    Attributes:
        verified_transform: First transform that passed all training pairs
        best_partial_score: Best score from any candidate (0.0-1.0)
        best_partial_transform: The "close but not quite" candidate
        best_pair_breakdown: Which pairs the best partial matched [True, True, False]
        candidates_tested: Total number of candidates evaluated
        total_passed: Number of candidates that passed all pairs
    """
    verified_transform: Optional[TransformResult]  # First that passed all pairs
    best_partial_score: float  # Best score from any candidate (0.0-1.0)
    best_partial_transform: Optional[TransformResult] = None  # The "close" one
    best_pair_breakdown: List[bool] = None  # Which pairs the best partial matched
    candidates_tested: int = 0
    total_passed: int = 0  # Number of candidates that passed all pairs

    def __post_init__(self):
        """Initialize best_pair_breakdown if not provided."""
        if self.best_pair_breakdown is None:
            self.best_pair_breakdown = []

    def diagnostic_message(self) -> str:
        """Generate a diagnostic message for debugging."""
        if self.verified_transform:
            return f"Verified: {self.verified_transform.action}({self.verified_transform.target}, {self.verified_transform.modifier})"
        elif self.best_partial_transform:
            matched = sum(self.best_pair_breakdown)
            total = len(self.best_pair_breakdown)
            return (
                f"Best partial: {self.best_partial_score:.0%} ({matched}/{total} pairs) - "
                f"{self.best_partial_transform.action}({self.best_partial_transform.target}, "
                f"{self.best_partial_transform.modifier})"
            )
        else:
            return f"No candidates found (tested {self.candidates_tested})"


class SearchVerifier:
    """
    Verify transformation candidates against training pairs.

    Only accepts candidates that produce correct outputs for ALL training pairs.
    This preserves the "no hallucination" guarantee by ensuring we only
    return transformations that are verifiably correct.

    Attributes:
        executor: TransformationExecutor for applying transforms
        detector: ObjectDetector for finding objects in grids
    """

    def __init__(
        self,
        executor: Optional[TransformationExecutor] = None,
        detector: Optional[ObjectDetector] = None,
    ):
        """
        Initialize verifier.

        Args:
            executor: TransformationExecutor (creates new if None)
            detector: ObjectDetector (creates new if None)
        """
        self._executor = executor or TransformationExecutor()
        self._detector = detector or ObjectDetector()

    def verify_transform(
        self,
        transform: TransformResult,
        training_pairs: List[TrainingPair],
    ) -> VerificationResult:
        """
        Verify a transformation candidate against training pairs.

        Applies the transform to each training input and checks if output
        matches the training output exactly.

        Args:
            transform: Transformation candidate to verify
            training_pairs: List of training input-output pairs

        Returns:
            VerificationResult indicating if transform passed all pairs,
            including pair_breakdown showing which pairs matched
        """
        if not training_pairs:
            return VerificationResult(
                passed=False,
                score=0.0,
                matched_pairs=0,
                total_pairs=0,
                transform=transform,
                pair_breakdown=[],
            )

        matched = 0
        pair_breakdown: List[bool] = []

        for pair in training_pairs:
            try:
                # Detect objects in input
                objects = self._detector.detect(pair.input)

                # Apply transformation
                output = self._executor.execute(
                    action=transform.action,
                    target=transform.target,
                    modifier=transform.modifier,
                    objects=objects,
                    grid=pair.input,
                )

                # Check exact match
                if output == pair.output:
                    matched += 1
                    pair_breakdown.append(True)
                else:
                    pair_breakdown.append(False)
            except Exception:
                # Any execution error = failure
                pair_breakdown.append(False)

        score = matched / len(training_pairs)
        passed = matched == len(training_pairs)

        return VerificationResult(
            passed=passed,
            score=score,
            matched_pairs=matched,
            total_pairs=len(training_pairs),
            transform=transform,
            pair_breakdown=pair_breakdown,
        )

    def verify_candidates(
        self,
        candidates: List[TransformResult],
        training_pairs: List[TrainingPair],
    ) -> Optional[TransformResult]:
        """
        Verify a list of candidates and return the first that passes all pairs.

        Args:
            candidates: List of transformation candidates (should be pre-sorted)
            training_pairs: List of training pairs to verify against

        Returns:
            First TransformResult that passes all pairs, or None if none pass
        """
        for candidate in candidates:
            result = self.verify_transform(candidate, training_pairs)
            if result.passed:
                return candidate

        return None

    def verify_candidates_with_stats(
        self,
        candidates: List[TransformResult],
        training_pairs: List[TrainingPair],
    ) -> VerificationStats:
        """
        Verify candidates and return detailed statistics.

        Unlike verify_candidates(), this method returns partial success
        information even when no candidate passes all pairs. This enables
        the metacognitive loop to adapt search parameters.

        Args:
            candidates: List of transformation candidates
            training_pairs: List of training pairs to verify against

        Returns:
            VerificationStats with best partial score, verified transform,
            and diagnostic info about the "close but not quite" candidate
        """
        best_partial_score = 0.0
        best_partial_transform: Optional[TransformResult] = None
        best_pair_breakdown: List[bool] = []
        verified_transform = None
        total_passed = 0

        for candidate in candidates:
            result = self.verify_transform(candidate, training_pairs)

            # Track best partial score and the transform that achieved it
            if result.score > best_partial_score:
                best_partial_score = result.score
                best_partial_transform = candidate
                best_pair_breakdown = result.pair_breakdown

            # Track first fully verified transform
            if result.passed:
                total_passed += 1
                if verified_transform is None:
                    verified_transform = candidate

        return VerificationStats(
            verified_transform=verified_transform,
            best_partial_score=best_partial_score,
            best_partial_transform=best_partial_transform,
            best_pair_breakdown=best_pair_breakdown,
            candidates_tested=len(candidates),
            total_passed=total_passed,
        )

    def verify_sequence(
        self,
        sequence: List[TransformResult],
        training_pairs: List[TrainingPair],
    ) -> VerificationResult:
        """
        Verify a sequence of transformations (multi-step).

        Applies transformations in order to each training input and checks
        if final output matches training output.

        Args:
            sequence: List of transformations to apply in order
            training_pairs: List of training pairs to verify against

        Returns:
            VerificationResult indicating if sequence passed all pairs
        """
        if not training_pairs:
            return VerificationResult(
                passed=False,
                score=0.0,
                matched_pairs=0,
                total_pairs=0,
                transform=sequence[0] if sequence else None,
            )

        matched = 0
        for pair in training_pairs:
            try:
                current_grid = pair.input

                # Apply each transformation in sequence
                for transform in sequence:
                    objects = self._detector.detect(current_grid)
                    current_grid = self._executor.execute(
                        action=transform.action,
                        target=transform.target,
                        modifier=transform.modifier,
                        objects=objects,
                        grid=current_grid,
                    )

                # Check if final output matches
                if current_grid == pair.output:
                    matched += 1
            except Exception:
                # Any execution error = failure
                pass

        score = matched / len(training_pairs)
        passed = matched == len(training_pairs)

        return VerificationResult(
            passed=passed,
            score=score,
            matched_pairs=matched,
            total_pairs=len(training_pairs),
            transform=sequence[0] if sequence else None,
        )

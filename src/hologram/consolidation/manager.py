"""
ConsolidationManager: Async background training for neural memory consolidation.

Implements sleep-inspired memory consolidation where:
- Working memory (HDC) handles fast recent facts
- Long-term memory (Neural) consolidates via background training
- Winner-take-all selection with calibrated confidence

Key Design Decisions:
- Non-blocking: Training happens in background thread
- Double-buffering: Snapshot state before training, don't block stores
- Decay instead of delete: HDC memory fades but isn't wiped
- Unbinding validation gate: Fall back to HDC if neural uncertain
"""

from __future__ import annotations

import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

import torch

from hologram.consolidation.calibration import ConfidenceCalibrator, CalibrationResult
from hologram.consolidation.neural_memory import NeuralMemory, ConsolidationFact
from hologram.core.vector_space import VectorSpace
from hologram.memory.memory_trace import MemoryTrace
from hologram.core.similarity import Similarity


logger = logging.getLogger(__name__)


@dataclass
class PendingFact:
    """A fact pending consolidation."""
    key_vector: torch.Tensor
    value_vector: torch.Tensor
    value_label: str
    timestamp: float


@dataclass
class ConsolidationSnapshot:
    """Snapshot of state for background consolidation."""
    facts: List[PendingFact]
    old_trace: torch.Tensor
    triggered_at: float


@dataclass
class QueryResult:
    """Result of a consolidated memory query."""
    value_label: str
    confidence: float
    source: str  # "hdc" or "neural"
    hdc_confidence: float
    neural_confidence: float
    calibration: Optional[CalibrationResult] = None


class ConsolidationManager:
    """
    Manages hybrid HDC + Neural memory with async consolidation.

    The manager maintains:
    - Working memory (MemoryTrace): Fast HDC for recent facts
    - Long-term memory (NeuralMemory): Consolidated neural network
    - Pending facts queue: Facts waiting for consolidation

    Consolidation is triggered automatically when pending facts exceed
    threshold, and runs in a background thread to avoid blocking.

    Args:
        space: VectorSpace configuration
        consolidation_threshold: Number of pending facts to trigger consolidation
        decay_factor: How much HDC signal to preserve after consolidation (0-1)
        neural_hidden_dim: Hidden dimension for neural network
        initial_vocab_size: Initial neural vocabulary capacity
    """

    def __init__(
        self,
        space: VectorSpace,
        consolidation_threshold: int = 20,
        decay_factor: float = 0.3,
        neural_hidden_dim: int = 256,
        initial_vocab_size: int = 1000,
    ):
        self._space = space
        self._consolidation_threshold = consolidation_threshold
        self._decay_factor = decay_factor

        # Working memory (HDC)
        self._working_memory = MemoryTrace(space)

        # Long-term memory (Neural)
        self._neural_memory = NeuralMemory(
            input_dim=space.dimensions,
            hidden_dim=neural_hidden_dim,
            initial_vocab_size=initial_vocab_size,
        )

        # Confidence calibrator
        self._calibrator = ConfidenceCalibrator()

        # Pending facts for consolidation
        self._pending_facts: List[PendingFact] = []

        # Thread synchronization
        self._lock = threading.RLock()
        self._consolidation_queue: queue.Queue[ConsolidationSnapshot] = queue.Queue()
        self._worker_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Vocabulary for HDC resonance cleanup
        self._value_vocab: dict[str, torch.Tensor] = {}

        # Statistics
        self._total_consolidated = 0
        self._consolidation_count = 0

        # Callbacks
        self._on_consolidation_complete: Optional[Callable[[int], None]] = None

    def start_worker(self) -> None:
        """Start the background consolidation worker thread."""
        with self._lock:
            if self._worker_running:
                return

            self._worker_running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="ConsolidationWorker",
            )
            self._worker_thread.start()
            logger.debug("Consolidation worker started")

    def stop_worker(self, timeout: float = 5.0) -> None:
        """Stop the background worker thread."""
        with self._lock:
            if not self._worker_running:
                return
            self._worker_running = False

        # Signal worker to stop
        self._consolidation_queue.put(None)  # Poison pill

        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

        logger.debug("Consolidation worker stopped")

    def _worker_loop(self) -> None:
        """Background worker loop for consolidation."""
        while self._worker_running:
            try:
                # Wait for consolidation task (with timeout to check running flag)
                try:
                    snapshot = self._consolidation_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Poison pill - exit
                if snapshot is None:
                    break

                # Perform consolidation
                self._do_consolidation(snapshot)

            except Exception as e:
                logger.error(f"Consolidation worker error: {e}", exc_info=True)

    def _do_consolidation(self, snapshot: ConsolidationSnapshot) -> None:
        """Perform actual consolidation (called from worker thread)."""
        start_time = time.time()
        n_facts = len(snapshot.facts)

        if n_facts == 0:
            return

        # Convert pending facts to consolidation facts
        consolidation_facts = []
        for pf in snapshot.facts:
            # Get or create value index
            cf = ConsolidationFact(
                key_vector=pf.key_vector.clone(),
                value_index=-1,  # Will be assigned by neural memory
                value_label=pf.value_label,
                timestamp=pf.timestamp,
            )
            consolidation_facts.append(cf)

        # Train neural memory
        loss = self._neural_memory.consolidate(
            consolidation_facts,
            epochs=50,  # More epochs for better learning
            batch_size=32,
            replay_ratio=0.3,
        )

        # Update statistics
        with self._lock:
            self._total_consolidated += n_facts
            self._consolidation_count += 1

            # Apply decay to working memory (preserves faint signal)
            self._apply_decay_to_working_memory(snapshot.old_trace)

        elapsed = time.time() - start_time
        # Log to root logger so it shows up in console
        logging.getLogger().info(
            f"  [Neural Consolidation] Processed {n_facts} facts in {elapsed:.2f}s (loss={loss:.4f})"
        )
        logger.debug(
            f"Consolidation complete: {n_facts} facts, "
            f"loss={loss:.4f}, time={elapsed:.2f}s"
        )

        # Callback
        if self._on_consolidation_complete:
            try:
                self._on_consolidation_complete(n_facts)
            except Exception as e:
                logger.error(f"Consolidation callback error: {e}")

    def _apply_decay_to_working_memory(self, old_trace: torch.Tensor) -> None:
        """
        Apply decay to working memory (preserves faint signal).
        
        Handles race condition where new facts may have been stored
        while consolidation was running.
        """
        with self._lock:
            current_trace = self._working_memory._trace
            
            # If nothing changed, just decay the old trace
            if torch.allclose(current_trace, old_trace):
                decayed_trace = old_trace * self._decay_factor
                self._working_memory._trace = decayed_trace
            else:
                # Facts were added during consolidation!
                # We want: (old_trace * decay) + (new_facts)
                # We approximate new_facts as (current_trace - old_trace)
                # So: current_trace - old_trace + (old_trace * decay)
                #   = current_trace - (old_trace * (1 - decay))
                
                decay_amount = old_trace * (1.0 - self._decay_factor)
                self._working_memory._trace = current_trace - decay_amount
                
        # Note: fact_count is not reset - it's a running total

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        value_label: str,
    ) -> float:
        """
        Store a key-value pair in working memory.

        Facts are stored in HDC immediately (non-blocking) and queued
        for neural consolidation. Returns latency in milliseconds.

        Args:
            key: Key vector (typically bind(subject, predicate))
            value: Value vector
            value_label: String label for the value

        Returns:
            Store latency in milliseconds
        """
        start_time = time.perf_counter()

        with self._lock:
            # Store in HDC working memory (fast)
            self._working_memory.store(key, value)

            # Add to vocabulary
            self._value_vocab[value_label] = value

            # Queue for consolidation
            self._pending_facts.append(PendingFact(
                key_vector=key.clone(),
                value_vector=value.clone(),
                value_label=value_label,
                timestamp=time.time(),
            ))

            # Check if consolidation should be triggered
            if len(self._pending_facts) >= self._consolidation_threshold:
                self._schedule_consolidation()

        latency_ms = (time.perf_counter() - start_time) * 1000
        return latency_ms

    def _schedule_consolidation(self) -> None:
        """Schedule background consolidation (must hold lock)."""
        if not self._pending_facts:
            return

        # Create snapshot
        snapshot = ConsolidationSnapshot(
            facts=self._pending_facts.copy(),
            old_trace=self._working_memory._trace.clone(),
            triggered_at=time.time(),
        )

        # Clear pending (will start fresh after decay)
        self._pending_facts = []

        # Queue for background processing
        self._consolidation_queue.put(snapshot)

        logger.debug(f"Scheduled consolidation of {len(snapshot.facts)} facts")

    def force_consolidation(self) -> None:
        """Force immediate consolidation of pending facts."""
        with self._lock:
            if self._pending_facts:
                self._schedule_consolidation()

    def query(
        self,
        key: torch.Tensor,
        require_unbind_safety: bool = False,
    ) -> QueryResult:
        """
        Query consolidated memory with winner-take-all selection.

        Queries both HDC and Neural memory, calibrates confidence,
        and returns the winner. If require_unbind_safety is True,
        falls back to HDC if neural is below unbinding threshold.

        Args:
            key: Query key vector
            require_unbind_safety: Require neural to pass unbinding gate

        Returns:
            QueryResult with value, confidence, and source
        """
        with self._lock:
            # Query HDC
            hdc_label, hdc_conf = self._query_hdc(key)

            # Query Neural
            neural_label, neural_conf = self._neural_memory.query(key)

            # Handle case where neural has no data
            if neural_label is None:
                return QueryResult(
                    value_label=hdc_label or "",
                    confidence=self._calibrator.calibrate_hdc(hdc_conf),
                    source="hdc",
                    hdc_confidence=hdc_conf,
                    neural_confidence=0.0,
                )

            # Pick winner with calibration
            calibration = self._calibrator.pick_winner(
                hdc_conf,
                neural_conf,
                require_unbind_safety=require_unbind_safety,
            )

            # Select result based on winner
            if calibration.winner == "hdc":
                result_label = hdc_label or ""
                result_conf = calibration.hdc_calibrated
            else:
                result_label = neural_label
                result_conf = calibration.neural_calibrated

            return QueryResult(
                value_label=result_label,
                confidence=result_conf,
                source=calibration.winner,
                hdc_confidence=hdc_conf,
                neural_confidence=neural_conf,
                calibration=calibration,
            )

    def _query_hdc(self, key: torch.Tensor) -> Tuple[Optional[str], float]:
        """Query HDC working memory with resonance cleanup."""
        if not self._value_vocab:
            return None, 0.0

        # Get candidate vectors
        labels = list(self._value_vocab.keys())
        candidates = torch.stack([self._value_vocab[l] for l in labels])

        # Resonance: unbind then find best match
        similarities = self._working_memory.resonance(key, candidates)
        best_idx = torch.argmax(similarities).item()
        confidence = float(similarities[best_idx].item())

        return labels[best_idx], confidence

    def query_hdc_only(self, key: torch.Tensor) -> Tuple[Optional[str], float]:
        """Query only HDC memory (for testing/debugging)."""
        with self._lock:
            return self._query_hdc(key)

    def query_neural_only(self, key: torch.Tensor) -> Tuple[Optional[str], float]:
        """Query only neural memory (for testing/debugging)."""
        with self._lock:
            return self._neural_memory.query(key)

    @property
    def pending_count(self) -> int:
        """Number of facts pending consolidation."""
        with self._lock:
            return len(self._pending_facts)

    @property
    def total_consolidated(self) -> int:
        """Total number of facts consolidated."""
        return self._total_consolidated

    @property
    def consolidation_count(self) -> int:
        """Number of consolidation cycles completed."""
        return self._consolidation_count

    @property
    def vocab_size(self) -> int:
        """Size of value vocabulary."""
        with self._lock:
            return len(self._value_vocab)

    @property
    def neural_vocab_size(self) -> int:
        """Size of neural vocabulary."""
        return self._neural_memory.vocab_size

    @property
    def hdc_saturation(self) -> float:
        """HDC memory saturation estimate."""
        return self._working_memory.saturation_estimate

    def set_consolidation_callback(
        self,
        callback: Optional[Callable[[int], None]],
    ) -> None:
        """Set callback for when consolidation completes."""
        self._on_consolidation_complete = callback

    def get_statistics(self) -> dict:
        """Get consolidation statistics."""
        with self._lock:
            return {
                "pending_facts": len(self._pending_facts),
                "total_consolidated": self._total_consolidated,
                "consolidation_count": self._consolidation_count,
                "vocab_size": len(self._value_vocab),
                "neural_vocab_size": self._neural_memory.vocab_size,
                "neural_trained_samples": self._neural_memory.trained_samples,
                "neural_replay_buffer": self._neural_memory.replay_buffer_size,
                "hdc_saturation": self._working_memory.saturation_estimate,
                "hdc_fact_count": self._working_memory.fact_count,
                "worker_running": self._worker_running,
            }

    def state_dict(self) -> dict:
        """Get state for persistence."""
        with self._lock:
            # Serialize pending facts (not yet consolidated)
            pending_data = [
                {
                    "key_vector": pf.key_vector.clone(),
                    "value_vector": pf.value_vector.clone(),
                    "value_label": pf.value_label,
                    "timestamp": pf.timestamp,
                }
                for pf in self._pending_facts
            ]
            
            return {
                "neural_memory": self._neural_memory.state_dict(),
                "working_memory_trace": self._working_memory._trace.clone(),
                "working_memory_fact_count": self._working_memory._fact_count,
                "value_vocab": {k: v.clone() for k, v in self._value_vocab.items()},
                "pending_facts": pending_data,
                "total_consolidated": self._total_consolidated,
                "consolidation_count": self._consolidation_count,
            }

    def load_state_dict(self, state: dict) -> None:
        """Load state from persistence."""
        with self._lock:
            self._neural_memory.load_state_dict(state["neural_memory"])
            self._working_memory._trace = state["working_memory_trace"]
            self._working_memory._fact_count = state["working_memory_fact_count"]
            self._value_vocab = state["value_vocab"]
            self._total_consolidated = state.get("total_consolidated", 0)
            self._consolidation_count = state.get("consolidation_count", 0)
            
            # Restore pending facts
            pending_data = state.get("pending_facts", [])
            self._pending_facts = [
                PendingFact(
                    key_vector=pf["key_vector"],
                    value_vector=pf["value_vector"],
                    value_label=pf["value_label"],
                    timestamp=pf["timestamp"],
                )
                for pf in pending_data
            ]

    def __enter__(self) -> "ConsolidationManager":
        """Context manager entry - start worker."""
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop worker."""
        self.stop_worker()

    def __repr__(self) -> str:
        return (
            f"ConsolidationManager("
            f"pending={self.pending_count}, "
            f"consolidated={self.total_consolidated}, "
            f"vocab={self.vocab_size})"
        )

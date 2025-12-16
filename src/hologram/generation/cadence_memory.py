"""
CadenceMemory: Neural memory for sentence structure patterns.

Uses the same NeuralMemory architecture but trained on:
- Key: context_vector (query context)
- Value: cadence_pattern_id (structure template)

This learns which structures work best for which contexts.
"""

from typing import Dict, Optional

import torch

from hologram.consolidation.neural_memory import NeuralMemory, ConsolidationFact
from hologram.generation.cadence_extractor import CadencePattern


class CadenceMemory:
    """
    Neural memory specialized for cadence patterns.

    Separate from fact memory to maintain 0% hallucination guarantee:
    - Fact memory: context → fact_value (grounded)
    - Cadence memory: context → structure_pattern (learned style)
    """

    def __init__(self, dimensions: int, hidden_dim: int = 256):
        """
        Initialize cadence memory.

        Args:
            dimensions: HDC vector dimension
            hidden_dim: Neural network hidden dimension
        """
        self._neural = NeuralMemory(
            input_dim=dimensions,
            hidden_dim=hidden_dim,
            initial_vocab_size=500,  # Cadence patterns are fewer than facts
        )

        # Store pattern templates
        self._patterns: Dict[str, CadencePattern] = {}
        self._pattern_counter = 0

    def store_cadence(
        self, context_vector: torch.Tensor, pattern: CadencePattern
    ) -> None:
        """
        Store a cadence pattern for a context.

        Args:
            context_vector: Query context (what the user asked)
            pattern: Extracted cadence pattern (how to respond)
        """
        pattern_id = self._get_or_create_pattern_id(pattern)

        # Store for neural training
        fact = ConsolidationFact(
            key_vector=context_vector,
            value_index=-1,  # Not used for cadence (we use labels)
            value_label=pattern_id,
        )
        self._neural.add_to_replay([fact])

    def _get_or_create_pattern_id(self, pattern: CadencePattern) -> str:
        """
        Get or create a pattern ID for a cadence pattern.

        Uses template as the key to deduplicate similar patterns.
        """
        # Use template as key (normalized)
        template_key = pattern.template.lower().strip()
        
        # Check if we've seen this template before
        for pattern_id, stored_pattern in self._patterns.items():
            if stored_pattern.template.lower().strip() == template_key:
                return pattern_id
        
        # Create new pattern ID
        pattern_id = f"cadence_pattern_{self._pattern_counter}"
        self._pattern_counter += 1
        self._patterns[pattern_id] = pattern
        
        return pattern_id

    def query_cadence(
        self, context_vector: torch.Tensor
    ) -> Optional[CadencePattern]:
        """
        Query for best cadence pattern given context.

        Returns learned structure template or None if below threshold.

        Args:
            context_vector: Query context vector

        Returns:
            CadencePattern if found with sufficient confidence, None otherwise
        """
        pattern_id, confidence = self._neural.query(context_vector)

        # Ensure confidence is a scalar (not a tensor)
        if isinstance(confidence, torch.Tensor):
            confidence = float(confidence.item()) if confidence.numel() == 1 else float(confidence.mean().item())

        if confidence < 0.3:  # Below threshold
            return None

        return self._patterns.get(pattern_id)

    def consolidate(self, epochs: int = 50) -> float:
        """
        Run neural consolidation on accumulated patterns.

        Args:
            epochs: Number of training epochs

        Returns:
            Final training loss
        """
        # Get all pending facts from replay buffer
        # We need to access the replay buffer, but it's private
        # So we'll consolidate with an empty list and let it use replay
        # Actually, we need to extract facts from replay buffer first
        
        # Access replay buffer via the internal attribute
        facts = list(self._neural._replay_buffer)
        if not facts:
            return 0.0
        
        return self._neural.consolidate(facts, epochs=epochs)

    @property
    def pattern_count(self) -> int:
        """Number of stored cadence patterns."""
        return len(self._patterns)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (number of learned patterns)."""
        return self._neural.vocab_size

    def get_state_dict(self) -> Dict:
        """
        Get state dict for persistence.

        Returns:
            Dictionary containing all state needed to restore cadence memory
        """
        # Serialize patterns (CadencePattern to dict)
        serialized_patterns = {}
        for pattern_id, pattern in self._patterns.items():
            serialized_patterns[pattern_id] = {
                "template": pattern.template,
                "structure_vector": pattern.structure_vector.cpu(),
                "slot_positions": pattern.slot_positions,
                "original_text": pattern.original_text,
            }

        # Get neural state if method exists
        neural_state = {}
        if hasattr(self._neural, "get_state_dict"):
            neural_state = self._neural.get_state_dict()
        elif hasattr(self._neural, "_model"):
            # Fallback: serialize model state directly
            neural_state = {"model_state": self._neural._model.state_dict()}

        return {
            "neural_state": neural_state,
            "patterns": serialized_patterns,
            "pattern_counter": self._pattern_counter,
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Load state from dict (restore from persistence).

        Args:
            state: State dict from get_state_dict()
        """
        from hologram.generation.cadence_extractor import CadencePattern

        # Load neural state
        if "neural_state" in state and state["neural_state"]:
            neural_state = state["neural_state"]
            if hasattr(self._neural, "load_state_dict"):
                self._neural.load_state_dict(neural_state)
            elif "model_state" in neural_state and hasattr(self._neural, "_model"):
                self._neural._model.load_state_dict(neural_state["model_state"])

        # Load patterns
        if "patterns" in state:
            self._patterns = {}
            for pattern_id, pattern_data in state["patterns"].items():
                self._patterns[pattern_id] = CadencePattern(
                    template=pattern_data["template"],
                    structure_vector=pattern_data["structure_vector"],
                    slot_positions=pattern_data["slot_positions"],
                    original_text=pattern_data["original_text"],
                )

        # Load counter
        if "pattern_counter" in state:
            self._pattern_counter = state["pattern_counter"]

    def get_replay_buffer_size(self) -> int:
        """Get size of replay buffer (pending patterns for consolidation)."""
        return len(self._neural._replay_buffer)


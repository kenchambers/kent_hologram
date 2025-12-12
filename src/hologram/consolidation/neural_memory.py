"""
NeuralMemory: Classification-head neural network for O(1) memory lookup.

Instead of reconstructing vectors (which requires O(vocab_size) cleanup),
this module predicts item indices directly via a classification head.

Key Design Decisions:
- Classification head eliminates O(C) candidate search
- GELU activation for smoother gradients
- Experience replay for continual learning (no EWC needed)
- Thread-safe training with proper locking
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConsolidationFact:
    """A fact prepared for neural consolidation."""
    key_vector: torch.Tensor
    value_index: int
    value_label: str
    timestamp: float = field(default_factory=lambda: 0.0)


class NeuralMemoryNetwork(nn.Module):
    """
    Neural network with classification head for O(1) memory lookup.

    Architecture:
    - Two-layer encoder with GELU activation
    - Classification head predicting item index directly
    - No reconstruction step needed (eliminates O(C) search)

    Args:
        input_dim: Dimension of input key vectors (HDC dimension)
        hidden_dim: Hidden layer dimension
        output_dim: Number of possible output classes (vocab size)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder: key vector -> hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Classification head: hidden -> vocab index
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: key vector -> logits over vocabulary.

        Args:
            key: Input key vector(s) of shape (batch, input_dim) or (input_dim,)

        Returns:
            Logits of shape (batch, output_dim) or (output_dim,)
        """
        # Handle single vector input
        squeeze_output = False
        if key.dim() == 1:
            key = key.unsqueeze(0)
            squeeze_output = True

        hidden = self.encoder(key)
        logits = self.classifier(hidden)

        if squeeze_output:
            logits = logits.squeeze(0)

        return logits

    def expand_vocab(self, new_size: int) -> None:
        """
        Expand vocabulary size (add new output classes).

        Preserves existing weights while adding new randomly-initialized
        weights for new classes.

        Args:
            new_size: New vocabulary size (must be >= current size)
        """
        if new_size <= self.output_dim:
            return

        old_classifier = self.classifier
        self.classifier = nn.Linear(self.hidden_dim, new_size)

        # Copy existing weights
        with torch.no_grad():
            self.classifier.weight[:self.output_dim] = old_classifier.weight
            self.classifier.bias[:self.output_dim] = old_classifier.bias
            # Initialize new weights
            nn.init.xavier_uniform_(
                self.classifier.weight[self.output_dim:], gain=0.1
            )
            nn.init.zeros_(self.classifier.bias[self.output_dim:])

        self.output_dim = new_size


class NeuralMemory:
    """
    High-level interface for neural memory with O(1) query.

    Wraps NeuralMemoryNetwork with:
    - Thread-safe training
    - Experience replay buffer
    - Vocabulary management
    - Query interface with confidence scores

    Args:
        input_dim: HDC vector dimension
        hidden_dim: Neural network hidden dimension
        initial_vocab_size: Initial vocabulary capacity
        replay_buffer_size: Size of experience replay buffer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        initial_vocab_size: int = 1000,
        replay_buffer_size: int = 5000,
    ):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._lock = threading.RLock()

        # Neural network
        self._network = NeuralMemoryNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=initial_vocab_size,
        )

        # Vocabulary mapping: index <-> label
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        self._next_index = 0

        # Experience replay buffer (FIFO with max size)
        self._replay_buffer: deque[ConsolidationFact] = deque(maxlen=replay_buffer_size)

        # Training state
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._trained_samples = 0

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        """Lazily create optimizer."""
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._network.parameters(),
                lr=5e-3,  # Higher LR for faster convergence on small datasets
                weight_decay=1e-5,
            )
        return self._optimizer

    def _get_or_create_index(self, label: str) -> int:
        """Get index for label, creating new one if needed."""
        if label in self._label_to_index:
            return self._label_to_index[label]

        # Create new index
        idx = self._next_index
        self._next_index += 1
        self._label_to_index[label] = idx
        self._index_to_label[idx] = label

        # Expand network if needed
        if idx >= self._network.output_dim:
            new_size = max(idx + 1, int(self._network.output_dim * 1.5))
            self._network.expand_vocab(new_size)

        return idx

    def add_to_replay(self, facts: List[ConsolidationFact]) -> None:
        """
        Add facts to replay buffer for later training.

        Thread-safe. Facts are added to a bounded FIFO queue.

        Args:
            facts: List of ConsolidationFact objects to add
        """
        with self._lock:
            for fact in facts:
                self._replay_buffer.append(fact)

    def consolidate(
        self,
        facts: List[ConsolidationFact],
        epochs: int = 50,
        batch_size: int = 32,
        replay_ratio: float = 0.3,
    ) -> float:
        """
        Train neural network on new facts with experience replay.

        This is the core consolidation operation, typically called
        from a background thread.

        Args:
            facts: New facts to learn
            epochs: Number of training epochs
            batch_size: Training batch size
            replay_ratio: Fraction of batch from replay buffer

        Returns:
            Final training loss
        """
        with self._lock:
            if not facts:
                return 0.0

            self._network.train()
            optimizer = self._ensure_optimizer()

            # Register new labels and prepare training data
            key_vectors = []
            target_indices = []

            for fact in facts:
                idx = self._get_or_create_index(fact.value_label)
                key_vectors.append(fact.key_vector)
                target_indices.append(idx)

            # Add to replay buffer
            self.add_to_replay(facts)

            keys_tensor = torch.stack(key_vectors)
            targets_tensor = torch.tensor(target_indices, dtype=torch.long)

            total_loss = 0.0
            n_batches = 0

            for epoch in range(epochs):
                # Shuffle for each epoch
                perm = torch.randperm(len(keys_tensor))
                keys_shuffled = keys_tensor[perm]
                targets_shuffled = targets_tensor[perm]

                for i in range(0, len(keys_shuffled), batch_size):
                    batch_keys = keys_shuffled[i:i + batch_size]
                    batch_targets = targets_shuffled[i:i + batch_size]

                    # Mix in replay samples
                    if self._replay_buffer and replay_ratio > 0:
                        n_replay = max(1, int(len(batch_keys) * replay_ratio))
                        replay_samples = self._sample_replay(n_replay)

                        if replay_samples:
                            replay_keys = torch.stack([f.key_vector for f in replay_samples])
                            replay_targets = torch.tensor(
                                [self._label_to_index.get(f.value_label, 0) for f in replay_samples],
                                dtype=torch.long
                            )
                            batch_keys = torch.cat([batch_keys, replay_keys])
                            batch_targets = torch.cat([batch_targets, replay_targets])

                    # Forward pass
                    optimizer.zero_grad()
                    logits = self._network(batch_keys)

                    # Clamp targets to valid range
                    batch_targets = batch_targets.clamp(0, logits.shape[-1] - 1)

                    loss = F.cross_entropy(logits, batch_targets)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

            self._trained_samples += len(facts)
            self._network.eval()

            return total_loss / max(1, n_batches)

    def _sample_replay(self, n: int) -> List[ConsolidationFact]:
        """Sample n random items from replay buffer."""
        if not self._replay_buffer or n <= 0:
            return []

        n = min(n, len(self._replay_buffer))
        indices = torch.randperm(len(self._replay_buffer))[:n].tolist()
        return [self._replay_buffer[i] for i in indices]

    def query(self, key: torch.Tensor) -> Tuple[Optional[str], float]:
        """
        Query neural memory for a key.

        O(1) lookup via classification head - no candidate search needed.

        Args:
            key: Query key vector

        Returns:
            Tuple of (predicted_label, confidence) where confidence is
            the softmax probability of the predicted class.
            Returns (None, 0.0) if no vocabulary exists.
        """
        with self._lock:
            if not self._index_to_label:
                return None, 0.0

            self._network.eval()
            with torch.no_grad():
                logits = self._network(key)
                probs = F.softmax(logits, dim=-1)

                # Get prediction
                max_prob, max_idx = probs.max(dim=-1)
                idx = max_idx.item()
                confidence = max_prob.item()

                # Map index to label
                label = self._index_to_label.get(idx)
                if label is None:
                    return None, 0.0

                return label, confidence

    def query_with_index(self, key: torch.Tensor) -> Tuple[int, float]:
        """
        Query and return index instead of label.

        Useful when working directly with indices for efficiency.

        Args:
            key: Query key vector

        Returns:
            Tuple of (predicted_index, confidence)
        """
        with self._lock:
            if not self._index_to_label:
                return -1, 0.0

            self._network.eval()
            with torch.no_grad():
                logits = self._network(key)
                probs = F.softmax(logits, dim=-1)
                max_prob, max_idx = probs.max(dim=-1)

                return max_idx.item(), max_prob.item()

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (number of unique labels)."""
        return len(self._index_to_label)

    @property
    def trained_samples(self) -> int:
        """Total number of samples trained on."""
        return self._trained_samples

    @property
    def replay_buffer_size(self) -> int:
        """Current size of replay buffer."""
        return len(self._replay_buffer)

    def state_dict(self) -> dict:
        """Get state dictionary for persistence."""
        with self._lock:
            return {
                "network": self._network.state_dict(),
                "index_to_label": dict(self._index_to_label),
                "label_to_index": dict(self._label_to_index),
                "next_index": self._next_index,
                "trained_samples": self._trained_samples,
                # Architecture params needed for reconstruction
                "input_dim": self._input_dim,
                "hidden_dim": self._hidden_dim,
                "output_dim": self._network.output_dim,
            }

    def load_state_dict(self, state: dict) -> None:
        """Load state dictionary."""
        with self._lock:
            # Restore vocabulary first (needed for network size)
            self._index_to_label = state.get("index_to_label", {})
            # Convert string keys back to int
            self._index_to_label = {int(k): v for k, v in self._index_to_label.items()}
            self._label_to_index = state.get("label_to_index", {})
            self._next_index = state.get("next_index", 0)
            self._trained_samples = state.get("trained_samples", 0)

            # Recreate network with matching architecture
            input_dim = state.get("input_dim", self._input_dim)
            hidden_dim = state.get("hidden_dim", self._hidden_dim)
            output_dim = state.get("output_dim", self._network.output_dim)

            self._input_dim = input_dim
            self._hidden_dim = hidden_dim
            self._network = NeuralMemoryNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )

            # Load network weights
            self._network.load_state_dict(state["network"])

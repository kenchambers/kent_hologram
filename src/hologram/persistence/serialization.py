"""
Serialization utilities for MemoryTrace persistence.

Provides functions to save and load MemoryTrace objects to/from disk.
Enables persistent storage of holographic interference patterns across
sessions.

Design:
- Serializes the bundled trace vector to binary format
- Stores metadata (fact count, saturation estimate) for validation
- Supports both individual trace serialization and batch operations
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from hologram.memory.memory_trace import MemoryTrace
from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook


class MemoryTraceSerializer:
    """
    Serialization/deserialization for MemoryTrace objects.

    Handles persistent storage of holographic memory traces.
    Includes validation to detect corruption on load.

    Implementation strategy:
    1. Save trace vector as torch binary (preserves float32 precision)
    2. Save metadata as JSON (fact_count, dimensions, saturation)
    3. On load: validate metadata before restoring trace

    Not part of the distributed system initially, but included for
    completeness in the persistence layer.
    """

    @staticmethod
    def save(
        trace: MemoryTrace,
        path: Path,
        name: str = "trace"
    ) -> None:
        """
        Save a MemoryTrace to disk.

        Saves both the bundled trace vector and metadata.

        Args:
            trace: MemoryTrace object to serialize
            path: Directory to save in
            name: Base filename (will create name.pt and name.json)

        Raises:
            IOError: If unable to write to path
            ValueError: If trace is invalid

        Example:
            >>> from hologram.memory.memory_trace import MemoryTrace
            >>> trace = MemoryTrace(VectorSpace(10000))
            >>> # ... store facts ...
            >>> MemoryTraceSerializer.save(trace, Path("/data"), "memory")
            >>> # Creates /data/memory.pt and /data/memory.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save trace vector
        vector_file = path / f"{name}.pt"
        torch.save(trace.trace_vector, vector_file)

        # Save metadata
        metadata_file = path / f"{name}.json"
        metadata = {
            'dimensions': trace._space.dimensions,
            'fact_count': trace.fact_count,
            'saturation': trace.saturation_estimate,
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load(
        path: Path,
        space: VectorSpace,
        name: str = "trace"
    ) -> MemoryTrace:
        """
        Load a MemoryTrace from disk.

        Restores the bundled trace vector and validates against metadata.

        Args:
            path: Directory containing saved trace
            space: VectorSpace to use (dimensions must match)
            name: Base filename used in save()

        Returns:
            Restored MemoryTrace object

        Raises:
            FileNotFoundError: If trace files don't exist
            ValueError: If metadata doesn't match VectorSpace
            RuntimeError: If trace vector is corrupted

        Example:
            >>> space = VectorSpace(10000)
            >>> trace = MemoryTraceSerializer.load(Path("/data"), space, "memory")
            >>> # Restores exact state from previous session
        """
        path = Path(path)

        # Validate files exist
        vector_file = path / f"{name}.pt"
        metadata_file = path / f"{name}.json"

        if not vector_file.exists():
            raise FileNotFoundError(f"Trace vector not found: {vector_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Validate metadata matches VectorSpace
        if metadata['dimensions'] != space.dimensions:
            raise ValueError(
                f"Saved trace dimensions {metadata['dimensions']} != "
                f"VectorSpace dimensions {space.dimensions}"
            )

        # Load trace vector
        trace_vector = torch.load(vector_file, weights_only=False)

        # Validate vector shape
        if trace_vector.shape != (space.dimensions,):
            raise ValueError(
                f"Trace vector shape {trace_vector.shape} != "
                f"expected {(space.dimensions,)}"
            )

        # Restore MemoryTrace
        trace = MemoryTrace(space)
        trace._trace = trace_vector
        trace._fact_count = metadata['fact_count']

        return trace


def save_memory_trace(
    trace: MemoryTrace,
    directory: str,
    name: str = "memory_trace"
) -> None:
    """
    Convenience function to save MemoryTrace.

    Args:
        trace: MemoryTrace to save
        directory: Directory path (string)
        name: Base filename

    Example:
        >>> from hologram.memory.memory_trace import MemoryTrace
        >>> trace = MemoryTrace(VectorSpace(10000))
        >>> save_memory_trace(trace, "/tmp/data", "session_1")
    """
    MemoryTraceSerializer.save(trace, Path(directory), name)


def load_memory_trace(
    directory: str,
    space: VectorSpace,
    name: str = "memory_trace"
) -> MemoryTrace:
    """
    Convenience function to load MemoryTrace.

    Args:
        directory: Directory path (string)
        space: VectorSpace (dimensions must match)
        name: Base filename

    Returns:
        Restored MemoryTrace

    Example:
        >>> from hologram.core.vector_space import VectorSpace
        >>> space = VectorSpace(10000)
        >>> trace = load_memory_trace("/tmp/data", space, "session_1")
    """
    return MemoryTraceSerializer.load(Path(directory), space, name)


class JsonSerializer:
    """Simple JSON serializer for dictionaries and lists."""

    @staticmethod
    def save(obj: Any, path: Path) -> None:
        """Save object as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load(path: Path) -> Any:
        """Load object from JSON."""
        path = Path(path)
        with open(path, 'r') as f:
            return json.load(f)


class TorchTensorSerializer:
    """Serializer for PyTorch tensors."""

    @staticmethod
    def save(tensor: torch.Tensor, path: Path) -> None:
        """Save tensor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)

    @staticmethod
    def load(path: Path) -> torch.Tensor:
        """Load tensor from disk."""
        path = Path(path)
        return torch.load(path, weights_only=False)


class VocabularySerializer:
    """Serializer for Codebook vocabulary."""

    @staticmethod
    def save(codebook: Codebook, path: Path) -> None:
        """
        Save codebook vocabulary (concept names only).

        Vectors are not saved since they can be regenerated
        deterministically from concept names.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract vocabulary from codebook cache
        vocabulary = list(codebook._cache.keys())

        with open(path, 'w') as f:
            json.dump({"vocabulary": vocabulary}, f, indent=2)

    @staticmethod
    def load(path: Path) -> List[str]:
        """
        Load vocabulary list.

        Returns list of concept names. Caller should regenerate
        vectors by encoding each concept.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data["vocabulary"]

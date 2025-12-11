"""
StateManager: Orchestrates saving and loading complete hologram state.

Manages persistence of the entire holographic memory system including:
- MemoryTrace (holographic interference pattern)
- Codebook vocabulary (for regenerating concept vectors)
- FactStore metadata (S-P-O triples with citations)
- Configuration (VectorSpace settings)
"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hologram.config.constants import DEFAULT_DIMENSIONS
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import Fact, FactStore
from hologram.persistence.serialization import (
    JsonSerializer,
    MemoryTraceSerializer,
    VocabularySerializer,
)


class StateManager:
    """
    Manages complete state persistence for holographic memory system.

    Provides atomic save/load operations that preserve the entire system state.
    Includes validation, versioning, and corruption detection.

    File structure:
    ```
    base_path/
        config.json              # VectorSpace configuration
        memory_trace/
            trace.pt             # Bundled interference pattern
            metadata.json        # Fact count, saturation
        codebook_vocab.json      # Concept names (vectors regenerated)
        facts.json               # S-P-O triples with metadata
        checksum.txt             # SHA-256 for validation
        manifest.json            # Save metadata (timestamp, version)
    ```

    Example:
        >>> manager = StateManager()
        >>> manager.save(fact_store, Path("./data/session1"))
        >>> # Later...
        >>> restored_fs = manager.load(Path("./data/session1"))
    """

    VERSION = "1.0.0"

    def __init__(self):
        self._memory_serializer = MemoryTraceSerializer()
        self._vocab_serializer = VocabularySerializer()
        self._json_serializer = JsonSerializer()

    def save(
        self,
        fact_store: FactStore,
        base_path: Path,
        description: str = ""
    ) -> None:
        """
        Save complete FactStore state to directory.

        Args:
            fact_store: FactStore to save
            base_path: Directory to save to (will be created)
            description: Optional description of this save

        Raises:
            IOError: If save fails
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # 1. Save configuration
        config = {
            "version": self.VERSION,
            "dimensions": fact_store._memory._space.dimensions,
            "dtype": str(fact_store._memory._space.dtype),
            "timestamp": datetime.now().isoformat(),
            "description": description,
        }
        config_path = base_path / "config.json"
        self._json_serializer.save(config, config_path)

        # 2. Save memory trace
        memory_path = base_path / "memory_trace"
        self._memory_serializer.save(fact_store._memory, memory_path)

        # 3. Save codebook vocabulary
        vocab_path = base_path / "codebook_vocab.json"
        self._vocab_serializer.save(fact_store._codebook, vocab_path)

        # 4. Save facts metadata
        facts_data = {
            "facts": [
                {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                    "source": f.source,
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in fact_store._facts
            ],
            "fact_count": fact_store.fact_count,
            "vocabulary_size": fact_store.vocabulary_size,
        }
        facts_path = base_path / "facts.json"
        self._json_serializer.save(facts_data, facts_path)

        # 5. Create manifest
        manifest = {
            "version": self.VERSION,
            "saved_at": datetime.now().isoformat(),
            "description": description,
            "files": {
                "config": "config.json",
                "memory_trace": "memory_trace/",
                "vocabulary": "codebook_vocab.json",
                "facts": "facts.json",
            },
            "stats": {
                "fact_count": fact_store.fact_count,
                "vocabulary_size": fact_store.vocabulary_size,
                "saturation": fact_store.saturation_estimate,
            }
        }
        manifest_path = base_path / "manifest.json"
        self._json_serializer.save(manifest, manifest_path)

        # 6. Generate checksum for validation
        checksum = self._compute_checksum(base_path)
        checksum_path = base_path / "checksum.txt"
        with open(checksum_path, 'w') as f:
            f.write(checksum)

    def load(
        self,
        base_path: Path,
        validate_checksum: bool = True
    ) -> FactStore:
        """
        Load complete FactStore state from directory.

        Args:
            base_path: Directory containing saved state
            validate_checksum: If True, verify checksum before loading

        Returns:
            Restored FactStore

        Raises:
            FileNotFoundError: If required files missing
            ValueError: If checksum invalid or version mismatch
        """
        base_path = Path(base_path)

        # 1. Validate checksum (optional but recommended)
        if validate_checksum:
            self._validate_checksum(base_path)

        # 2. Load and validate configuration
        config_path = base_path / "config.json"
        config = self._json_serializer.load(config_path)

        if config["version"] != self.VERSION:
            raise ValueError(
                f"Version mismatch: saved {config['version']}, "
                f"expected {self.VERSION}"
            )

        # 3. Create VectorSpace from config
        dimensions = config["dimensions"]
        vector_space = VectorSpace(dimensions=dimensions)

        # 4. Load memory trace
        memory_path = base_path / "memory_trace"
        memory_trace = self._memory_serializer.load(memory_path, vector_space)

        # 5. Create and populate Codebook
        vocab_path = base_path / "codebook_vocab.json"
        vocabulary = self._vocab_serializer.load(vocab_path)

        codebook = Codebook(vector_space)
        # Pre-populate cache by encoding all concepts
        for concept in vocabulary:
            codebook.encode(concept)

        # 6. Create FactStore and restore state
        fact_store = FactStore(vector_space, codebook)
        fact_store._memory = memory_trace

        # 7. Load facts metadata
        facts_path = base_path / "facts.json"
        facts_data = self._json_serializer.load(facts_path)

        for fact_dict in facts_data["facts"]:
            fact = Fact(
                subject=fact_dict["subject"],
                predicate=fact_dict["predicate"],
                object=fact_dict["object"],
                confidence=fact_dict["confidence"],
                source=fact_dict.get("source"),
                timestamp=datetime.fromisoformat(fact_dict["timestamp"])
            )
            fact_store._facts.append(fact)
            fact_store._value_vocab.add(fact.object)

        return fact_store

    def _compute_checksum(self, base_path: Path) -> str:
        """
        Compute SHA-256 checksum of all data files.

        Args:
            base_path: Directory containing files

        Returns:
            Hex digest of combined file hashes
        """
        hasher = hashlib.sha256()

        # Hash files in deterministic order
        files_to_hash = [
            "config.json",
            "memory_trace/trace.pt",
            "memory_trace/metadata.json",
            "codebook_vocab.json",
            "facts.json",
        ]

        for filepath in files_to_hash:
            full_path = base_path / filepath
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    def _validate_checksum(self, base_path: Path) -> None:
        """
        Validate saved checksum matches computed checksum.

        Args:
            base_path: Directory containing files

        Raises:
            ValueError: If checksums don't match (corruption detected)
            FileNotFoundError: If checksum file missing
        """
        checksum_path = base_path / "checksum.txt"
        if not checksum_path.exists():
            raise FileNotFoundError(f"Checksum file not found: {checksum_path}")

        with open(checksum_path, 'r') as f:
            saved_checksum = f.read().strip()

        computed_checksum = self._compute_checksum(base_path)

        if saved_checksum != computed_checksum:
            raise ValueError(
                "Checksum mismatch - data may be corrupted! "
                f"Expected {saved_checksum}, got {computed_checksum}"
            )

    def list_saves(self, base_directory: Path) -> list[dict]:
        """
        List all saved states in a directory.

        Args:
            base_directory: Directory containing multiple saves

        Returns:
            List of save metadata dicts
        """
        saves = []
        base_directory = Path(base_directory)

        if not base_directory.exists():
            return saves

        for item in base_directory.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = self._json_serializer.load(manifest_path)
                        manifest["path"] = str(item)
                        saves.append(manifest)
                    except Exception:
                        # Skip invalid saves
                        continue

        # Sort by timestamp (newest first)
        saves.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return saves

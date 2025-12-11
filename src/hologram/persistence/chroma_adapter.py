"""
ChromaDB adapter for persistent fact storage.

Stores facts with their holographic vectors in ChromaDB for persistence
across sessions. Facts can be retrieved via similarity search or exact lookup.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


class ChromaFactStore:
    """
    Persistent fact store using ChromaDB.

    Stores facts as documents with:
    - Embedding: HDC vector representation
    - Metadata: subject, predicate, object, source, timestamp
    - Document: human-readable fact string

    Supports:
    - Add facts with automatic persistence
    - Query by subject/predicate using HDC similarity
    - List all facts
    - Load existing facts on startup

    Example:
        >>> store = ChromaFactStore(codebook, "./data/facts")
        >>> store.add_fact("France", "capital", "Paris")
        >>> answer, conf = store.query("France", "capital")
        >>> # Facts persist across sessions
    """

    def __init__(
        self,
        codebook,
        persist_dir: str = "./data/hologram_facts",
        collection_name: str = "facts",
        auto_recover: bool = True,
    ):
        """
        Initialize ChromaDB fact store.

        Args:
            codebook: Codebook for encoding facts to vectors
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
            auto_recover: If True, automatically recover from corrupted database
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for persistent storage. "
                "Install with: pip install chromadb"
            )

        self._codebook = codebook
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        # Initialize ChromaDB with persistence, with error recovery
        try:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection with cosine distance
            # HDC vectors encode meaning in direction, not magnitude
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Hologram fact storage",
                    "hnsw:space": "cosine",  # Use cosine distance for HDC vectors
                },
            )
        except Exception as e:
            # Handle ChromaDB corruption errors
            error_msg = str(e)
            is_corruption_error = (
                "PanicException" in error_msg
                or "range start index" in error_msg
                or "out of range" in error_msg
                or "corrupted" in error_msg.lower()
            )

            if is_corruption_error and auto_recover:
                logger.warning(
                    f"ChromaDB database appears corrupted. Attempting recovery..."
                )
                self._recover_from_corruption()
            else:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB: {e}\n"
                    f"If this is a corruption issue, try deleting {self._persist_dir} "
                    f"or set auto_recover=True"
                ) from e

        logger.info(
            f"ChromaFactStore initialized: {self._collection.count()} facts loaded"
        )

    def _recover_from_corruption(self) -> None:
        """
        Recover from corrupted ChromaDB database by deleting and recreating it.
        
        This will lose all stored facts, but allows the system to continue working.
        """
        import shutil
        import chromadb
        from chromadb.config import Settings

        logger.warning(
            f"Recovering from corrupted database at {self._persist_dir}. "
            "All existing facts will be lost."
        )

        # Backup the corrupted directory
        backup_dir = self._persist_dir.parent / f"{self._persist_dir.name}_corrupted_backup"
        if self._persist_dir.exists():
            try:
                shutil.move(str(self._persist_dir), str(backup_dir))
                logger.info(f"Corrupted database backed up to: {backup_dir}")
            except Exception as backup_error:
                logger.warning(f"Could not backup corrupted database: {backup_error}")
                # Try to delete individual files
                try:
                    for file in self._persist_dir.rglob("*"):
                        if file.is_file():
                            file.unlink()
                    for dir in reversed(list(self._persist_dir.rglob("*"))):
                        if dir.is_dir():
                            dir.rmdir()
                    self._persist_dir.rmdir()
                except Exception:
                    pass

        # Recreate directory
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Reinitialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "description": "Hologram fact storage",
                "hnsw:space": "cosine",
            },
        )

        logger.info("Database recovery complete. Starting with empty fact store.")

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add a fact to persistent storage.

        Args:
            subject: Fact subject (e.g., "France")
            predicate: Fact predicate (e.g., "capital")
            obj: Fact object (e.g., "Paris")
            source: Optional source attribution

        Returns:
            Fact ID if new fact, None if duplicate (same subject+predicate)
        """
        from hologram.core.operations import Operations

        # Check for duplicate: query by subject+predicate metadata
        # Use query_by_metadata which handles the filtering correctly
        try:
            existing_facts = self.query_by_metadata(subject=subject, predicate=predicate)
            if existing_facts:
                # Duplicate found - return None to indicate no new fact learned
                logger.debug(f"Duplicate fact skipped: {subject} {predicate} {obj}")
                return None
        except Exception as e:
            # If query fails, continue with adding (fallback behavior)
            logger.warning(f"Duplicate check failed, proceeding: {e}")

        # Create fact ID (include object for uniqueness)
        fact_id = f"{subject}_{predicate}_{obj}".lower().replace(" ", "_")

        # Encode fact as HDC vector
        subj_vec = self._codebook.encode(subject)
        pred_vec = self._codebook.encode(predicate)

        # CRITICAL FIX: Store ONLY the key vector (subject bound with predicate)
        # DO NOT bundle with object - bundling destroys similarity!
        # The object is stored in metadata and returned on match.
        key_vec = Operations.bind(subj_vec, pred_vec)

        # Convert to list for ChromaDB
        embedding = key_vec.tolist()

        # Metadata
        metadata = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source": source or "conversation",
            "timestamp": datetime.now().isoformat(),
        }

        # Document (human-readable)
        document = f"{subject} {predicate} {obj}"

        # Upsert to handle duplicates (by ID)
        self._collection.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )

        logger.debug(f"Added fact: {document}")
        return fact_id

    def query(
        self,
        subject: str,
        predicate: str,
        n_results: int = 1,
    ) -> Tuple[str, float]:
        """
        Query for a fact by subject and predicate.

        Args:
            subject: Subject to query
            predicate: Predicate to query
            n_results: Number of results to consider

        Returns:
            Tuple of (object, confidence)
        """
        from hologram.core.operations import Operations
        from hologram.core.similarity import Similarity

        # Create query vector
        subj_vec = self._codebook.encode(subject)
        pred_vec = self._codebook.encode(predicate)
        query_vec = Operations.bind(subj_vec, pred_vec)

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_results,
            include=["metadatas", "embeddings", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return "", 0.0

        # Get best match
        metadata = results["metadatas"][0][0]
        distance = results["distances"][0][0]

        # Convert distance to similarity (using cosine distance)
        # For cosine distance: distance = 1 - cosine_similarity
        # So: similarity = 1 - distance
        confidence = max(0.0, 1.0 - distance)

        return metadata["object"], confidence

    def query_by_metadata(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[dict]:
        """
        Query facts by exact metadata match.

        Args:
            subject: Optional subject filter
            predicate: Optional predicate filter

        Returns:
            List of matching fact metadata
        """
        # ChromaDB limitation: where clause supports max one operator/condition in some versions.
        # We prioritizing filtering by subject (usually more selective) and filter the rest in Python.
        where = {}
        if subject:
            where["subject"] = subject
        elif predicate:
            where["predicate"] = predicate

        if not where:
            if not subject and not predicate:
                # No filters - return all
                results = self._collection.get(include=["metadatas"])
            else:
                # Should not happen given logic above, but safe fallback
                results = self._collection.get(include=["metadatas"])
        else:
            results = self._collection.get(
                where=where,
                include=["metadatas"],
            )

        # Post-filtering to ensure all conditions are met
        metadatas = results.get("metadatas", [])
        filtered_metadatas = []
        
        for meta in metadatas:
            if subject and meta.get("subject") != subject:
                continue
            if predicate and meta.get("predicate") != predicate:
                continue
            filtered_metadatas.append(meta)

        return filtered_metadatas

    def get_all_facts(self) -> List[dict]:
        """
        Get all stored facts.

        Returns:
            List of fact dictionaries with subject, predicate, object, source
        """
        results = self._collection.get(include=["metadatas"])
        return results.get("metadatas", [])

    @property
    def fact_count(self) -> int:
        """Number of stored facts."""
        return self._collection.count()

    def delete_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """
        Delete a specific fact.

        Args:
            subject: Fact subject
            predicate: Fact predicate
            obj: Fact object

        Returns:
            True if deleted, False if not found
        """
        fact_id = f"{subject}_{predicate}_{obj}".lower().replace(" ", "_")
        try:
            self._collection.delete(ids=[fact_id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Delete all facts."""
        # Get all IDs and delete
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def __repr__(self) -> str:
        return f"ChromaFactStore(facts={self.fact_count}, path={self._persist_dir})"


class ChromaResponseCorpus:
    """
    Persistent response corpus using ChromaDB.

    Stores conversational responses with their context vectors for retrieval
    across sessions. Similar to ChromaFactStore but for response patterns.

    Example:
        >>> corpus = ChromaResponseCorpus(codebook, "./data/corpus")
        >>> corpus.add_response(context_vec, "That's interesting!", IntentType.STATEMENT)
        >>> matches = corpus.retrieve(query_vec, IntentType.STATEMENT)
    """

    def __init__(
        self,
        codebook,
        persist_dir: str = "./data/hologram_corpus",
        collection_name: str = "responses",
        auto_recover: bool = True,
    ):
        """
        Initialize ChromaDB response corpus.

        Args:
            codebook: Codebook for encoding
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
            auto_recover: If True, automatically recover from corrupted database
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for persistent storage. "
                "Install with: pip install chromadb"
            )

        self._codebook = codebook
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        # Initialize ChromaDB with persistence
        try:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Hologram response corpus",
                    "hnsw:space": "cosine",
                },
            )
        except Exception as e:
            if auto_recover:
                logger.warning("ChromaDB corpus appears corrupted. Attempting recovery...")
                self._recover_from_corruption()
            else:
                raise RuntimeError(
                    f"Failed to initialize ChromaDB corpus: {e}"
                ) from e

        logger.info(
            f"ChromaResponseCorpus initialized: {self._collection.count()} responses loaded"
        )

    def _recover_from_corruption(self) -> None:
        """Recover from corrupted ChromaDB database."""
        import shutil
        import chromadb
        from chromadb.config import Settings

        logger.warning(
            f"Recovering from corrupted corpus database at {self._persist_dir}."
        )

        backup_dir = self._persist_dir.parent / f"{self._persist_dir.name}_corrupted_backup"
        if self._persist_dir.exists():
            try:
                shutil.move(str(self._persist_dir), str(backup_dir))
                logger.info(f"Corrupted database backed up to: {backup_dir}")
            except Exception:
                # Try to delete files individually
                try:
                    for file in self._persist_dir.rglob("*"):
                        if file.is_file():
                            file.unlink()
                    for dir in reversed(list(self._persist_dir.rglob("*"))):
                        if dir.is_dir():
                            dir.rmdir()
                    self._persist_dir.rmdir()
                except Exception:
                    pass

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "description": "Hologram response corpus",
                "hnsw:space": "cosine",
            },
        )

        logger.info("Corpus database recovery complete.")

    def add_response(
        self,
        context_vector: torch.Tensor,
        response: str,
        intent: str,
        style: str = "neutral",
        source: str = "learned",
    ) -> str:
        """
        Add a response to persistent storage.

        Args:
            context_vector: HDC vector representing context
            response: Response text
            intent: Intent type (as string)
            style: Style type (as string)
            source: Source identifier

        Returns:
            Response ID
        """
        import hashlib

        # Handle Enums
        if hasattr(intent, "value"):
            intent = intent.value
        if hasattr(style, "value"):
            style = style.value

        # Create response ID
        response_id = hashlib.md5(
            f"{response}_{intent}_{style}".encode()
        ).hexdigest()[:16]

        # Convert to list for ChromaDB
        embedding = context_vector.tolist()

        # Metadata
        metadata = {
            "response": response,
            "intent": intent,
            "style": style,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        # Upsert to handle duplicates
        self._collection.upsert(
            ids=[response_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[response],
        )

        logger.debug(f"Added response: {response[:50]}...")
        return response_id

    def retrieve(
        self,
        query_vector: torch.Tensor,
        intent: Optional[str] = None,
        style: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar responses via HDC resonance.

        Args:
            query_vector: Query vector to match against
            intent: Optional intent filter
            style: Optional style filter
            top_k: Number of results to return

        Returns:
            List of (response, similarity) tuples
        """
        from hologram.core.similarity import Similarity

        # Handle Enums
        if hasattr(intent, "value"):
            intent = intent.value
        if hasattr(style, "value"):
            style = style.value

        # Build where clause. Chroma expects at most one operator; if both intent
        # and style are provided, we query without filters and filter client-side.
        where = None
        if intent and not style:
            where = {"intent": intent}
        elif style and not intent:
            where = {"style": style}

        # Query ChromaDB (if both filters, query without where and post-filter)
        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k * 3,  # fetch extra for post-filtering
            where=where,
            include=["metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Convert distances to similarities
        matches = []
        for i, metadata in enumerate(results["metadatas"][0]):
            # Post-filter when both intent and style are requested
            if intent and metadata.get("intent") != intent:
                continue
            if style and metadata.get("style") != style:
                continue

            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance)
            response = metadata["response"]
            matches.append((response, similarity))

        # Sort and trim to requested top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def get_entry_count(self) -> int:
        """Number of stored responses."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all responses."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def __repr__(self) -> str:
        return f"ChromaResponseCorpus(responses={self.get_entry_count()}, path={self._persist_dir})"

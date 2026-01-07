"""
EmergentLayerFactStore: Scalable fact storage with emergent layers.

Pattern: HierarchicalFactStore + EmergentLayerManager

Main entry point for CRAG + Emergent Category Networks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace
from hologram.retrieval.emergent_layers import (
    EmergentLayerManager,
    SemanticLayer,
)
from hologram.retrieval.layer_description import LayerDescriptionGenerator
from hologram.retrieval.description_cache import LayerDescriptionCache
from hologram.memory.transient_working_memory import TransientWorkingMemory
from hologram.core.crag_resonator import CRAGResonator


@dataclass
class FactAddResult:
    """
    Result of adding a fact.
    
    Attributes:
        layer_id: Layer where fact was stored
        layer_description: Description of the layer
        is_new_layer: Whether a new layer was created
        surprise: Surprise score
    """
    layer_id: str
    layer_description: str
    is_new_layer: bool
    surprise: float


@dataclass
class EmergentQueryResult:
    """
    Result of querying emergent layer store.
    
    Attributes:
        answer: Retrieved object/answer
        confidence: Confidence score
        layer_ids: Layers that were queried
        facts: Retrieved facts
    """
    answer: Optional[str]
    confidence: float
    layer_ids: List[str]
    facts: List[Tuple[str, str, str]]


@dataclass
class IngestResult:
    """
    Result of bulk ingestion.
    
    Attributes:
        total_facts: Total facts ingested
        new_layers_created: Number of new layers created
        layer_descriptions: List of all layer descriptions
        elapsed_time: Time taken in seconds
    """
    total_facts: int
    new_layers_created: int
    layer_descriptions: List[str]
    elapsed_time: float


class EmergentLayerFactStore:
    """
    Scalable fact store with emergent semantic layers.
    
    Pattern: EmergentLayerManager + FAISS per layer
    
    Zero hallucination guarantee:
    - Returns empty answer with 0 confidence if no facts retrieved
    - Layer provenance tracked for citations
    
    Methods:
        add_fact: Add fact with automatic layer routing
        query: Query with layer-aware retrieval
        bulk_ingest: Batch ingestion with progress tracking
        get_layer_stats: Get statistics about layers
    """
    
    def __init__(
        self,
        space: VectorSpace,
        codebook: Codebook,
        persist_path: str,
        use_hnsw: bool = True,
    ):
        """
        Initialize emergent layer fact store.
        
        Args:
            space: VectorSpace for dimensionality
            codebook: Codebook for encoding
            persist_path: Base path for persistence
            use_hnsw: Use HNSW indices
        """
        self._space = space
        self._codebook = codebook
        self._persist_path = persist_path
        
        # Create layer manager
        self._layer_manager = EmergentLayerManager(
            dimensions=space.dimensions,
            persist_base_path=f"{persist_path}/layers",
            use_hnsw=use_hnsw,
        )
        
        # Create description cache
        self._description_cache = LayerDescriptionCache(
            dimensions=space.dimensions,
            codebook=codebook,
            persist_path=f"{persist_path}/descriptions",
            use_hnsw=use_hnsw,
        )
        
        # Create description generator (needs resonator)
        # Will be set externally via set_description_generator
        self._description_generator: Optional[LayerDescriptionGenerator] = None
        
        # CRAG components for grounded querying
        self._crag_resonator: Optional[CRAGResonator] = None
        self._working_memory_capacity: int = 50
    
    def set_description_generator(
        self,
        generator: LayerDescriptionGenerator,
    ) -> None:
        """Set description generator (needs to be created with resonator)."""
        self._description_generator = generator
    
    def set_crag_resonator(
        self,
        resonator: CRAGResonator,
    ) -> None:
        """Set CRAG resonator for grounded querying."""
        self._crag_resonator = resonator
    
    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: Optional[str] = None,
    ) -> FactAddResult:
        """
        Add fact with automatic layer routing.
        
        Args:
            subject: Subject string
            predicate: Predicate string
            obj: Object string
            source: Optional source/citation
        
        Returns:
            FactAddResult with layer information
        """
        # Encode fact
        s_vec = self._codebook.encode(subject)
        p_vec = self._codebook.encode(predicate)
        o_vec = self._codebook.encode(obj)
        
        # Create content vector (for routing)
        # Use bundle(s, p) for consistency with query routing
        # This represents "what kind of questions this fact can answer"
        content_vec = Operations.bundle(s_vec, p_vec)
        content_text = f"{subject} {predicate} {obj}"
        
        # Route to layer or create new one
        routing_result = self._layer_manager.route_or_create(
            content_vec=content_vec,
            content_text=content_text,
            description_generator=self._description_generator,
        )
        
        layer = routing_result.layer
        
        # Store fact in layer's FAISS index
        # Key: bind(subject, predicate)
        key_vec = Operations.bind(s_vec, p_vec)
        
        layer.faiss_index.store(
            key_vec,
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "source": source or "",
            }
        )
        
        layer.fact_count += 1
        
        # Strengthen layer prototype
        self._layer_manager.strengthen_layer(
            layer.layer_id,
            content_vec,
        )
        
        # Update description cache if new layer
        if routing_result.is_new:
            desc_vec = self._codebook.encode(layer.description)
            self._description_cache.add_layer(
                layer.layer_id,
                layer.description,
                desc_vec,
            )
        
        return FactAddResult(
            layer_id=layer.layer_id,
            layer_description=layer.description,
            is_new_layer=routing_result.is_new,
            surprise=routing_result.surprise,
        )
    
    def query(
        self,
        subject: str,
        predicate: str,
        top_k: int = 20,
    ) -> EmergentQueryResult:
        """
        Query with layer-aware retrieval.
        
        Args:
            subject: Subject to query
            predicate: Predicate to query
            top_k: Number of facts to retrieve per layer
        
        Returns:
            EmergentQueryResult with answer and provenance
        """
        # Encode query
        s_vec = self._codebook.encode(subject)
        p_vec = self._codebook.encode(predicate)
        # Use consistent bundling with add_fact for routing
        # Note: We don't have the object in query, so we use subject+predicate
        # This will find layers that contain facts with this subject+predicate pattern
        query_vec = Operations.bundle(s_vec, p_vec)
        
        # Find matching layers via description cache
        layer_matches = self._description_cache.find_matching_layers(
            query_vec,
            top_k=3,
        )
        
        # If no layers, return empty
        if not layer_matches:
            return EmergentQueryResult(
                answer=None,
                confidence=0.0,
                layer_ids=[],
                facts=[],
            )
        
        # Retrieve facts from top matching layers
        all_facts = []
        layer_ids = []
        
        for match in layer_matches:
            layer = self._layer_manager.get_layer(match.layer_id)
            if layer is None or layer.fact_count == 0:
                continue
            
            layer_ids.append(layer.layer_id)
            
            # Query layer's FAISS index
            key_vec = Operations.bind(s_vec, p_vec)
            try:
                results = layer.faiss_index.query(key_vec, k=min(top_k, layer.fact_count))
                
                for _, similarity, metadata in results:
                    fact = (
                        metadata.get("subject", ""),
                        metadata.get("predicate", ""),
                        metadata.get("object", ""),
                    )
                    all_facts.append(fact)
            except ValueError:
                # Empty index or invalid query
                continue
        
        # If no facts retrieved, return empty
        if not all_facts:
            return EmergentQueryResult(
                answer=None,
                confidence=0.0,
                layer_ids=layer_ids,
                facts=[],
            )
        
        # Use CRAG resonator if available for grounded reasoning
        if self._crag_resonator is not None:
            # Load facts into transient working memory
            working_memory = TransientWorkingMemory(
                space=self._space,
                codebook=self._codebook,
                capacity=self._working_memory_capacity,
            )
            working_memory.load_facts(all_facts)

            # Create query thought vector
            query_thought = Operations.bundle(s_vec, p_vec)

            # Resonate with working memory for grounded result
            crag_result = self._crag_resonator.resonate_with_working_memory(
                thought=query_thought,
                working_memory=working_memory,
            )

            # Clear working memory
            working_memory.clear()

            # Simple grounding: answer is valid if it appears in retrieved facts
            # This handles HDC noise and predicate variations gracefully
            all_objects = [obj for _, _, obj in all_facts]
            is_grounded = crag_result.object_word in all_objects

            return EmergentQueryResult(
                answer=crag_result.object_word if is_grounded else None,
                confidence=crag_result.confidence if is_grounded else 0.0,
                layer_ids=layer_ids,
                facts=all_facts,
            )
        
        # Fallback: Simple heuristic if CRAG resonator not available
        # This maintains backward compatibility
        best_answer = None
        best_confidence = 0.0
        
        for subj, pred, obj in all_facts:
            if subj.lower() == subject.lower() and pred.lower() == predicate.lower():
                best_answer = obj
                best_confidence = 0.9  # High confidence for exact match
                break
        
        # If no exact match, use first fact
        if best_answer is None and all_facts:
            best_answer = all_facts[0][2]
            best_confidence = 0.5  # Lower confidence for approximate match
        
        return EmergentQueryResult(
            answer=best_answer,
            confidence=best_confidence,
            layer_ids=layer_ids,
            facts=all_facts,
        )
    
    def bulk_ingest(
        self,
        facts: List[Tuple[str, str, str]],
        batch_size: int = 1000,
        progress_callback=None,
    ) -> IngestResult:
        """
        Bulk ingest facts with automatic layer organization.
        
        Args:
            facts: List of (subject, predicate, object) tuples
            batch_size: Batch size for progress reporting
            progress_callback: Optional callback(done, total)
        
        Returns:
            IngestResult with statistics
        """
        import time
        start_time = time.time()
        
        initial_layers = len(self._layer_manager.get_all_layers())
        
        for i, (subject, predicate, obj) in enumerate(facts):
            self.add_fact(subject, predicate, obj)
            
            # Report progress
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, len(facts))
        
        # Final progress report
        if progress_callback:
            progress_callback(len(facts), len(facts))
        
        elapsed_time = time.time() - start_time
        final_layers = len(self._layer_manager.get_all_layers())
        new_layers = final_layers - initial_layers
        
        layer_descriptions = [
            layer.description
            for layer in self._layer_manager.get_all_layers()
        ]
        
        return IngestResult(
            total_facts=len(facts),
            new_layers_created=new_layers,
            layer_descriptions=layer_descriptions,
            elapsed_time=elapsed_time,
        )
    
    def get_layers(self) -> List[SemanticLayer]:
        """Get all semantic layers."""
        return self._layer_manager.get_all_layers()
    
    def get_layer_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about all layers.
        
        Returns:
            Dict mapping layer_id to stats dict
        """
        stats = {}
        for layer in self._layer_manager.get_all_layers():
            stats[layer.layer_id] = {
                "description": layer.description,
                "fact_count": layer.fact_count,
                "created_at": layer.created_at,
                "last_accessed": layer.last_accessed,
            }
        return stats
    
    def merge_similar_layers(self, threshold: float = 0.9) -> int:
        """Merge similar layers to prevent explosion."""
        return self._layer_manager.merge_similar_layers(threshold)
    
    def save(self) -> None:
        """Save all layers and caches to disk."""
        # Save description cache
        self._description_cache.save()
        
        # Save each layer's FAISS index
        for layer in self._layer_manager.get_all_layers():
            try:
                layer.faiss_index.save()
            except:
                pass  # Skip if empty
    
    def load(self) -> None:
        """Load layers and caches from disk."""
        # Load description cache
        try:
            self._description_cache.load()
        except:
            pass  # Skip if doesn't exist

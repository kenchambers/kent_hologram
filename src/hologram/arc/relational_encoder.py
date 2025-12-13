"""
RelationalEncoder: Encode ONLY salient object relationships.

Key insight from Dr. Nexus: The brain does NOT encode every relationship
between every object. It uses attention (thalamic gating) to select
*salient* relationships. We must do the same.

This encoder limits to:
- Adjacency (shares edge or corner)
- Identity (same color, same shape)
- Containment (bounding box overlap)

Capacity constraint: Typical ARC grid has 5-10 objects yielding ~15-30
relations (within 10k vector capacity). Hard cap at 30 relations.

This avoids "Holographic Saturation" from N^2 relationship explosion.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import torch

from hologram.arc.types import Object, Color
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace


@dataclass(frozen=True)
class SalientRelation:
    """A salient relationship between two objects."""
    subject: Object
    relation: str
    object_: Object
    salience: float  # 0.0-1.0 importance score


class RelationalEncoder:
    """
    Encode ONLY salient object relationships.

    Key constraint: Avoid N^2 explosion by limiting to:
    - Adjacency (shares edge or corner)
    - Identity (same color, same shape)
    - Containment (bounding box overlap)

    Typical ARC grid: 5-10 objects -> ~15-30 relations (within capacity)

    Attributes:
        MAX_RELATIONS: Hard cap on encoded relations (default: 30)
        SHAPE_SIMILARITY_THRESHOLD: Minimum similarity for same_shape (0.8)

    Example:
        >>> encoder = RelationalEncoder(codebook)
        >>> relations = encoder.extract_salient_relations(objects)
        >>> context_vec = encoder.encode_relation_context(relations)
    """

    # Salient relation types ordered by priority
    SALIENT_RELATIONS = [
        "adjacent_to",      # Shares edge or corner
        "same_color_as",    # Color identity
        "same_shape_as",    # Shape identity (DNA similarity > 0.8)
        "inside_of",        # Bounding box containment
    ]

    MAX_RELATIONS = 30  # Hard cap to respect bundling capacity
    SHAPE_SIMILARITY_THRESHOLD = 0.8  # For same_shape_as detection

    def __init__(
        self,
        codebook: Codebook,
        space: Optional[VectorSpace] = None,
        max_relations: int = MAX_RELATIONS,
    ):
        """
        Initialize relational encoder.

        Args:
            codebook: Codebook for encoding relation types
            space: VectorSpace for empty vectors (defaults to codebook's space)
            max_relations: Maximum relations to encode (default: 30)
        """
        self._codebook = codebook
        self._space = space or VectorSpace(dimensions=10000)
        self._ops = Operations
        self._max_relations = max_relations

        # Pre-encode relation type vectors
        self._relation_vectors = {
            rel: self._codebook.encode(f"relation_{rel}")
            for rel in self.SALIENT_RELATIONS
        }

        # Pre-encode role vectors for subject-relation-object triples
        self._role_subject = self._codebook.encode("__ROLE_REL_SUBJECT__")
        self._role_relation = self._codebook.encode("__ROLE_REL_TYPE__")
        self._role_object = self._codebook.encode("__ROLE_REL_OBJECT__")

    def extract_salient_relations(
        self,
        objects: List[Object],
    ) -> List[SalientRelation]:
        """
        Extract salient relations between objects.

        Only extracts relations between ADJACENT objects plus
        identity relations (same color/shape).

        Args:
            objects: List of detected objects in grid

        Returns:
            List of SalientRelation tuples, sorted by salience
        """
        if len(objects) < 2:
            return []

        relations: List[SalientRelation] = []

        for i, obj_a in enumerate(objects):
            # Only check relations with objects we haven't paired with yet
            for obj_b in objects[i + 1:]:
                # Check adjacency first (most important for spatial reasoning)
                if self._are_adjacent(obj_a, obj_b):
                    relations.append(SalientRelation(
                        subject=obj_a,
                        relation="adjacent_to",
                        object_=obj_b,
                        salience=1.0,  # Highest priority
                    ))
                    # Also check for color/shape identity of adjacent objects
                    if obj_a.color == obj_b.color:
                        relations.append(SalientRelation(
                            subject=obj_a,
                            relation="same_color_as",
                            object_=obj_b,
                            salience=0.8,
                        ))
                    if self._shapes_similar(obj_a, obj_b):
                        relations.append(SalientRelation(
                            subject=obj_a,
                            relation="same_shape_as",
                            object_=obj_b,
                            salience=0.7,
                        ))

                # Check containment
                if self._is_inside(obj_a, obj_b):
                    relations.append(SalientRelation(
                        subject=obj_a,
                        relation="inside_of",
                        object_=obj_b,
                        salience=0.9,
                    ))
                elif self._is_inside(obj_b, obj_a):
                    relations.append(SalientRelation(
                        subject=obj_b,
                        relation="inside_of",
                        object_=obj_a,
                        salience=0.9,
                    ))

        # Sort by salience (highest first) and enforce capacity limit
        relations.sort(key=lambda r: r.salience, reverse=True)
        if len(relations) > self._max_relations:
            relations = relations[:self._max_relations]

        return relations

    def encode_salient_relations(
        self,
        objects: List[Object],
    ) -> torch.Tensor:
        """
        Bundle only salient relations between objects.

        Args:
            objects: List of objects to find relations between

        Returns:
            Bundled relation vector, or empty vector if no relations
        """
        relations = self.extract_salient_relations(objects)
        return self.encode_relation_context(relations)

    def encode_relation_context(
        self,
        relations: List[SalientRelation],
    ) -> torch.Tensor:
        """
        Encode extracted relations into context vector.

        Each relation is encoded as:
            bind(subject_id, bind(relation_type, object_id))

        Where subject_id and object_id are derived from object hashes.

        Args:
            relations: List of salient relations

        Returns:
            Bundled context vector (sum of all relation encodings)
        """
        if not relations:
            return self._space.empty_vector()

        relation_vecs = []
        for rel in relations:
            vec = self._encode_triple(rel.subject, rel.relation, rel.object_)
            # Weight by salience
            relation_vecs.append(rel.salience * vec)

        return self._ops.bundle(*relation_vecs)

    def _encode_triple(
        self,
        subject: Object,
        relation: str,
        obj: Object,
    ) -> torch.Tensor:
        """
        Encode a subject-relation-object triple.

        Structure: bind(subject_vec, bind(relation_vec, object_vec))

        Args:
            subject: Subject object
            relation: Relation type string
            obj: Object (target of relation)

        Returns:
            Triple encoding vector
        """
        # Use shape hash as object identifier (position-invariant)
        subject_vec = self._codebook.encode(f"obj_{subject.shape_hash()}")
        object_vec = self._codebook.encode(f"obj_{obj.shape_hash()}")
        relation_vec = self._relation_vectors[relation]

        # Hierarchical binding: bind(subj, bind(rel, obj))
        rel_obj = self._ops.bind(relation_vec, object_vec)
        triple = self._ops.bind(subject_vec, rel_obj)

        return triple

    def _are_adjacent(self, a: Object, b: Object) -> bool:
        """
        Check if two objects share an edge or corner.

        Uses 8-connectivity (includes diagonals).

        Args:
            a: First object
            b: Second object

        Returns:
            True if any pixel of A is adjacent to any pixel of B
        """
        # Get pixel sets for efficient lookup
        b_pixels = b.pixels

        # 8-connectivity offsets
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal
        ]

        for ar, ac in a.pixels:
            for dr, dc in offsets:
                if (ar + dr, ac + dc) in b_pixels:
                    return True
        return False

    def _shapes_similar(self, a: Object, b: Object) -> bool:
        """
        Check if two objects have similar shapes.

        Uses shape hash equality for exact match, or could be
        extended to use mask IoU for fuzzy matching.

        Args:
            a: First object
            b: Second object

        Returns:
            True if shapes are similar enough
        """
        # Exact shape match via hash
        return a.shape_hash() == b.shape_hash()

    def _is_inside(self, inner: Object, outer: Object) -> bool:
        """
        Check if inner object is contained within outer's bounding box.

        For strict containment, all pixels of inner must be within
        outer's bounding box AND outer must be larger.

        Args:
            inner: Potentially contained object
            outer: Potentially containing object

        Returns:
            True if inner is inside outer
        """
        # Outer must be larger
        if outer.size <= inner.size:
            return False

        # Check bounding box containment
        ib = inner.bbox
        ob = outer.bbox

        return (
            ib.min_row >= ob.min_row and
            ib.max_row <= ob.max_row and
            ib.min_col >= ob.min_col and
            ib.max_col <= ob.max_col
        )

    def get_adjacent_objects(
        self,
        obj: Object,
        all_objects: List[Object],
    ) -> List[Object]:
        """
        Find all objects adjacent to a given object.

        Args:
            obj: Reference object
            all_objects: All objects to check against

        Returns:
            List of objects that share edge/corner with obj
        """
        adjacent = []
        for other in all_objects:
            if other is obj:
                continue
            if self._are_adjacent(obj, other):
                adjacent.append(other)
        return adjacent

    def __repr__(self) -> str:
        return (
            f"RelationalEncoder(max_relations={self._max_relations}, "
            f"relation_types={self.SALIENT_RELATIONS})"
        )

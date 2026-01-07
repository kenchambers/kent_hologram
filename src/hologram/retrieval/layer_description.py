"""
LayerDescriptionGenerator: Generate semantic descriptions for layers.

Pattern: Resonator (S-V-O factorization) + Codebook (vocabulary)

Key innovation: S-V-O factorization naturally forms layer descriptions!
Example:
  - Content: Facts about France, Paris, capitals, European cities
  - Resonator extracts: (Geography, contains, European_capitals)
  - Generated description: "Geography: European capitals and cities"
"""

from typing import List

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.resonator import Resonator


class LayerDescriptionGenerator:
    """
    Generates semantic descriptions for layers using resonator.
    
    Pattern: Resonator S-V-O factorization → human-readable description
    
    Methods:
        generate_description: Generate description from content samples
        extract_layer_topic: Extract (domain, relation, concept) tuple
    """
    
    def __init__(
        self,
        codebook: Codebook,
        resonator: Resonator,
    ):
        """
        Initialize description generator.
        
        Args:
            codebook: Codebook for encoding
            resonator: Resonator for factorization
        """
        self._codebook = codebook
        self._resonator = resonator
    
    def generate_description(
        self,
        content_samples: List[str],
        max_samples: int = 5,
    ) -> str:
        """
        Generate layer description from content samples.
        
        Args:
            content_samples: Sample content strings
            max_samples: Max samples to use for generation
        
        Returns:
            Human-readable layer description
        """
        if not content_samples:
            return "Empty Layer"
        
        # Limit samples
        samples = content_samples[:max_samples]
        
        # Encode samples as vectors
        sample_vecs = [self._codebook.encode(s) for s in samples]
        
        # Bundle into single thought vector
        thought = Operations.bundle(*sample_vecs)
        
        # Extract topic using resonator
        domain, relation, concept = self.extract_layer_topic(thought)
        
        # Compose description
        if domain and relation and concept:
            return f"{domain}: {relation} {concept}"
        elif domain and concept:
            return f"{domain}: {concept}"
        elif concept:
            return concept
        else:
            # Fallback to first sample
            return samples[0][:50] + ("..." if len(samples[0]) > 50 else "")
    
    def extract_layer_topic(
        self,
        content_vec: torch.Tensor,
    ) -> tuple[str, str, str]:
        """
        Extract (domain, relation, concept) from content vector.
        
        Args:
            content_vec: Bundled content vector
        
        Returns:
            Tuple of (domain, relation, concept)
        """
        # Define broad vocabulary for layer topics
        domain_vocab = [
            "Geography", "Science", "History", "Technology",
            "Culture", "Nature", "Mathematics", "Art",
            "Literature", "Sports", "Politics", "Economy",
        ]
        
        relation_vocab = [
            "contains", "describes", "explains", "relates",
            "discusses", "covers", "includes", "focuses",
        ]
        
        concept_vocab = [
            "capitals", "cities", "countries", "regions",
            "concepts", "theories", "facts", "events",
            "people", "places", "ideas", "principles",
            "systems", "processes", "structures", "patterns",
        ]
        
        # Combine domain and concept for subject/object
        noun_vocab = domain_vocab + concept_vocab
        
        # Resonate to extract structure
        result = self._resonator.resonate(
            content_vec,
            noun_vocab,
            relation_vocab,
        )
        
        # Map to domain/relation/concept
        # Subject is domain, verb is relation, object is concept
        domain = result.subject_word if result.subject_word in domain_vocab else ""
        relation = result.verb_word
        concept = result.object_word if result.object_word in concept_vocab else result.object_word
        
        return domain, relation, concept

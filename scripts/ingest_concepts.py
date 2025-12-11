#!/usr/bin/env python3
"""
Concept Ingester: Pre-train Hologram with general software engineering knowledge.

This script populates the FactStore with:
- Design Patterns (structure, purpose, implementation)
- Common Algorithms (complexity, implementation)
- Best Practices (SOLID principles, etc.)

This enables the system to "know" techniques before encountering specific tasks,
forming a "coding brain" that can be applied creatively via GLM.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hologram.container import HologramContainer
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import FactStore


# Canonicalization maps: normalize synonyms to standard terms
CANONICAL_PREDICATES = {
    # Complexity synonyms
    "performance": "time_complexity",
    "runtime": "time_complexity",
    "speed": "time_complexity",
    "efficiency": "time_complexity",
    "space": "space_complexity",
    "memory": "space_complexity",
    
    # Code structure synonyms
    "structure": "implementation",
    "code": "implementation",
    "example": "implementation",
    "template": "implementation",
    
    # Purpose synonyms
    "use": "purpose",
    "intent": "purpose",
    "goal": "purpose",
    "why": "purpose",
    
    # Type synonyms
    "kind": "type",
    "category": "type",
    "classification": "type",
}

CANONICAL_VALUES = {
    # Complexity notation standardization
    "O(N)": "O(n)",
    "o(n)": "O(n)",
    "linear time": "O(n)",
    "linear": "O(n)",
    
    "O(N^2)": "O(n^2)",
    "o(n^2)": "O(n^2)",
    "quadratic": "O(n^2)",
    
    "O(LOG N)": "O(log n)",
    "o(log n)": "O(log n)",
    "logarithmic": "O(log n)",
    
    "O(1)": "O(1)",
    "constant time": "O(1)",
    "constant": "O(1)",
    
    "O(N LOG N)": "O(n log n)",
    "o(n log n)": "O(n log n)",
    "linearithmic": "O(n log n)",
}


def canonicalize_predicate(predicate: str) -> str:
    """
    Normalize predicate to standard term.
    
    Args:
        predicate: Original predicate string
        
    Returns:
        Canonical predicate string
    """
    predicate_lower = predicate.lower().strip()
    return CANONICAL_PREDICATES.get(predicate_lower, predicate_lower)


def canonicalize_value(value: str) -> str:
    """
    Normalize value to standard term.
    
    Args:
        value: Original value string
        
    Returns:
        Canonical value string
    """
    value_stripped = value.strip()
    return CANONICAL_VALUES.get(value_stripped, value_stripped)


class ConceptIngester:
    """
    Ingests general coding concepts into the Hologram.
    """
    
    def __init__(self, persist_dir: str = "./data/code_concepts"):
        """
        Initialize concept ingester.
        
        Args:
            persist_dir: Directory for persisting learned concepts
        """
        self.persist_dir = persist_dir
        
        # Initialize Hologram container
        print("Initializing Hologram container...")
        self.container = HologramContainer(dimensions=10000)
        
        # Create fact store (will be persisted)
        print(f"Loading/creating fact store at: {persist_dir}")
        # For now, use in-memory FactStore (can integrate ChromaDB later)
        self.fact_store = FactStore(
            space=self.container._vector_space,
            codebook=self.container._codebook
        )
        
        self.facts_added = 0
        self.facts_skipped = 0
    
    def add_concept(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "Manual",
        confidence: float = 1.0
    ) -> bool:
        """
        Add a single concept fact with canonicalization.
        
        Args:
            subject: Subject (e.g., "BinarySearch", "Singleton")
            predicate: Predicate (e.g., "complexity", "purpose")
            obj: Object/value (e.g., "O(log n)", "Ensure single instance")
            source: Source of the concept
            confidence: Confidence in this fact
            
        Returns:
            True if fact was added, False if skipped (duplicate)
        """
        # Apply canonicalization
        predicate_canonical = canonicalize_predicate(predicate)
        obj_canonical = canonicalize_value(obj)
        
        # Add to fact store
        fact = self.fact_store.add_fact(
            subject=subject,
            predicate=predicate_canonical,
            obj=obj_canonical,
            source=source,
            confidence=confidence
        )
        
        if fact:
            self.facts_added += 1
            print(f"  ✓ {subject} --{predicate_canonical}--> {obj_canonical}")
            return True
        else:
            self.facts_skipped += 1
            return False
    
    def ingest_design_patterns(self) -> int:
        """
        Ingest common design patterns.
        
        Returns:
            Number of facts added
        """
        print("\n[Design Patterns] Ingesting...")
        
        patterns = {
            "Singleton": {
                "purpose": "Ensure a class has only one instance",
                "type": "Creational Pattern",
                "implementation": "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"
            },
            "Factory": {
                "purpose": "Create objects without specifying exact class",
                "type": "Creational Pattern",
                "implementation": "class Factory:\n    @staticmethod\n    def create(type):\n        if type == 'A': return ClassA()\n        elif type == 'B': return ClassB()"
            },
            "Observer": {
                "purpose": "Define one-to-many dependency between objects",
                "type": "Behavioral Pattern",
                "implementation": "class Subject:\n    def __init__(self):\n        self._observers = []\n    def attach(self, observer):\n        self._observers.append(observer)\n    def notify(self):\n        for obs in self._observers:\n            obs.update()"
            },
            "Strategy": {
                "purpose": "Define family of algorithms, make them interchangeable",
                "type": "Behavioral Pattern",
                "implementation": "class Context:\n    def __init__(self, strategy):\n        self._strategy = strategy\n    def execute(self):\n        return self._strategy.do_algorithm()"
            },
            "Decorator": {
                "purpose": "Add responsibilities to object dynamically",
                "type": "Structural Pattern",
                "implementation": "class Decorator:\n    def __init__(self, component):\n        self._component = component\n    def operation(self):\n        return self._component.operation() + ' + decoration'"
            },
            "Adapter": {
                "purpose": "Convert interface of class into another interface",
                "type": "Structural Pattern",
                "implementation": "class Adapter:\n    def __init__(self, adaptee):\n        self._adaptee = adaptee\n    def request(self):\n        return self._adaptee.specific_request()"
            },
        }
        
        initial_count = self.facts_added
        
        for pattern_name, properties in patterns.items():
            for predicate, value in properties.items():
                self.add_concept(
                    subject=pattern_name,
                    predicate=predicate,
                    obj=value,
                    source="Design Patterns Catalog"
                )
        
        added = self.facts_added - initial_count
        print(f"[Design Patterns] Added {added} facts")
        return added
    
    def ingest_algorithms(self) -> int:
        """
        Ingest common algorithms with complexity and implementation.
        
        Returns:
            Number of facts added
        """
        print("\n[Algorithms] Ingesting...")
        
        algorithms = {
            "BinarySearch": {
                "time_complexity": "O(log n)",
                "space_complexity": "O(1)",
                "type": "Search Algorithm",
                "implementation": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
            },
            "QuickSort": {
                "time_complexity": "O(n log n)",
                "space_complexity": "O(log n)",
                "type": "Sorting Algorithm",
                "implementation": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
            },
            "MergeSort": {
                "time_complexity": "O(n log n)",
                "space_complexity": "O(n)",
                "type": "Sorting Algorithm",
                "implementation": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)"
            },
            "BreadthFirstSearch": {
                "time_complexity": "O(V + E)",
                "space_complexity": "O(V)",
                "type": "Graph Algorithm",
                "implementation": "def bfs(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        vertex = queue.pop(0)\n        for neighbor in graph[vertex]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)"
            },
            "DepthFirstSearch": {
                "time_complexity": "O(V + E)",
                "space_complexity": "O(V)",
                "type": "Graph Algorithm",
                "implementation": "def dfs(graph, vertex, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(vertex)\n    for neighbor in graph[vertex]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited"
            },
            "DynamicProgramming": {
                "purpose": "Break problem into overlapping subproblems",
                "type": "Algorithmic Technique",
                "implementation": "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]"
            },
        }
        
        initial_count = self.facts_added
        
        for algo_name, properties in algorithms.items():
            for predicate, value in properties.items():
                self.add_concept(
                    subject=algo_name,
                    predicate=predicate,
                    obj=value,
                    source="Algorithms Catalog"
                )
        
        added = self.facts_added - initial_count
        print(f"[Algorithms] Added {added} facts")
        return added
    
    def ingest_best_practices(self) -> int:
        """
        Ingest software engineering best practices and principles.
        
        Returns:
            Number of facts added
        """
        print("\n[Best Practices] Ingesting...")
        
        practices = {
            "SOLID": {
                "type": "Design Principles",
                "purpose": "Five principles of object-oriented design"
            },
            "SingleResponsibility": {
                "type": "SOLID Principle",
                "purpose": "A class should have only one reason to change",
                "example": "class UserAuthenticator:\n    def authenticate(self, credentials):\n        pass\n\nclass UserPersistence:\n    def save(self, user):\n        pass"
            },
            "OpenClosed": {
                "type": "SOLID Principle",
                "purpose": "Open for extension, closed for modification",
                "example": "class Shape:\n    def area(self):\n        raise NotImplementedError\n\nclass Circle(Shape):\n    def area(self):\n        return 3.14 * self.radius ** 2"
            },
            "LiskovSubstitution": {
                "type": "SOLID Principle",
                "purpose": "Subtypes must be substitutable for their base types"
            },
            "InterfaceSegregation": {
                "type": "SOLID Principle",
                "purpose": "Clients should not depend on interfaces they don't use"
            },
            "DependencyInversion": {
                "type": "SOLID Principle",
                "purpose": "Depend on abstractions, not concretions",
                "example": "class Database:\n    def save(self, data):\n        raise NotImplementedError\n\nclass UserService:\n    def __init__(self, db: Database):\n        self.db = db"
            },
            "DRY": {
                "type": "Coding Principle",
                "purpose": "Don't Repeat Yourself - avoid code duplication",
                "full_name": "Don't Repeat Yourself"
            },
            "KISS": {
                "type": "Coding Principle",
                "purpose": "Keep It Simple, Stupid - favor simplicity",
                "full_name": "Keep It Simple Stupid"
            },
            "YAGNI": {
                "type": "Coding Principle",
                "purpose": "You Aren't Gonna Need It - don't add unused features",
                "full_name": "You Aren't Gonna Need It"
            },
        }
        
        initial_count = self.facts_added
        
        for concept_name, properties in practices.items():
            for predicate, value in properties.items():
                self.add_concept(
                    subject=concept_name,
                    predicate=predicate,
                    obj=value,
                    source="Best Practices Catalog"
                )
        
        added = self.facts_added - initial_count
        print(f"[Best Practices] Added {added} facts")
        return added
    
    def ingest_python_stdlib(self) -> int:
        """
        Ingest common Python standard library patterns.
        
        Returns:
            Number of facts added
        """
        print("\n[Python Stdlib] Ingesting...")
        
        stdlib = {
            "list_comprehension": {
                "type": "Python Syntax",
                "purpose": "Create lists concisely",
                "implementation": "[x for x in items if condition]"
            },
            "dict_comprehension": {
                "type": "Python Syntax",
                "purpose": "Create dictionaries concisely",
                "implementation": "{k: v for k, v in items if condition}"
            },
            "context_manager": {
                "type": "Python Pattern",
                "purpose": "Manage resources with automatic cleanup",
                "implementation": "with open('file.txt') as f:\n    data = f.read()"
            },
            "decorator": {
                "type": "Python Pattern",
                "purpose": "Modify function behavior without changing code",
                "implementation": "@decorator\ndef function():\n    pass"
            },
            "generator": {
                "type": "Python Pattern",
                "purpose": "Lazy iteration over data",
                "implementation": "def generator():\n    for i in range(n):\n        yield i"
            },
        }
        
        initial_count = self.facts_added
        
        for concept_name, properties in stdlib.items():
            for predicate, value in properties.items():
                self.add_concept(
                    subject=concept_name,
                    predicate=predicate,
                    obj=value,
                    source="Python Stdlib"
                )
        
        added = self.facts_added - initial_count
        print(f"[Python Stdlib] Added {added} facts")
        return added
    
    def ingest_all(self) -> Dict[str, int]:
        """
        Ingest all concept categories.
        
        Returns:
            Dictionary with counts per category
        """
        stats = {
            "design_patterns": self.ingest_design_patterns(),
            "algorithms": self.ingest_algorithms(),
            "best_practices": self.ingest_best_practices(),
            "python_stdlib": self.ingest_python_stdlib(),
        }
        return stats
    
    def print_summary(self):
        """Print ingestion summary."""
        print("\n" + "="*70)
        print("CONCEPT INGESTION SUMMARY")
        print("="*70)
        print(f"  Facts Added: {self.facts_added}")
        print(f"  Facts Skipped (duplicates): {self.facts_skipped}")
        print(f"  Total Facts in Store: {self.fact_store.fact_count}")
        print(f"  Vocabulary Size: {self.fact_store.vocabulary_size}")
        print(f"  Memory Saturation: {self.fact_store.saturation_estimate:.2%}")
        print("="*70)
    
    def test_retrieval(self):
        """Test that concepts can be retrieved correctly."""
        print("\n[Testing Retrieval]")
        
        test_queries = [
            ("BinarySearch", "time_complexity"),
            ("Singleton", "purpose"),
            ("DRY", "full_name"),
            ("QuickSort", "type"),
        ]
        
        for subject, predicate in test_queries:
            answer, confidence = self.fact_store.query(subject, predicate)
            print(f"  Query: {subject} --{predicate}--> ?")
            print(f"  Answer: {answer} (confidence: {confidence:.3f})")
        
        # Test reverse query
        print("\n[Testing Reverse Query]")
        subject, confidence = self.fact_store.query_subject("type", "Creational Pattern")
        print(f"  Query: ? --type--> Creational Pattern")
        print(f"  Answer: {subject} (confidence: {confidence:.3f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest general coding concepts into Hologram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all concepts
  python ingest_concepts.py
  
  # Ingest with custom persist directory
  python ingest_concepts.py --persist-dir ./my_concepts
  
  # Ingest and test retrieval
  python ingest_concepts.py --test
        """
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/code_concepts",
        help="Directory for persisting concepts (default: ./data/code_concepts)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test retrieval after ingestion"
    )
    
    args = parser.parse_args()
    
    # Create ingester
    ingester = ConceptIngester(persist_dir=args.persist_dir)
    
    # Ingest all concepts
    print("\n" + "="*70)
    print("CONCEPT INGESTION - Pre-training Coding Brain")
    print("="*70)
    
    stats = ingester.ingest_all()
    
    # Print summary
    ingester.print_summary()
    
    # Test retrieval if requested
    if args.test:
        ingester.test_retrieval()
    
    print(f"\nConcepts stored in memory (not yet persisted to {args.persist_dir})")
    print("Note: Persistence integration coming soon (ChromaDB/FAISS)")


if __name__ == "__main__":
    main()


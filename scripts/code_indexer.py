#!/usr/bin/env python3
"""
Code Indexer: Index a Python codebase into the Hologram.

This script uses AST (Abstract Syntax Tree) parsing to extract structured
information from a codebase and store it as facts:
- Function/Method signatures
- Class definitions and inheritance
- Function calls (caller -> callee relationships)
- Return type annotations
- Module structure

This provides project-specific "context" that complements the general
"coding brain" from concept ingestion.
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hologram.container import HologramContainer
from hologram.memory.fact_store import FactStore


class CodeIndexer:
    """
    Indexes Python codebase using AST parsing.
    """
    
    def __init__(self, persist_dir: str = "./data/code_index"):
        """
        Initialize code indexer.
        
        Args:
            persist_dir: Directory for persisting indexed code facts
        """
        self.persist_dir = persist_dir
        
        # Initialize Hologram container
        print("Initializing Hologram container...")
        self.container = HologramContainer(dimensions=10000)
        
        # Create fact store
        print(f"Loading/creating code index at: {persist_dir}")
        self.fact_store = FactStore(
            space=self.container._vector_space,
            codebook=self.container._codebook
        )
        
        self.facts_added = 0
        self.facts_skipped = 0
        self.files_processed = 0
        self.files_failed = 0
    
    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Add a code fact to the store.
        
        Args:
            subject: Subject (e.g., function name, class name)
            predicate: Relationship (e.g., "signature", "calls", "inherits")
            obj: Object/value
            source: Source file path
            confidence: Confidence in this fact
            
        Returns:
            True if added, False if skipped
        """
        fact = self.fact_store.add_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source=source,
            confidence=confidence
        )
        
        if fact:
            self.facts_added += 1
            return True
        else:
            self.facts_skipped += 1
            return False
    
    def extract_function_signature(self, node: ast.FunctionDef) -> str:
        """
        Extract function signature as a string.
        
        Args:
            node: AST FunctionDef node
            
        Returns:
            Signature string like "(arg1, arg2, kwarg=default)"
        """
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return f"({', '.join(args)})"
    
    def extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """
        Extract return type annotation if present.
        
        Args:
            node: AST FunctionDef node
            
        Returns:
            Return type string or None
        """
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except:
                return None
        return None
    
    def extract_function_calls(self, node: ast.AST) -> Set[str]:
        """
        Extract all function calls within a node.
        
        Args:
            node: AST node to analyze
            
        Returns:
            Set of called function names
        """
        calls = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Extract function name
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # For method calls like obj.method()
                    calls.add(child.func.attr)
        
        return calls
    
    def index_file(self, file_path: Path, base_path: Path) -> int:
        """
        Index a single Python file.
        
        Args:
            file_path: Path to Python file
            base_path: Base path of the project (for relative paths)
            
        Returns:
            Number of facts extracted
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            print(f"  ✗ Failed to read {file_path}: {e}")
            self.files_failed += 1
            return 0
        
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            print(f"  ✗ Syntax error in {file_path}: {e}")
            self.files_failed += 1
            return 0
        
        # Get relative path for source attribution
        try:
            rel_path = file_path.relative_to(base_path)
        except ValueError:
            rel_path = file_path
        
        source_str = str(rel_path)
        initial_count = self.facts_added
        
        # Extract module-level information
        module_name = file_path.stem
        
        # Walk the AST
        for node in ast.walk(tree):
            # Class definitions
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Store class existence
                self.add_fact(
                    subject=class_name,
                    predicate="type",
                    obj="class",
                    source=source_str
                )
                
                # Store module
                self.add_fact(
                    subject=class_name,
                    predicate="module",
                    obj=module_name,
                    source=source_str
                )
                
                # Store inheritance
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        self.add_fact(
                            subject=class_name,
                            predicate="inherits",
                            obj=base.id,
                            source=source_str
                        )
                        # Also store reverse relationship
                        self.add_fact(
                            subject=base.id,
                            predicate="parent_of",
                            obj=class_name,
                            source=source_str
                        )
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    # Store first line of docstring as purpose
                    purpose = docstring.split('\n')[0].strip()
                    if purpose:
                        self.add_fact(
                            subject=class_name,
                            predicate="purpose",
                            obj=purpose,
                            source=source_str
                        )
            
            # Function/Method definitions
            elif isinstance(node, ast.FunctionDef):
                func_name = node.name
                
                # Store function existence
                self.add_fact(
                    subject=func_name,
                    predicate="type",
                    obj="function",
                    source=source_str
                )
                
                # Store signature
                signature = self.extract_function_signature(node)
                self.add_fact(
                    subject=func_name,
                    predicate="signature",
                    obj=signature,
                    source=source_str
                )
                
                # Store return type if available
                return_type = self.extract_return_type(node)
                if return_type:
                    self.add_fact(
                        subject=func_name,
                        predicate="returns",
                        obj=return_type,
                        source=source_str
                    )
                
                # Store docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    purpose = docstring.split('\n')[0].strip()
                    if purpose:
                        self.add_fact(
                            subject=func_name,
                            predicate="purpose",
                            obj=purpose,
                            source=source_str
                        )
                
                # Extract function calls (call graph)
                calls = self.extract_function_calls(node)
                for called_func in calls:
                    # Store caller -> callee relationship
                    self.add_fact(
                        subject=func_name,
                        predicate="calls",
                        obj=called_func,
                        source=source_str
                    )
                    # Store reverse relationship
                    self.add_fact(
                        subject=called_func,
                        predicate="called_by",
                        obj=func_name,
                        source=source_str
                    )
        
        self.files_processed += 1
        facts_from_file = self.facts_added - initial_count
        
        if facts_from_file > 0:
            print(f"  ✓ {rel_path}: {facts_from_file} facts")
        
        return facts_from_file
    
    def index_directory(
        self,
        directory: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Recursively index all Python files in a directory.
        
        Args:
            directory: Directory to index
            exclude_patterns: List of patterns to exclude (e.g., ["test", "__pycache__"])
            
        Returns:
            Total number of facts extracted
        """
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "env",
                ".tox",
                "build",
                "dist",
                ".egg-info",
            ]
        
        print(f"\n[Indexing] {directory}")
        
        initial_count = self.facts_added
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        print(f"Found {len(python_files)} Python files")
        
        # Index each file
        for file_path in python_files:
            self.index_file(file_path, directory)
        
        return self.facts_added - initial_count
    
    def print_summary(self):
        """Print indexing summary."""
        print("\n" + "="*70)
        print("CODE INDEXING SUMMARY")
        print("="*70)
        print(f"  Files Processed: {self.files_processed}")
        print(f"  Files Failed: {self.files_failed}")
        print(f"  Facts Added: {self.facts_added}")
        print(f"  Facts Skipped (duplicates): {self.facts_skipped}")
        print(f"  Total Facts in Store: {self.fact_store.fact_count}")
        print(f"  Vocabulary Size: {self.fact_store.vocabulary_size}")
        print(f"  Memory Saturation: {self.fact_store.saturation_estimate:.2%}")
        print("="*70)
    
    def test_retrieval(self):
        """Test that indexed code can be queried."""
        print("\n[Testing Retrieval]")
        
        # Get a few facts to test
        sample_facts = self.fact_store.get_all_facts()[:5]
        
        if not sample_facts:
            print("  No facts to test")
            return
        
        for fact in sample_facts:
            answer, confidence = self.fact_store.query(fact.subject, fact.predicate)
            print(f"  Query: {fact.subject} --{fact.predicate}--> ?")
            print(f"  Expected: {fact.object}")
            print(f"  Got: {answer} (confidence: {confidence:.3f})")
            print()
    
    def query_call_graph(self, function_name: str):
        """
        Query the call graph for a function.
        
        Args:
            function_name: Name of the function to analyze
        """
        print(f"\n[Call Graph for '{function_name}']")
        
        # Find what this function calls
        calls_facts = self.fact_store.get_facts_by_subject(function_name)
        calls = [f.object for f in calls_facts if f.predicate == "calls"]
        
        if calls:
            print(f"  Calls: {', '.join(calls)}")
        else:
            print(f"  Calls: (none)")
        
        # Find what calls this function
        called_by_facts = self.fact_store.get_facts_by_subject(function_name)
        called_by = [f.object for f in called_by_facts if f.predicate == "called_by"]
        
        if called_by:
            print(f"  Called by: {', '.join(called_by)}")
        else:
            print(f"  Called by: (none)")
    
    def query_class_hierarchy(self, class_name: str):
        """
        Query the inheritance hierarchy for a class.
        
        Args:
            class_name: Name of the class to analyze
        """
        print(f"\n[Class Hierarchy for '{class_name}']")
        
        # Find parents
        inherits_facts = self.fact_store.get_facts_by_subject(class_name)
        parents = [f.object for f in inherits_facts if f.predicate == "inherits"]
        
        if parents:
            print(f"  Inherits from: {', '.join(parents)}")
        else:
            print(f"  Inherits from: (none - base class)")
        
        # Find children
        children_facts = self.fact_store.get_facts_by_subject(class_name)
        children = [f.object for f in children_facts if f.predicate == "parent_of"]
        
        if children:
            print(f"  Parent of: {', '.join(children)}")
        else:
            print(f"  Parent of: (none - leaf class)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index Python codebase into Hologram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current project
  python code_indexer.py --directory ./src
  
  # Index with custom persist directory
  python code_indexer.py --directory ./myproject --persist-dir ./data/myproject_index
  
  # Index and test retrieval
  python code_indexer.py --directory ./src --test
  
  # Query call graph after indexing
  python code_indexer.py --directory ./src --test --query-call-graph add_fact
  
  # Query class hierarchy
  python code_indexer.py --directory ./src --test --query-hierarchy FactStore
        """
    )
    
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Directory to index (will recursively process all .py files)"
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/code_index",
        help="Directory for persisting code index (default: ./data/code_index)"
    )
    
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Patterns to exclude (default: __pycache__, .git, venv, etc.)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test retrieval after indexing"
    )
    
    parser.add_argument(
        "--query-call-graph",
        type=str,
        metavar="FUNCTION",
        help="Query call graph for a specific function"
    )
    
    parser.add_argument(
        "--query-hierarchy",
        type=str,
        metavar="CLASS",
        help="Query class hierarchy for a specific class"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    if not args.directory.is_dir():
        print(f"Error: Not a directory: {args.directory}")
        sys.exit(1)
    
    # Create indexer
    indexer = CodeIndexer(persist_dir=args.persist_dir)
    
    # Index directory
    print("\n" + "="*70)
    print("CODE INDEXING - Building Project Context")
    print("="*70)
    
    indexer.index_directory(args.directory, exclude_patterns=args.exclude)
    
    # Print summary
    indexer.print_summary()
    
    # Test retrieval if requested
    if args.test:
        indexer.test_retrieval()
    
    # Query call graph if requested
    if args.query_call_graph:
        indexer.query_call_graph(args.query_call_graph)
    
    # Query hierarchy if requested
    if args.query_hierarchy:
        indexer.query_class_hierarchy(args.query_hierarchy)
    
    print(f"\nCode index stored in memory (not yet persisted to {args.persist_dir})")
    print("Note: Persistence integration coming soon (ChromaDB/FAISS)")


if __name__ == "__main__":
    main()




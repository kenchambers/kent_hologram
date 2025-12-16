"""BFS traversal over FactStore S-P-O triples for multi-file impact analysis."""

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Set

from hologram.memory.fact_store import FactStore


@dataclass
class DependencyResult:
    root_file: str
    affected_files: List[str]
    dependency_paths: Dict[str, List[str]]
    total_depth: int


class CodeDependencyGraph:
    def __init__(self, fact_store: FactStore, max_depth: int = 3):
        self._fact_store = fact_store
        self._max_depth = max_depth

    def get_affected_files(self, changed_file: str) -> DependencyResult:
        """BFS to find files affected by changes to this file."""
        visited: Set[str] = set()
        affected: List[str] = []
        paths: Dict[str, List[str]] = {}
        queue: deque = deque([(changed_file, 0, [changed_file])])

        while queue:
            current, depth, path = queue.popleft()
            if current in visited or depth > self._max_depth:
                continue
            visited.add(current)
            affected.append(current)
            paths[current] = path

            # Get callers of entities in this file
            for entity in self._get_entities_in_file(current):
                for dep_file in self._get_caller_files(entity):
                    if dep_file not in visited:
                        queue.append((dep_file, depth + 1, path + [dep_file]))

        return DependencyResult(changed_file, affected, paths, self._max_depth)

    def _get_entities_in_file(self, file_path: str) -> List[str]:
        """Get functions/classes defined in a file."""
        facts = self._fact_store.get_facts_by_predicate("module")
        return [f.subject for f in facts if file_path.endswith(f.object)]

    def _get_caller_files(self, entity: str) -> List[str]:
        """Get files containing functions that call this entity."""
        files = []
        seen = set()
        # Query "called_by" predicate (reverse of "calls")
        for fact in self._fact_store.get_facts_by_subject(entity):
            if fact.predicate == "called_by":
                caller = fact.object
                # Get file for caller
                for cf in self._fact_store.get_facts_by_subject(caller):
                    if cf.predicate == "module" and cf.object not in seen:
                        files.append(cf.object)
                        seen.add(cf.object)
        return files

"""
SWE-bench types for code generation.

Dataclasses for representing SWE tasks, patches, and results.
Follows the same pattern as hologram.arc.types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SWETask:
    """
    A SWE-bench task for code generation.

    Attributes:
        task_id: Unique task identifier
        repo: Repository name (e.g., "django/django")
        issue_text: Natural language issue description
        code_before: Files before the fix {filepath: content}
        code_after: Ground truth files after fix {filepath: content}
    """
    task_id: str
    repo: str
    issue_text: str
    code_before: Dict[str, str]
    code_after: Dict[str, str]


@dataclass
class CodePatch:
    """
    A single code patch representing one change.

    Attributes:
        file: Target file path
        operation: Operation type (add_line, delete_line, modify_line,
                   add_function, delete_function, modify_function)
        location: Location specifier (line number, function name, or class name)
        content: New content to add/modify
    """
    file: str
    operation: str
    location: str
    content: str

    def __str__(self) -> str:
        return f"{self.operation}({self.file}:{self.location})"


@dataclass
class PatchResult:
    """
    Result of code generation.

    Attributes:
        patches: List of code patches to apply
        confidence: Overall confidence score [0, 1]
        verification_passed: Whether HDC verification passed
        factorization: Optional factorization details
    """
    patches: List[CodePatch]
    confidence: float
    verification_passed: bool
    factorization: Optional[Dict[str, str]] = None

    @property
    def is_valid(self) -> bool:
        """Check if result is usable."""
        return self.verification_passed and self.confidence > 0.3 and len(self.patches) > 0


# Operation vocabulary - matches TransformationResonator pattern
OPERATIONS = [
    "add_line",
    "delete_line",
    "modify_line",
    "add_function",
    "delete_function",
    "modify_function",
    "add_import",
    "delete_import",
    "add_class",
    "modify_class",
]

# Location types for targeting
LOCATION_TYPES = [
    "line_number",
    "function_name",
    "class_name",
    "module_level",
    "after_import",
]

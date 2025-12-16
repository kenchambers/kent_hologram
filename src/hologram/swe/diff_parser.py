"""Safe diff parsing using stdlib difflib."""

import difflib
import re
from typing import List

from hologram.swe.types import CodePatch


def parse_unified_diff(before: str, after: str, file_path: str) -> List[CodePatch]:
    """Parse before/after into structured patches using difflib."""
    patches = []
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=''))
    old_line = 0  # Line number in "before" file
    new_line = 0  # Line number in "after" file

    for line in diff:
        if line.startswith('@@'):
            # Parse @@ -old_start,old_count +new_start,new_count @@
            match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)', line)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
        elif line.startswith('---') or line.startswith('+++'):
            continue  # Skip file headers
        elif line.startswith('-'):
            # Deletion from old file
            content = line[1:].strip()
            patches.append(CodePatch(
                file=file_path,
                operation="delete_line",
                location=str(old_line),
                content=content,
            ))
            old_line += 1
        elif line.startswith('+'):
            # Addition to new file
            content = line[1:].strip()
            patches.append(CodePatch(
                file=file_path,
                operation=_classify_operation(content),
                location=str(new_line),
                content=content,
            ))
            new_line += 1
        elif line.startswith(' '):
            # Context line - present in both files
            old_line += 1
            new_line += 1

    return patches


def _classify_operation(content: str) -> str:
    """Classify operation from content."""
    c = content.lower().strip()
    if c.startswith('import ') or c.startswith('from '): return "add_import"
    if c.startswith('def '): return "add_function"
    if c.startswith('class '): return "add_class"
    return "add_line" if 'raise ' in c or 'if ' in c else "modify_line"

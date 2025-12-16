#!/usr/bin/env python3
"""
SWE-bench Training Data Extractor.

Extracts training data for Kent's SWE module from multiple sources:
1. SWE-bench dataset (Princeton's benchmark)
2. GitHub Issues + PRs (via web search)
3. Synthetic patterns (generated)

Uses DuckDuckGo search + LLM extraction (like crew_trainer's WebTeacher)
to enhance training data with real-world code fix patterns.

Usage:
    # Extract from SWE-bench dataset
    python scripts/swe_data_extractor.py --source swe-bench --limit 100

    # Search web for code fix patterns
    python scripts/swe_data_extractor.py --source web --topics "Python null check" "input validation"

    # Generate synthetic patterns
    python scripts/swe_data_extractor.py --source synthetic --count 50

    # Combined: all sources
    python scripts/swe_data_extractor.py --all --output data/swe_training.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import difflib
import re

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import web search
try:
    from ddgs import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        WEB_SEARCH_AVAILABLE = True
    except ImportError:
        WEB_SEARCH_AVAILABLE = False
        print("Note: ddgs not installed. Web search unavailable.")
        print("To enable: pip install ddgs")

# Try to import LLM for extraction
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Note: LangChain LLMs not available. Using pattern-based extraction.")


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class CodeFixPattern:
    """A single code fix pattern for training."""
    pattern_id: str
    issue_keywords: List[str]  # Keywords that identify this issue type
    operation: str  # add_line, delete_line, modify_line, etc.
    location_type: str  # function, class, module, line
    template: str  # Code template with {placeholders}
    example_issue: str  # Example issue text
    example_fix: str  # Example fix code
    source: str  # Where this pattern came from
    confidence: float = 1.0


@dataclass
class SWETrainingExample:
    """A training example extracted from SWE-bench or web."""
    task_id: str
    repo: str
    issue_text: str
    file_path: str
    operation: str
    location: str
    code_before: str
    code_after: str
    diff_summary: str
    source: str


# =============================================================================
# Pattern Categories (Pre-defined)
# =============================================================================

SYNTHETIC_PATTERNS = [
    CodeFixPattern(
        pattern_id="null_check",
        issue_keywords=["null", "none", "undefined", "NoneType", "AttributeError"],
        operation="add_line",
        location_type="function",
        template="if {arg} is None:\n    raise ValueError('{arg} cannot be None')",
        example_issue="Fix null pointer exception in process()",
        example_fix="if x is None:\n    raise ValueError('x cannot be None')",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="division_zero",
        issue_keywords=["division", "zero", "ZeroDivisionError", "divide by zero"],
        operation="add_line",
        location_type="function",
        template="if {divisor} == 0:\n    raise ZeroDivisionError('Cannot divide by zero')",
        example_issue="Fix division by zero in divide()",
        example_fix="if b == 0:\n    raise ZeroDivisionError('Cannot divide by zero')",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="bounds_check",
        issue_keywords=["index", "range", "IndexError", "out of bounds", "overflow"],
        operation="add_line",
        location_type="function",
        template="if {index} < 0 or {index} >= len({array}):\n    raise IndexError('Index out of bounds')",
        example_issue="Fix index out of bounds in get_item()",
        example_fix="if i < 0 or i >= len(items):\n    raise IndexError('Index out of bounds')",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="type_check",
        issue_keywords=["type", "TypeError", "expected", "invalid type"],
        operation="add_line",
        location_type="function",
        template="if not isinstance({arg}, {expected_type}):\n    raise TypeError(f'Expected {expected_type}, got {{type({arg}).__name__}}')",
        example_issue="Add type validation to process()",
        example_fix="if not isinstance(x, int):\n    raise TypeError(f'Expected int, got {type(x).__name__}')",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="add_logging",
        issue_keywords=["logging", "log", "debug", "trace", "monitor"],
        operation="add_import",
        location_type="module",
        template="import logging\nlogger = logging.getLogger(__name__)",
        example_issue="Add logging to module",
        example_fix="import logging\nlogger = logging.getLogger(__name__)",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="add_logging_call",
        issue_keywords=["logging", "log", "debug", "trace"],
        operation="add_line",
        location_type="function",
        template="logger.{level}('{message}')",
        example_issue="Add logging to calculate()",
        example_fix="logger.info('Calculating result')",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="try_except",
        issue_keywords=["exception", "catch", "handle", "error handling", "try"],
        operation="modify_function",
        location_type="function",
        template="try:\n    {original_code}\nexcept {exception_type} as e:\n    {handler}",
        example_issue="Add error handling to fetch_data()",
        example_fix="try:\n    response = requests.get(url)\nexcept RequestException as e:\n    logger.error(f'Failed: {e}')\n    return None",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="add_import",
        issue_keywords=["import", "module", "ModuleNotFoundError", "ImportError"],
        operation="add_import",
        location_type="module",
        template="from {module} import {name}",
        example_issue="Add missing import for datetime",
        example_fix="from datetime import datetime",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="fix_return",
        issue_keywords=["return", "missing return", "None returned", "no return"],
        operation="add_line",
        location_type="function",
        template="return {value}",
        example_issue="Fix missing return in calculate()",
        example_fix="return result",
        source="synthetic",
    ),
    CodeFixPattern(
        pattern_id="string_format",
        issue_keywords=["format", "f-string", "string", "concatenation"],
        operation="modify_line",
        location_type="line",
        template="f'{{{var1}}} {{{var2}}}'",
        example_issue="Fix string formatting in log message",
        example_fix="f'User {user_id} logged in at {timestamp}'",
        source="synthetic",
    ),
]


# =============================================================================
# Web Search + LLM Extraction (like WebTeacher)
# =============================================================================

class CodePatternSearcher:
    """
    Search web for code fix patterns and extract structured data.

    Based on crew_trainer's WebTeacher but specialized for code fixes.
    """

    def __init__(self, llm_provider: str = "anthropic"):
        """Initialize with LLM for extraction."""
        if not WEB_SEARCH_AVAILABLE:
            raise RuntimeError("Web search requires ddgs. Install with: pip install ddgs")

        if not LLM_AVAILABLE:
            print("Warning: LLM not available. Using pattern-based extraction only.")
            self.llm = None
            return

        # Initialize LLM
        if llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            self.llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=api_key,
                temperature=0.3,
            )
        elif llm_provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.3,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search web for code patterns."""
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            return results if isinstance(results, list) else list(results)
        except Exception as e:
            print(f"Web search failed: {e}")
            return []

    def extract_code_patterns(
        self,
        text: str,
        topic: str
    ) -> List[CodeFixPattern]:
        """
        Extract code fix patterns from text using LLM.

        Args:
            text: Text from web search result
            topic: Search topic for context

        Returns:
            List of CodeFixPattern instances
        """
        if self.llm is None:
            return []

        extraction_prompt = f"""Extract code fix patterns from the following text about "{topic}".

For each pattern, identify:
- pattern_id: Short identifier (e.g., "null_check", "bounds_check")
- issue_keywords: List of keywords that identify this issue type
- operation: Type of fix (add_line, delete_line, modify_line, add_import, add_function, modify_function)
- location_type: Where in code (function, class, module, line)
- template: Code template with {{placeholders}} for variables
- example_issue: Example issue description
- example_fix: Example fix code

Output ONLY valid JSON:
{{
  "patterns": [
    {{
      "pattern_id": "...",
      "issue_keywords": ["...", "..."],
      "operation": "...",
      "location_type": "...",
      "template": "...",
      "example_issue": "...",
      "example_fix": "..."
    }}
  ]
}}

Rules:
1. Focus on REUSABLE patterns, not one-off fixes
2. Templates should have {{variable}} placeholders
3. Keep example_fix concise (max 5 lines)
4. issue_keywords should be words that appear in bug reports

Text:
{text[:3000]}

JSON Output:"""

        try:
            response = self.llm.invoke(extraction_prompt)
            content = response.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            patterns = []

            for p in data.get("patterns", []):
                if all(k in p for k in ["pattern_id", "issue_keywords", "operation"]):
                    patterns.append(CodeFixPattern(
                        pattern_id=p["pattern_id"],
                        issue_keywords=p.get("issue_keywords", []),
                        operation=p.get("operation", "modify_line"),
                        location_type=p.get("location_type", "function"),
                        template=p.get("template", ""),
                        example_issue=p.get("example_issue", ""),
                        example_fix=p.get("example_fix", ""),
                        source=f"web:{topic}",
                        confidence=0.8,  # Web-extracted patterns are less certain
                    ))

            return patterns

        except Exception as e:
            print(f"Pattern extraction failed: {e}")
            return []

    def search_and_extract(
        self,
        topics: List[str],
        max_results_per_topic: int = 3,
    ) -> List[CodeFixPattern]:
        """
        Search web for topics and extract code patterns.

        Args:
            topics: List of topics to search
            max_results_per_topic: Web results per topic

        Returns:
            List of extracted CodeFixPattern
        """
        all_patterns = []

        for topic in topics:
            print(f"\n[WebSearch] Searching: {topic}")

            # Add "code fix" or "bug fix" to improve results
            search_query = f"{topic} code fix example python"
            results = self.search_web(search_query, max_results=max_results_per_topic)

            print(f"  Found {len(results)} results")

            for i, result in enumerate(results):
                title = result.get("title", "")[:50]
                print(f"  [{i+1}] {title}...")

                text = f"{result.get('title', '')} {result.get('body', '')}"
                patterns = self.extract_code_patterns(text, topic)

                print(f"      Extracted {len(patterns)} patterns")
                all_patterns.extend(patterns)

        return all_patterns


# =============================================================================
# SWE-bench Dataset Extraction
# =============================================================================

def extract_from_swe_bench(
    data_dir: Optional[Path] = None,
    split: str = "dev",
    limit: int = 100,
) -> List[SWETrainingExample]:
    """
    Extract training examples from SWE-bench dataset.

    Args:
        data_dir: Path to SWE-bench data
        split: Dataset split (dev, test, lite)
        limit: Max examples to extract

    Returns:
        List of SWETrainingExample
    """
    # Try to find SWE-bench data
    if data_dir is None:
        candidates = [
            Path.home() / ".swe-bench" / split,
            Path("data") / "swe-bench" / split,
            Path(__file__).parent.parent / "data" / "swe-bench" / split,
        ]
        for candidate in candidates:
            if candidate.exists():
                data_dir = candidate
                break

    if data_dir is None or not data_dir.exists():
        print(f"SWE-bench data not found. Download from:")
        print("  https://github.com/princeton-nlp/SWE-bench")
        print("  Or use --source synthetic or --source web")
        return []

    print(f"[SWE-bench] Loading from {data_dir}")

    examples = []
    task_files = sorted(data_dir.glob("*.json"))[:limit]

    for task_file in task_files:
        try:
            with open(task_file) as f:
                data = json.load(f)

            # Extract diff information
            instance_id = data.get("instance_id", task_file.stem)
            repo = data.get("repo", "unknown")
            problem = data.get("problem_statement", "")

            # Get patch information
            patch = data.get("patch", "")

            # Parse diff to extract operation type
            operation, location, diff_summary = parse_diff(patch)

            # Get file paths
            files_before = data.get("base_commit_files", {})
            files_after = data.get("patch_files", {})

            for filepath in files_after.keys():
                examples.append(SWETrainingExample(
                    task_id=instance_id,
                    repo=repo,
                    issue_text=problem[:500],
                    file_path=filepath,
                    operation=operation,
                    location=location,
                    code_before=files_before.get(filepath, "")[:1000],
                    code_after=files_after.get(filepath, "")[:1000],
                    diff_summary=diff_summary,
                    source="swe-bench",
                ))

        except Exception as e:
            print(f"  Error processing {task_file}: {e}")

    print(f"[SWE-bench] Extracted {len(examples)} examples")
    return examples


def parse_diff(patch: str) -> Tuple[str, str, str]:
    """
    Parse a unified diff to extract operation type and location.

    Returns:
        Tuple of (operation, location, summary)
    """
    if not patch:
        return "modify_line", "unknown", "No diff available"

    lines_added = 0
    lines_removed = 0
    location = "unknown"

    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
            # Try to extract function/class name
            if "def " in line:
                match = re.search(r"def (\w+)", line)
                if match:
                    location = f"function:{match.group(1)}"
            elif "class " in line:
                match = re.search(r"class (\w+)", line)
                if match:
                    location = f"class:{match.group(1)}"
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1
        elif line.startswith("@@"):
            # Extract line numbers
            match = re.search(r"@@ -(\d+)", line)
            if match and location == "unknown":
                location = f"line:{match.group(1)}"

    # Determine operation type
    if lines_added > 0 and lines_removed == 0:
        operation = "add_line" if lines_added < 5 else "add_function"
    elif lines_removed > 0 and lines_added == 0:
        operation = "delete_line"
    else:
        operation = "modify_line" if lines_added < 10 else "modify_function"

    summary = f"+{lines_added}/-{lines_removed} lines at {location}"

    return operation, location, summary


# =============================================================================
# Training Data Generation
# =============================================================================

def generate_training_data(
    patterns: List[CodeFixPattern],
    examples: List[SWETrainingExample],
    output_path: Path,
) -> int:
    """
    Generate training data file from patterns and examples.

    Args:
        patterns: List of code fix patterns
        examples: List of SWE training examples
        output_path: Where to save the JSON file

    Returns:
        Number of entries written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "pattern_count": len(patterns),
            "example_count": len(examples),
        },
        "patterns": [asdict(p) for p in patterns],
        "examples": [asdict(e) for e in examples],
    }

    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\n[Output] Saved {len(patterns)} patterns + {len(examples)} examples to {output_path}")
    return len(patterns) + len(examples)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Training Data Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic patterns only
  python scripts/swe_data_extractor.py --source synthetic

  # Search web for code fix patterns
  python scripts/swe_data_extractor.py --source web --topics "null check python" "input validation"

  # Extract from SWE-bench dataset
  python scripts/swe_data_extractor.py --source swe-bench --limit 100

  # All sources combined
  python scripts/swe_data_extractor.py --all --output data/swe_training.json
        """
    )

    parser.add_argument(
        "--source",
        choices=["synthetic", "web", "swe-bench", "all"],
        default="synthetic",
        help="Data source (default: synthetic)"
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "python null check fix",
            "python input validation",
            "python error handling",
            "python type checking",
            "python logging best practices",
            "python bounds checking",
            "python import fix",
            "python return statement fix",
        ],
        help="Topics to search for web extraction"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max examples to extract from SWE-bench (default: 100)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/swe_training.json",
        help="Output file path (default: data/swe_training.json)"
    )

    parser.add_argument(
        "--web-results",
        type=int,
        default=3,
        help="Web results per topic (default: 3)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all sources (synthetic + web + swe-bench)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SWE-bench Training Data Extractor")
    print("=" * 60)

    patterns = []
    examples = []

    sources = []
    if args.all:
        sources = ["synthetic", "web", "swe-bench"]
    else:
        sources = [args.source]

    # Collect from each source
    for source in sources:
        print(f"\n--- Source: {source} ---")

        if source == "synthetic":
            patterns.extend(SYNTHETIC_PATTERNS)
            print(f"Added {len(SYNTHETIC_PATTERNS)} synthetic patterns")

        elif source == "web":
            if not WEB_SEARCH_AVAILABLE:
                print("Web search unavailable. Skipping.")
                continue
            try:
                searcher = CodePatternSearcher()
                web_patterns = searcher.search_and_extract(
                    topics=args.topics,
                    max_results_per_topic=args.web_results,
                )
                patterns.extend(web_patterns)
                print(f"Extracted {len(web_patterns)} patterns from web")
            except Exception as e:
                print(f"Web extraction failed: {e}")

        elif source == "swe-bench":
            swe_examples = extract_from_swe_bench(limit=args.limit)
            examples.extend(swe_examples)

    # Generate output
    output_path = Path(args.output)
    total = generate_training_data(patterns, examples, output_path)

    print("\n" + "=" * 60)
    print(f"Done! Generated {total} training entries.")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Show summary
    print("\nPattern Summary:")
    by_operation = {}
    for p in patterns:
        by_operation[p.operation] = by_operation.get(p.operation, 0) + 1
    for op, count in sorted(by_operation.items()):
        print(f"  {op}: {count}")

    if examples:
        print("\nExample Summary:")
        by_repo = {}
        for e in examples:
            repo = e.repo.split("/")[0] if "/" in e.repo else e.repo
            by_repo[repo] = by_repo.get(repo, 0) + 1
        for repo, count in sorted(by_repo.items(), key=lambda x: -x[1])[:10]:
            print(f"  {repo}: {count}")


if __name__ == "__main__":
    main()

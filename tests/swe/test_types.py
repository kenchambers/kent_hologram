"""Tests for SWE types."""

import pytest
from hologram.swe.types import SWETask, CodePatch, PatchResult, OPERATIONS


class TestSWETask:
    """Tests for SWETask dataclass."""

    def test_create_task(self):
        """Should create task with all fields."""
        task = SWETask(
            task_id="test_001",
            repo="test/repo",
            issue_text="Fix bug",
            code_before={"main.py": "x = 1"},
            code_after={"main.py": "x = 2"},
        )

        assert task.task_id == "test_001"
        assert task.repo == "test/repo"
        assert task.issue_text == "Fix bug"
        assert "main.py" in task.code_before
        assert "main.py" in task.code_after


class TestCodePatch:
    """Tests for CodePatch dataclass."""

    def test_create_patch(self):
        """Should create patch with all fields."""
        patch = CodePatch(
            file="utils.py",
            operation="add_line",
            location="42",
            content="x = 1",
        )

        assert patch.file == "utils.py"
        assert patch.operation == "add_line"
        assert patch.location == "42"
        assert patch.content == "x = 1"

    def test_patch_str(self):
        """Should format as readable string."""
        patch = CodePatch(
            file="utils.py",
            operation="add_line",
            location="42",
            content="x = 1",
        )

        assert "add_line" in str(patch)
        assert "utils.py" in str(patch)


class TestPatchResult:
    """Tests for PatchResult dataclass."""

    def test_create_result(self):
        """Should create result with patches."""
        patches = [
            CodePatch(file="a.py", operation="add_line", location="1", content="x"),
        ]
        result = PatchResult(
            patches=patches,
            confidence=0.8,
            verification_passed=True,
        )

        assert len(result.patches) == 1
        assert result.confidence == 0.8
        assert result.verification_passed is True

    def test_is_valid(self):
        """Should check validity correctly."""
        # Valid result
        result = PatchResult(
            patches=[CodePatch(file="a.py", operation="add", location="1", content="x")],
            confidence=0.5,
            verification_passed=True,
        )
        assert result.is_valid is True

        # Invalid: no patches
        result = PatchResult(patches=[], confidence=0.8, verification_passed=True)
        assert result.is_valid is False

        # Invalid: low confidence
        result = PatchResult(
            patches=[CodePatch(file="a.py", operation="add", location="1", content="x")],
            confidence=0.1,
            verification_passed=True,
        )
        assert result.is_valid is False

        # Invalid: verification failed
        result = PatchResult(
            patches=[CodePatch(file="a.py", operation="add", location="1", content="x")],
            confidence=0.8,
            verification_passed=False,
        )
        assert result.is_valid is False


class TestOperationsVocabulary:
    """Tests for OPERATIONS vocabulary."""

    def test_operations_defined(self):
        """Should have required operations."""
        assert "add_line" in OPERATIONS
        assert "delete_line" in OPERATIONS
        assert "modify_line" in OPERATIONS
        assert "add_function" in OPERATIONS

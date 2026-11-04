"""Tests for the Claude Code hook scripts in .claude/hooks/."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def run_hook(script: str, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git diff --staged",
        "git diff main...HEAD -- src/keel/core",
        "git log --oneline -n 20",
        "git show HEAD~1:src/keel/core/worker.py",
        "git status --short",
        "git blame -L 10,20 src/keel/core/replay.py",
    ],
)
def test_readonly_git_allows(command: str) -> None:
    result = run_hook("readonly_git.py", {"tool_input": {"command": command}})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m x",
        "git push",
        "git checkout -- .",
        "git diff; rm -rf src",
        "git diff && git push",
        "git log | tee out.txt",
        "git diff > patch.txt",
        "git show $(rm -rf src)",
        "git show `whoami`",
        "git diff --output=patch.txt",
        "git diff --ext-diff",
        "git -c alias.x=!rm x",
        "rm -rf src",
        "python -c 'print(1)'",
        "git",
        "git diff 'unterminated",
        "",
    ],
)
def test_readonly_git_blocks(command: str) -> None:
    result = run_hook("readonly_git.py", {"tool_input": {"command": command}})
    assert result.returncode == 2
    assert "Blocked" in result.stderr


def test_ruff_hook_formats_and_fixes(tmp_path: Path) -> None:
    target = tmp_path / "fixable.py"
    target.write_text("import os\nx = [1,2,   3]\n")

    result = run_hook("ruff_on_edit.py", {"tool_input": {"file_path": str(target)}})

    assert result.returncode == 0, result.stderr
    assert target.read_text() == "x = [1, 2, 3]\n"


def test_ruff_hook_reports_unfixable_issues(tmp_path: Path) -> None:
    target = tmp_path / "broken.py"
    target.write_text("def f():\n    return undefined_name\n")

    result = run_hook("ruff_on_edit.py", {"tool_input": {"file_path": str(target)}})

    assert result.returncode == 2
    assert "F821" in result.stderr


def test_ruff_hook_ignores_non_python_files(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("import os\n")

    result = run_hook("ruff_on_edit.py", {"tool_input": {"file_path": str(target)}})

    assert result.returncode == 0
    assert target.read_text() == "import os\n"

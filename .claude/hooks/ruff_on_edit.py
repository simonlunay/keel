"""Claude Code PostToolUse hook: format and lint a Python file right after it's edited.

Reads the hook payload (JSON) from stdin, and for ``.py`` files runs ``ruff check --fix``
and ``ruff format``. Problems ruff can't fix are printed to stderr with exit code 2,
which Claude Code shows to Claude so it fixes them in the same turn.

Written in Python rather than shell so it behaves the same on Windows and Linux. It uses
the project's own ruff from .venv, so the version matches the lockfile and CI.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


def find_ruff() -> str | None:
    for candidate in (
        PROJECT_DIR / ".venv" / "Scripts" / "ruff.exe",  # Windows
        PROJECT_DIR / ".venv" / "bin" / "ruff",  # POSIX
    ):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def main() -> int:
    payload = json.load(sys.stdin)
    file_path = payload.get("tool_input", {}).get("file_path")
    if not file_path or Path(file_path).suffix != ".py" or not Path(file_path).exists():
        return 0

    ruff = find_ruff()
    if ruff is None:
        print("ruff hook: ruff not found; run `uv sync`", file=sys.stderr)
        return 0  # Don't block edits just because the environment isn't set up.

    # Fix before formatting: fixes like removing an unused import can leave blank lines
    # behind that only the formatter cleans up.
    lint = subprocess.run(
        [ruff, "check", "--fix", "--quiet", file_path],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run([ruff, "format", "--quiet", file_path], cwd=PROJECT_DIR, check=False)
    if lint.returncode != 0:
        print(f"ruff found issues it couldn't fix in {file_path}:", file=sys.stderr)
        print(lint.stdout + lint.stderr, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

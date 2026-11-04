"""Claude Code PreToolUse hook: allow only read-only git commands through Bash.

Used by the reviewer subagent (.claude/agents/reviewer.md). Subagent ``tools`` lists can't
restrict Bash to particular commands, so this hook inspects each command and blocks (exit
code 2) anything that isn't a single, plain, read-only git invocation.

Deliberately strict: it's an allowlist, and anything it can't parse confidently is denied.
"""

import json
import shlex
import sys

ALLOWED_SUBCOMMANDS = frozenset({"diff", "log", "show", "status", "blame"})

# Characters that let a shell run a second command or redirect output.
SHELL_METACHARACTERS = (";", "&", "|", ">", "<", "`", "$(", "\n", "\r")

# Options of allowed subcommands that write files or run external programs.
FORBIDDEN_OPTION_PREFIXES = ("--output", "--ext-diff", "--textconv")


def check(command: str) -> str | None:
    """Return a reason the command is denied, or None if it's allowed."""
    if any(m in command for m in SHELL_METACHARACTERS):
        return "shell operators, redirection, and substitution are not allowed"
    try:
        argv = shlex.split(command)
    except ValueError:
        return "could not parse command"
    if len(argv) < 2 or argv[0] != "git":
        return "only git commands are allowed"
    # Requiring the subcommand right after `git` also rules out global options like
    # `git -c alias.x=!cmd x`, which can run arbitrary programs.
    if argv[1] not in ALLOWED_SUBCOMMANDS:
        return f"only `git {{{', '.join(sorted(ALLOWED_SUBCOMMANDS))}}}` are allowed"
    for arg in argv[2:]:
        if arg.startswith(FORBIDDEN_OPTION_PREFIXES):
            return f"option {arg.split('=')[0]} is not allowed"
    return None


def main() -> int:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    reason = check(command)
    if reason is not None:
        print(f"Blocked (reviewer is read-only): {reason}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

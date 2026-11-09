# keel

A durable execution runtime for AI agents. Runs survive crashes and resume from their last
completed step; side-effecting tool calls never execute twice. On recovery, completed steps
are replayed from a Postgres step log, returning recorded results instead of re-executing
tools or letting the LLM regenerate tool calls.

## Commands

```sh
uv sync                          # install deps into .venv
cp .env.example .env             # once; local config
docker compose up -d --wait      # start Postgres + Redis
docker compose down -v           # stop and delete data

uv run pytest                    # all tests (integration tests skip if services aren't configured)
uv run pytest tests/unit         # unit tests only, no Docker needed
uv run ruff check . --fix        # lint
uv run ruff format .             # format
uv run mypy                      # type check (strict)
```

## Layout

- `src/keel/core/`: the runtime: step log, runs, replay, run lifecycle (`runtime.py`).
  `worker.py` is still a stub (milestone 2).
- `src/keel/_api.py`: the public programming model (`@keel.tool`, `@keel.agent`, `keel.llm`).
- `src/keel/migrations/`: SQL applied by `keel.migrate(conn)`.
- `src/keel/testing/`: test helpers, including `MockLLM`. Not the runtime's LLM interface.
- `tests/unit/`: fast, no services. `tests/integration/`: need Postgres/Redis, marked
  `integration`.
- `docs/design.md`: the design. Source of truth for how the core works.

## Ownership: the design belongs to the project owner

- The owner owns `docs/design.md`, including the **invariants** and the **data model**.
  Do not edit the design doc unless explicitly asked to.
- Before implementing anything in `src/keel/core/`, check it against `docs/design.md`. If
  the design doesn't cover something (a table, a column, a state transition, a failure
  case, a public API shape), stop and ask. Don't fill the gap with your own design.
- Don't introduce interfaces, schemas, or abstractions in `core/` that commit to design
  decisions the doc hasn't made, even as "temporary" scaffolding.

## Definition of done

- Every change comes with tests. Bug fixes come with a test that fails without the fix.
- Code is not done until all three pass: `uv run ruff check . && uv run ruff format --check .`,
  `uv run mypy`, and `uv run pytest`.
- Changes to `core/` should be reviewed with the `reviewer` subagent
  (`.claude/agents/reviewer.md`) before they're called done.

## Working style

- Keep changes small and focused; one logical change per commit.
- Explain non-obvious decisions, in code comments where they'll stay true and in your
  summary otherwise, so the owner can understand and defend every line. Especially:
  ordering of writes, transaction boundaries, what happens on a crash between two lines,
  and why something is or isn't idempotent.
- Prefer clear over clever. This codebase's correctness has to be argued about.

## Environment notes

- Development happens on Windows; CI runs on Linux. Keep code and tests portable.
- Async tests on Windows use a selector event loop (`tests/conftest.py`) because psycopg
  async can't run on the default Proactor loop. The runtime will need the same care.
- A Claude Code hook runs ruff on every edited `.py` file (`.claude/hooks/ruff_on_edit.py`).

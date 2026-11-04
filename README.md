# keel

A durable execution runtime for AI agents.

Agents built with keel survive crashes and resume from their last completed step.
Side-effecting tool calls never execute twice: on recovery, completed steps are
replayed from a Postgres step log, returning recorded results instead of re-executing
tools or letting the LLM regenerate tool calls.

> **Status:** pre-alpha. The runtime is being designed in [docs/design.md](docs/design.md);
> nothing in `keel.core` is implemented yet.

## Development

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker.

```sh
uv sync                     # create .venv and install dev dependencies
cp .env.example .env        # local config
docker compose up -d        # Postgres + Redis
uv run pytest               # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## License

Apache-2.0

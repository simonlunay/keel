-- Milestone 1 schema: the runs and steps tables from "Data model" in docs/design.md.
-- Column types, nullability, and CHECK constraints were chosen by the owner when this
-- slice was implemented; the design doc lists the columns.
--
-- IF NOT EXISTS makes this safe to apply more than once. There is deliberately no
-- migration-tracking table yet (the design names only these two tables); how later
-- migrations are versioned is still to be decided.

CREATE TABLE IF NOT EXISTS runs (
    id             uuid PRIMARY KEY,
    agent_name     text NOT NULL,
    status         text NOT NULL
                   CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    input          jsonb NOT NULL,
    result         jsonb,            -- NULL until completed
    failure_reason text,             -- NULL unless failed
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS steps (
    run_id          uuid NOT NULL REFERENCES runs (id),
    step_number     integer NOT NULL CHECK (step_number >= 1),
    kind            text NOT NULL CHECK (kind IN ('llm', 'tool')),
    name            text NOT NULL,
    idempotency_key text NOT NULL UNIQUE,
    status          text NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
    attempts        integer NOT NULL CHECK (attempts >= 1),
    input           jsonb NOT NULL,
    result          jsonb,           -- NULL until completed
    started_at      timestamptz NOT NULL DEFAULT now(),
    completed_at    timestamptz,
    -- The design's "(run_id, step_number) is unique": the database itself rejects a
    -- duplicate step.
    PRIMARY KEY (run_id, step_number)
);

-- input_hash: hash of a step's canonicalized input, compared on replay alongside kind
-- and name (see "Executing a step" and Invariant 7 in docs/design.md).
--
-- The value is computed in Python (keel.core._json.input_hash), so there is no SQL
-- backfill: Postgres's jsonb::text is not the same canonical JSON and would produce
-- different hashes. NOT NULL with no default fails on a non-empty steps table, which is
-- intended pre-release (owner decision): there is no data to preserve, and a silent
-- NULL would weaken replay checks. IF NOT EXISTS keeps re-running migrate() safe.

ALTER TABLE steps ADD COLUMN IF NOT EXISTS input_hash text NOT NULL;

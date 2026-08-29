# DevelopmentApp AI Models

This repository currently contains the early Better You adaptive user model
backend: deterministic data models, SQLite persistence, scoring/recompute
logic, manual CLI tools, demo manifests, and tests.

The current goal is to make the model testable and auditable before building a
full app UI. Most workflows are driven through small command-line tools in
`tools/`.

## What Works Now

The manual lifecycle is runnable end to end:

```text
user event
-> user observation
-> belief evidence
-> belief-key canonicalization
-> belief recompute
-> duplicate evidence suppression
-> evidence invalidation/reset handling
-> read-only inspection
```

Important boundaries already implemented:

- LLM/extractor output can propose evidence and belief keys, but backend code
  authorizes persistence-affecting decisions.
- `BeliefEvidenceProposal` cannot set backend-owned fields such as confidence,
  aggregation authorization, duplicate suppression, or invalidation state.
- Evidence replacement can be proposed, but `authorize_evidence()` is the only
  place `aggregate_replacement` can be granted.
- Duplicate evidence is suppressed non-destructively; ledger rows remain
  auditable and are excluded from scoring via the shared active-evidence
  predicate.
- Belief-key canonicalization uses a backend registry. Unknown keys default to
  keep-separate, and unverified alias/merge requests go to manual review.
- Recompute uses only active, non-suppressed evidence and clears stale locks
  only after a successful recompute.

## Quick Demo

Run the basic demo manifest:

```bash
python tools/run_manifest.py --db demo.sqlite3 --manifest tools/manifests/demo_after_work_workout.json
```

Run the canonicalization demo manifest:

```bash
python tools/run_manifest.py --db canonical.sqlite3 --manifest tools/manifests/canonicalized_after_work_workout.json
```

Inspect a stored user model:

```bash
python tools/inspect_user_model.py --db canonical.sqlite3 --user-id usr_31 --include-inactive-evidence --pretty
```

The inspector shows events, observations, evidence, beliefs, and belief-key
canonicalization decisions from the SQLite database. It is read-only.

View the same data as an HTML page instead of JSON:

```bash
python tools/view_user_model.py --db canonical.sqlite3          # writes an HTML file and opens it
python tools/serve_user_model.py --db canonical.sqlite3         # serves it at http://localhost:8000
```

Both are strictly read-only (`Repository.readonly_at_path`, no writes). The
server re-reads the database on every request and accepts optional
`?user_id=` / `?belief_id=` query parameters. Pass `--demo` to either tool to
seed and use a throwaway demo database.

## Main Tools

- `tools/add_user_event.py`: insert one raw user event.
- `tools/add_user_observation.py`: insert one observation linked to events.
- `tools/add_belief_evidence.py`: create authorized belief evidence from an
  observation.
- `tools/resolve_belief_key.py`: resolve a proposed belief key through the
  backend canonicalization policy and save an audit record.
- `tools/recompute_belief.py`: recompute and save one belief from active
  evidence.
- `tools/suppress_duplicate_evidence.py`: mark duplicate evidence rows as
  suppressed without deleting them.
- `tools/invalidate_belief_evidence.py`: mark evidence inactive for deletion,
  reset, duplicate-suppression, policy-invalidation, or manual-review reasons.
- `tools/inspect_user_model.py`: read-only JSON inspection of stored model
  state.
- `tools/view_user_model.py`: render the stored model state as one read-only
  HTML page.
- `tools/serve_user_model.py`: serve that page locally (stdlib `http.server`),
  re-reading the database on every request. Read-only; GET/HEAD only.
- `tools/run_manifest.py`: run a JSON manifest through the existing tools.

## Manifest References

`tools/run_manifest.py` supports references from later steps to earlier step
outputs:

```json
{
  "type": "recompute_belief",
  "belief_key": "$steps.can_1.canonical_key"
}
```

This is how an authorized canonical key from `resolve_belief_key` flows into a
later recompute step without duplicating canonicalization logic in the manifest
runner.

## Tests

Run everything:

```bash
python -m unittest discover -s tests
```

Useful focused suites:

```bash
python -m unittest tests.event_intake.test_run_manifest -v
python -m unittest tests.beliefs.test_canonicalization -v
python -m unittest tests.beliefs.test_duplicate_suppression -v
python -m unittest tests.event_intake.test_resolve_belief_key -v
```

## Current Limitations

- There is no user-facing app yet; model state is inspected as JSON or through
  the read-only HTML viewer (`tools/view_user_model.py` /
  `tools/serve_user_model.py`).
- There is no live data collection pipeline yet; events are inserted manually
  or through manifests.
- Recommendation ranking/policy application is not built yet.
- Manual-review queues are represented by decisions/statuses, not a full review
  product workflow.
- The canonical belief-key registry is intentionally small and exact-match
  only.

## Good Next Step

Build the recommendation loop on top of the context/risk policy foundation in
`src/recommendations/context_policy.py`: candidate generation and ranking that
consumes only beliefs already authorized by `authorize_beliefs_for_context`.

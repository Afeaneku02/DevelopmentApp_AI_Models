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

Generate a recommendation from that model:

```bash
python tools/make_recommendation.py --db canonical.sqlite3 \
    --user-id usr_31 --context-key fitness_scheduling --recommendation-id rec_1
```

The recommendation is deterministic: beliefs are filtered by the context/risk
policy, candidate actions come from a fixed template table, and the ranking
score is `confidence * status_weight * diversity_factor`. High-risk or
manual-review contexts persist a `pending` recommendation instead of issuing
one. Every record freezes the belief state it saw and records the risk tier,
resolution path, and policy versions used.

Record what happened after a recommendation:

```bash
python tools/add_recommendation_outcome.py --db canonical.sqlite3 \
    --outcome-id out_1 --recommendation-id rec_1 \
    --followed followed --result successful --source app_event \
    --user-feedback "did the after-work slot, felt easier"
```

Outcomes are append-only and descriptive: many outcomes can reference one
recommendation, an outcome must point at an existing recommendation, and
recording one never changes the recommendation's frozen state.

Turn repeated outcomes into conservative learning signals:

```bash
python tools/learn_from_recommendation_outcomes.py --db canonical.sqlite3           # print proposals
python tools/learn_from_recommendation_outcomes.py --db canonical.sqlite3 --persist  # also store them
```

This groups outcomes by recommendation pattern (context + beliefs used) and,
only once enough repeated trials exist, proposes weak `repeated_pattern_summary`
belief-evidence for or against the beliefs the recommendation used. It makes
no causal claim, never mutates a belief, and never recomputes; without
`--persist` the database is opened read-only. The proposals are stored in
`outcome_learning_signals` -- promoting one into the real belief-evidence
ledger is a separate, later, backend-authorized step.

Inspect a stored user model:

```bash
python tools/inspect_user_model.py --db canonical.sqlite3 --user-id usr_31 --include-inactive-evidence --pretty
```

The inspector and the HTML viewer both show the full loop from the SQLite
database -- events, observations, observation-event links, evidence, beliefs,
belief-key canonicalization decisions, recommendations, and recommendation
outcomes. Both are strictly read-only.

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
- `tools/make_recommendation.py`: generate and persist one deterministic
  recommendation for a user in a context, using only beliefs authorized by
  the recommendation context/risk policy.
- `tools/add_recommendation_outcome.py`: append one outcome (followed /
  result / user feedback / measured result / source) for an existing
  recommendation. Descriptive only -- no causal claim, does not update
  beliefs.
- `tools/learn_from_recommendation_outcomes.py`: analyse repeated outcomes
  into conservative `repeated_pattern_summary` belief-evidence proposals.
  Prints by default; `--persist` appends analysis signals. No causal claim,
  no belief mutation, no recompute.
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
- Recommendation generation is a deterministic MVP
  (`tools/make_recommendation.py`): candidate actions come from a fixed
  template table and an auditable heuristic score, not an LLM or a learned
  ranker.
- Manual-review queues are represented by decisions/statuses, not a full review
  product workflow.
- The canonical belief-key registry is intentionally small and exact-match
  only.
- Outcome learning stops at conservative `belief_evidence` proposals in
  `outcome_learning_signals`; it does not yet authorize them into the ledger
  or recompute beliefs.

## Good Next Step

Add the backend authorization step that reviews `outcome_learning_signals`
proposals and, for the ones that clear policy, authorizes them into the
`belief_evidence` ledger (with proper leaf-event provenance and
`repeated_pattern_summary` overlap suppression) and triggers a recompute.

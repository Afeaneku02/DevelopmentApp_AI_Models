# DevelopmentApp AI Models

[![CI](https://github.com/Afeaneku02/DevelopmentApp_AI_Models/actions/workflows/ci.yml/badge.svg)](https://github.com/Afeaneku02/DevelopmentApp_AI_Models/actions/workflows/ci.yml)

This repository currently contains the early Better You adaptive user model
backend: deterministic data models, SQLite persistence, scoring/recompute
logic, manual CLI tools, demo manifests, and tests.

Every push and pull request runs GitHub Actions
(`.github/workflows/ci.yml`, Ubuntu + Python 3.12): it installs
`requirements.txt`, runs the full `unittest` suite
(`python -m unittest discover -s tests`), and then runs the evaluation
harness (`python tools/evaluate_user_model.py --manifest examples/evals`).
The build fails if any test or any eval scenario fails.

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

### See the whole closed loop in one command

```bash
# 1. build the full loop: event -> observation -> evidence -> canonicalization
#    -> initial belief -> recommendations -> outcomes -> learning signal
#    -> promoted belief_evidence -> recomputed belief
python tools/run_closed_loop_demo.py --db canonical.sqlite3 --run-id demo1

# 2. serve the read-only viewer (http://localhost:8000)
python tools/serve_user_model.py --db canonical.sqlite3

# 3. inspect the same data as JSON
python tools/inspect_user_model.py --db canonical.sqlite3 --user-id usr_demo1 --pretty
```

The demo is deterministic (pin `--run-id` for byte-identical output), makes
no LLM calls, and needs no external services. Re-running with the same
`--run-id` refuses and exits 1; use a new `--run-id` (or a fresh `--db`) to
build another independent chain.

### Or run the pieces by hand

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
`outcome_learning_signals` -- learning never promotes them itself.

Promote one signal's proposals into the real belief-evidence ledger (a
manual, one-signal, gated step):

```bash
python tools/promote_outcome_learning_signal.py --db canonical.sqlite3 --signal-id ols-...            # dry run
python tools/promote_outcome_learning_signal.py --db canonical.sqlite3 --signal-id ols-... --persist
python tools/promote_outcome_learning_signal.py --db canonical.sqlite3 --signal-id ols-... --persist --recompute
```

The deterministic gate requires `causal_claim=false`, a
`support`/`weak_contradiction` kind, existing proposals, and a trial
breakdown that still matches policy. Each proposal is re-provenanced to the
belief's current leaf events and authorized via `authorize_evidence` with
the signal's `independence_group`, so re-running promotion for the same
signal cannot add a second row. Nothing is written without `--persist`;
without `--recompute` the affected beliefs are left
`locked_until_recompute`.

Record a manual review of a signal instead of promoting it ad hoc, so the
decision is auditable:

```bash
python tools/review_outcome_learning_signal.py --db canonical.sqlite3 --signal-id ols-... \
    --review-id rev_1 --reviewer-id alice --decision rejected --notes "self-report only"
python tools/review_outcome_learning_signal.py --db canonical.sqlite3 --signal-id ols-... \
    --review-id rev_2 --reviewer-id alice --decision approved --promote --recompute \
    --notes "repeated positive trials, low risk"
```

Each review is appended to `outcome_learning_signal_reviews` with its
`reviewer_id`, `decision`, `notes`, `created_at`, and `policy_version`. A
`rejected` review promotes nothing. An `approved` review promotes only when
`--promote` is passed (and recomputes only with `--recompute`), delegating to
the same gated `promote_outcome_learning_signal` and recording exactly what it
did on the review. The promotion and the review's own audit row are written in
one atomic transaction, so an approved review can never leave promoted evidence
or a recomputed belief behind without a stored review to explain it. A signal
that already has an approved review is not
re-approved unless `--allow-duplicate` is given, and then both reviews stay in
the trail. The `--decision` and `--reviewer-id` always come from these flags,
never from model output -- `OutcomeLearningSignalReviewProposal` (the only
LLM-fillable surface) carries just `signal_id` and an optional
`suggested_notes`.

Inspect a stored user model:

```bash
python tools/inspect_user_model.py --db canonical.sqlite3 --user-id usr_31 --include-inactive-evidence --pretty
```

The inspector and the HTML viewer both show the full loop from the SQLite
database -- events, observations, observation-event links, evidence, beliefs,
belief-key canonicalization decisions, recommendations, recommendation
outcomes, outcome-learning signals (each flagged with whether it has been
promoted into belief_evidence and where it stands in manual review), and the
outcome-learning signal reviews themselves. Both are strictly read-only.

View the same data as an HTML page instead of JSON:

```bash
python tools/view_user_model.py --db canonical.sqlite3          # writes an HTML file and opens it
python tools/serve_user_model.py --db canonical.sqlite3         # serves it at http://localhost:8000
```

Both are strictly read-only (`Repository.readonly_at_path`, no writes). The
server re-reads the database on every request and accepts optional
`?user_id=` / `?belief_id=` query parameters. Pass `--demo` to either tool to
seed and use a throwaway demo database.

`tools/serve_user_model.py` also serves a read-only evaluation scorecard at
`/evals`: it runs the harness over `examples/evals/*.json` (each scenario in
its own fresh in-memory database -- the served database is never touched) and
shows summary counts, every scenario's pass/fail, and every check. A missing
or broken manifest renders as a failed scenario instead of crashing the
server. Point it elsewhere with `--evals-dir`.

A read-only manual review queue is served at `/reviews`: outcome-learning
signals still awaiting a human decision before promotion, shown with their
proposed evidence, plus the signals already approved/rejected (with a status
badge) and the full review trail. It reads the served database read-only,
honours `?user_id=`, and has no approve/reject controls -- decisions are
made from `tools/review_outcome_learning_signal.py`.

Every page carries a plain-link nav header ("User Model" / "Eval Scorecard"
/ "Review Queue") so `/`, `/evals`, and `/reviews` cross-link; there is no
JavaScript and no write action anywhere in the viewer.

Run the evaluation harness:

```bash
python tools/evaluate_user_model.py --manifest examples/evals              # scorecard, exits nonzero on failure
python tools/evaluate_user_model.py --manifest examples/evals --format json
python tools/evaluate_user_model.py --manifest examples/evals/clean_support_raises_confidence.json --verbose
```

Each *scenario manifest* under `examples/evals/` describes a sequence of
lifecycle steps (events, evidence, recompute, recommendation, outcomes,
outcome-learning, manual review, promotion) plus the expectations that should
hold afterwards. The harness replays every scenario against its own fresh
in-memory database through the same sanctioned functions the CLIs use -- it
adds no model behaviour -- then scores confidence, belief status, evidence
counts, recommendation belief-eligibility, outcome-learning signals, review
gates, and promoted evidence against the manifest. It exits `0` only when
every expectation in every scenario holds (`1` on any failure, `2` on a usage
error), so it drops straight into CI. The eight bundled scenarios cover clean
support raising confidence, mild vs. strong contradiction, invalidation
locking a stale belief, context-restricted recommendation inputs, conservative
weak-only outcome-learning proposals, and review approval vs. rejection.

## Main Tools

- `tools/run_closed_loop_demo.py`: build the entire loop (event -> ... ->
  promoted evidence -> recomputed belief) in one deterministic, re-run-safe
  command.
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
  re-reading the database on every request. Read-only; GET/HEAD only. Also
  serves the evaluation scorecard at `/evals` and the manual review queue at
  `/reviews`.
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
- `tools/promote_outcome_learning_signal.py`: promote one outcome-learning
  signal's proposals into real belief_evidence after deterministic checks.
  Dry run unless `--persist`; locks affected beliefs unless `--recompute`.
- `tools/review_outcome_learning_signal.py`: append one auditable manual
  review (`approved` / `rejected`, reviewer, notes) of an outcome-learning
  signal. A rejection promotes nothing; an approval promotes only with
  `--promote` (and recomputes only with `--recompute`), via the same gated
  promotion path. Duplicate approvals are blocked unless `--allow-duplicate`.
- `tools/run_manifest.py`: run a JSON manifest through the existing tools.
- `tools/evaluate_user_model.py`: run scenario manifests
  (`examples/evals/*.json`) through the full lifecycle and score the result
  against their expectations. Prints a scorecard (text or `--format json`);
  exits nonzero if any expectation fails.

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
python -m unittest tests.evals.test_evaluate_user_model -v
```

The evaluation harness (`tools/evaluate_user_model.py`) is itself a
scenario-level test of the whole lifecycle and is safe to run in CI:

```bash
python tools/evaluate_user_model.py --manifest examples/evals
```

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request (Ubuntu,
Python 3.12). It installs `requirements.txt`, then runs the same two commands
as above -- `python -m unittest discover -s tests` and
`python tools/evaluate_user_model.py --manifest examples/evals` -- so a
regression in either the unit tests or the end-to-end eval scenarios fails
the build. The badge at the top of this file reflects the latest run on the
default branch.

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
- Manual-review queues are represented by decisions/statuses and a per-signal
  review CLI (`tools/review_outcome_learning_signal.py`), not a full review
  product workflow with a queue UI or reviewer assignment.
- The canonical belief-key registry is intentionally small and exact-match
  only.
- Outcome-learning promotion is manual and one signal at a time
  (`tools/promote_outcome_learning_signal.py`); there is no scheduler or
  batch reviewer that promotes signals automatically.

## Good Next Step

Build a promotable-signal queue on top of `outcome_learning_signal_reviews`:
list the signals still `pending` review with their trial breakdowns, let a
reviewer work through them, and track reviewer assignment and policy-version
churn -- rather than reviewing one signal per CLI call.

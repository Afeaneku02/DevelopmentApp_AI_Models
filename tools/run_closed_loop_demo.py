#!/usr/bin/env python3
"""End-to-end closed-loop demo for the Better You adaptive user model.

Builds the whole loop in one command, using the same sanctioned functions
the individual CLIs use (no core logic is re-implemented here):

    user event
    -> user observation
    -> belief evidence
    -> initial belief (recompute)
    -> recommendations (deterministic, no LLM)
    -> recommendation outcomes
    -> outcome-learning signal
    -> promoted belief_evidence (backend-authorized)
    -> recomputed belief

Deterministic and safe to re-run:

- Every record and the user id is namespaced with ``--run-id`` (default: a
  wall-clock stamp, so a bare re-run just builds a fresh independent chain).
  Pin ``--run-id`` for byte-identical output across runs.
- If the chosen ``--run-id`` already has data in ``--db``, the demo refuses
  to touch it and exits 1 with a clear message -- it never overwrites or
  half-rebuilds an existing run.

No external services, no LLM calls.

Run:
    python tools/run_closed_loop_demo.py --db canonical.sqlite3
    python tools/run_closed_loop_demo.py --db canonical.sqlite3 --run-id demo1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beliefs.canonicalization import (  # noqa: E402
    BeliefKeyCanonicalizationProposal,
    authorize_belief_key_canonicalization,
)
from src.beliefs.models import authorize_evidence  # noqa: E402
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated  # noqa: E402
from src.beliefs.recompute import recompute_belief  # noqa: E402
from src.events.models import UserEvent  # noqa: E402
from src.observations.create_observation import create_observation_from_event  # noqa: E402
from src.recommendations.engine import generate_recommendation  # noqa: E402
from src.recommendations.models import RecommendationOutcome  # noqa: E402
from src.recommendations.outcome_learning import analyze_recommendation_outcomes  # noqa: E402
from src.recommendations.promotion import promote_outcome_learning_signal  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

_VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
_DEFAULT_AS_OF = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
_CONTEXT = "fitness_scheduling"
_BELIEF_KEY = "higher_adherence_after_work"
_TRIALS = 4


class DemoRunExists(RuntimeError):
    """The chosen run-id already has data in this database."""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full adaptive-user-model loop (event -> ... -> promoted evidence -> "
            "recomputed belief) in one command. Deterministic; safe to re-run with a new --run-id."
        ),
    )
    parser.add_argument("--db", required=True, help="SQLite database to build the demo in (created if new).")
    parser.add_argument(
        "--run-id", default=None, dest="run_id",
        help="Namespace for every record and the user id (default: a wall-clock stamp).",
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 base timestamp for the demo timeline; defaults to a fixed date for determinism.",
    )
    return parser.parse_args(argv)


def _parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def build_closed_loop(repo: Repository, *, run_id: str, as_of: datetime) -> dict:
    """Build the whole chain for ``run_id`` and return a summary dict.

    Raises ``DemoRunExists`` -- writing nothing -- if ``run_id`` already has
    events in this database.
    """
    user_id = f"usr_{run_id}"
    if repo.list_events(user_id=user_id):
        raise DemoRunExists(
            f"run-id {run_id!r} already has data for user {user_id!r} in this database; "
            "choose a different --run-id or database"
        )

    belief_id = f"{run_id}_bel"
    event_id = f"{run_id}_evt"
    steps: list[str] = []

    # 1. user event
    event = UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=as_of - timedelta(days=20), source="app", **_VERSION_FIELDS,
    )
    repo.insert_event(event)
    steps.append(f"1. event {event_id}")

    # 2. observation
    observation, links = create_observation_from_event(
        event, observation_id=f"{run_id}_obs", category="routine",
        observation_text="User completed an after-work workout.",
        importance=0.6, confidence=0.6, created_at=event.timestamp, **_VERSION_FIELDS,
    )
    repo.insert_observation(observation, links)
    steps.append(f"2. observation {observation.observation_id}")

    # 3. belief evidence (recorded_event leaf)
    proposal = propose_evidence_from_observation_validated(
        observation, links, [event], belief_id=belief_id, direction="support",
        source_type="recorded_event", context_key="fitness", strength=0.9,
        model_version="closed-loop-demo-0.6", belief_type="behavioral_tendency", **_VERSION_FIELDS,
    )
    evidence = authorize_evidence(
        proposal, evidence_id=f"{run_id}_bev", created_at=event.timestamp,
        aggregation_policy_version="evidence-aggregation-0.6",
    )
    repo.insert_evidence(evidence)
    steps.append(f"3. evidence {evidence.evidence_id}")

    # 3b. belief-key canonicalization: a synonym resolves to the canonical key
    canonicalization = authorize_belief_key_canonicalization(
        BeliefKeyCanonicalizationProposal(
            user_id=user_id, belief_type="behavioral_tendency",
            proposed_key="prefers_evening_exercise_sessions", **_VERSION_FIELDS,
        ),
        canonicalization_id=f"{run_id}_can",
        authorized_at=event.timestamp,
    )
    repo.insert_belief_key_canonicalization(canonicalization)
    steps.append(
        f"3b. canonicalization {canonicalization.canonicalization_id} "
        f"{canonicalization.proposed_key!r} -> {canonicalization.canonical_key!r} "
        f"({canonicalization.decision.value})"
    )

    # 4. initial belief (recompute from the ledger)
    initial_belief = recompute_belief(
        belief_id=belief_id, user_id=user_id, belief_type="behavioral_tendency",
        belief_key=_BELIEF_KEY, belief_value=True,
        evidence=repo.list_active_evidence(user_id=user_id, belief_id=belief_id),
        as_of=as_of, first_observed=event.timestamp, **_VERSION_FIELDS,
    )
    repo.save_belief(initial_belief)
    steps.append(
        f"4. initial belief {belief_id} confidence={initial_belief.confidence:.4f} "
        f"status={initial_belief.status.value}"
    )

    # 5 + 6. recommendations and their (followed, successful) outcomes
    recommendation_ids: list[str] = []
    outcome_ids: list[str] = []
    for index in range(1, _TRIALS + 1):
        rec_id = f"{run_id}_rec_{index}"
        recommendation = generate_recommendation(
            recommendation_id=rec_id, user_id=user_id, context_key=_CONTEXT,
            beliefs=[initial_belief], created_at=as_of, goal="be more consistent after work",
        )
        repo.insert_recommendation(recommendation)
        recommendation_ids.append(rec_id)

        outcome = RecommendationOutcome(
            outcome_id=f"{run_id}_out_{index}", recommendation_id=rec_id, followed="followed",
            result="successful", source="app_event",
            created_at=as_of + timedelta(days=index),
            user_feedback="did the after-work slot; felt easier", **_VERSION_FIELDS,
        )
        repo.insert_recommendation_outcome(outcome)
        outcome_ids.append(outcome.outcome_id)
    steps.append(f"5. {_TRIALS} recommendations {recommendation_ids}")
    steps.append(f"6. {_TRIALS} outcomes (followed/successful) {outcome_ids}")

    # 7. outcome-learning signal
    recommendations = repo.list_recommendations(user_id=user_id)
    shown_ids = {r.recommendation_id for r in recommendations}
    outcomes = [o for o in repo.list_recommendation_outcomes() if o.recommendation_id in shown_ids]
    belief_source_events = {
        belief_id: sorted(
            {
                source
                for row in repo.list_active_evidence(user_id=user_id, belief_id=belief_id)
                for source in row.source_event_ids
            }
        )
    }
    signals = analyze_recommendation_outcomes(
        recommendations, outcomes, as_of=as_of, belief_source_events=belief_source_events
    )
    if not signals:
        raise RuntimeError("expected the demo outcomes to produce a learning signal, but none was produced")
    signal = signals[0]
    repo.insert_outcome_learning_signal(signal)
    steps.append(
        f"7. learning signal {signal.signal_id} kind={signal.kind.value} "
        f"trials={signal.trial_count} ({signal.supportive_count} supportive)"
    )

    # 8 + 9. promote the signal's proposal into belief_evidence, then recompute
    promotion = promote_outcome_learning_signal(
        repo, signal_id=signal.signal_id, as_of=as_of + timedelta(days=10),
        persist=True, recompute=True,
    )
    final_belief = repo.get_latest_belief(user_id=user_id, belief_id=belief_id)
    steps.append(f"8. promoted evidence {promotion.inserted_evidence_ids}")
    steps.append(
        f"9. recomputed belief {belief_id} confidence={final_belief.confidence:.4f} "
        f"status={final_belief.status.value} locked={final_belief.locked_until_recompute}"
    )

    return {
        "run_id": run_id,
        "user_id": user_id,
        "belief_id": belief_id,
        "belief_key": _BELIEF_KEY,
        "recommendation_context": _CONTEXT,
        "event_id": event_id,
        "observation_id": observation.observation_id,
        "canonicalization_id": canonicalization.canonicalization_id,
        "recorded_event_evidence_id": evidence.evidence_id,
        "recommendation_ids": recommendation_ids,
        "outcome_ids": outcome_ids,
        "signal_id": signal.signal_id,
        "signal_kind": signal.kind.value,
        "promoted_evidence_ids": promotion.inserted_evidence_ids,
        "initial_confidence": initial_belief.confidence,
        "final_confidence": final_belief.confidence,
        "final_status": final_belief.status.value,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        as_of = _parse_timestamp(args.as_of) if args.as_of else _DEFAULT_AS_OF
    except ValueError as exc:
        print(f"Invalid --as-of {args.as_of!r}: {exc}", file=sys.stderr)
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run%Y%m%d%H%M%S")
    if not run_id.replace("_", "").replace("-", "").isalnum():
        print(f"--run-id {run_id!r} must be alphanumeric (plus _ or -).", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db)
    try:
        try:
            summary = build_closed_loop(repo, run_id=run_id, as_of=as_of)
        except DemoRunExists as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        repo.close()

    print(json.dumps(summary, indent=2, default=str))
    for line in summary["steps"]:
        print(f"  {line}", file=sys.stderr)
    print(
        "\nnext:\n"
        f"  python tools/serve_user_model.py --db {args.db}\n"
        f"  python tools/inspect_user_model.py --db {args.db} --user-id {summary['user_id']} --pretty",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

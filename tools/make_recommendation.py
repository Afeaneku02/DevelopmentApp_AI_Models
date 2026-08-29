#!/usr/bin/env python3
"""Deterministic recommendation MVP CLI for the Better You adaptive user model.

Given a ``--user-id`` and a ``--context-key``, generates one recommendation
from that user's latest beliefs and persists it.

The whole decision path is deterministic and backend-owned:

- beliefs are read with ``Repository.list_latest_beliefs(user_id=...)`` (the
  latest saved row per belief scope, exactly what a recompute leaves behind);
- ``src.recommendations.engine.generate_recommendation`` authorizes them
  through ``authorize_beliefs_for_context`` (the existing context/risk policy
  foundation), so only policy-eligible beliefs can influence the result, and
  ranks candidate actions with the versioned heuristic in
  ``RECOMMENDATION_RANKING``;
- a HIGH-risk or ``requires_manual_review`` context yields a *proposed*
  recommendation with ``review_required=true`` / ``review_status=pending``
  that is NOT auto-issued (blueprint section 6.4);
- the persisted row carries ``risk_tier``, ``risk_resolution_path``, the
  exact ``risk_policy_version`` / ``risk_domain_policy_version`` used,
  ``scoring_version`` / ``policy_version`` / ``ranking_policy_version``,
  ``belief_ids_used``, a full ``candidate_trace`` and ``blocked_beliefs``
  list, and a ``frozen_belief_state`` snapshot of every candidate belief.

No LLM is called. This never recomputes a belief, rewrites belief/evidence
rows, or resolves a manual-review gate.

Run:
    python tools/make_recommendation.py --db events.sqlite3 \\
        --user-id usr_31 --context-key fitness_scheduling \\
        --recommendation-id rec_1 --goal "exercise more consistently"
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendations.engine import DEFAULT_MODEL_VERSION, generate_recommendation  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and persist one deterministic recommendation for a user in a context, "
            "using only beliefs authorized by the recommendation context/risk policy."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument("--context-key", required=True, dest="context_key")
    parser.add_argument("--recommendation-id", required=True, dest="recommendation_id")
    parser.add_argument("--goal", default=None, help="Optional goal string recorded on the recommendation.")
    parser.add_argument(
        "--expected-outcome", default=None, dest="expected_outcome",
        help="Optional expected-outcome string recorded on the recommendation.",
    )
    parser.add_argument(
        "--model-version", default=DEFAULT_MODEL_VERSION, dest="model_version",
        help=f"Deterministic engine version tag (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--created-at", default=None, dest="created_at",
        help="ISO 8601 datetime for the recommendation; defaults to now (UTC).",
    )
    return parser.parse_args(argv)


def _parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.created_at is not None:
        try:
            created_at = _parse_timestamp(args.created_at)
        except ValueError as exc:
            print(f"Invalid --created-at {args.created_at!r}: {exc}", file=sys.stderr)
            return 1
    else:
        created_at = datetime.now(timezone.utc)

    repo = Repository.at_path(args.db)
    try:
        beliefs = repo.list_latest_beliefs(user_id=args.user_id)
        recommendation = generate_recommendation(
            recommendation_id=args.recommendation_id,
            user_id=args.user_id,
            context_key=args.context_key,
            beliefs=beliefs,
            created_at=created_at,
            goal=args.goal,
            expected_outcome=args.expected_outcome,
            model_version=args.model_version,
        )
        try:
            repo.insert_recommendation(recommendation)
        except sqlite3.IntegrityError as exc:
            print(
                f"Duplicate recommendation_id {args.recommendation_id!r} "
                f"(already stored in {args.db!r}): {exc}",
                file=sys.stderr,
            )
            return 1
    finally:
        repo.close()

    print(recommendation.model_dump_json(indent=2))
    if recommendation.review_required:
        print(
            f"note: held for manual review (review_status={recommendation.review_status.value}, "
            f"required_resolution_mode={recommendation.required_resolution_mode.value}); not auto-issued.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Record one recommendation outcome for the Better You adaptive user model.

Appends a single ``recommendation_outcomes`` row (blueprint section 5,
section 12.6) linked to an existing recommendation: what the user did
(``--followed``), how things turned out (``--result``), what the user said
(``--user-feedback``), any measured signal (``--measured-result``), and
where the record came from (``--source``).

This is purely descriptive. It records *what was observed*, never *why*: no
causal claim, no belief link, and it does not update any belief -- mapping
outcomes back to evidence is a separate, later step. It also never mutates
the recommendation it points at; the recommendation's frozen decision state
stays exactly as issued.

Outcomes are append-only. Recording a second outcome for the same
``--recommendation-id`` (a later follow-up, a measured signal arriving after
the user's feedback) is expected; only ``--outcome-id`` must be unique.

Run:
    python tools/add_recommendation_outcome.py --db events.sqlite3 \\
        --outcome-id out_1 --recommendation-id rec_1 \\
        --followed followed --result successful --source app_event \\
        --user-feedback "did the after-work slot, felt easier"
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.enums import OutcomeFollowed, OutcomeResult  # noqa: E402
from src.recommendations.models import RecommendationOutcome  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append one recommendation outcome linked to an existing recommendation. "
            "Descriptive only -- no causal claim, does not update beliefs."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--outcome-id", required=True, dest="outcome_id")
    parser.add_argument("--recommendation-id", required=True, dest="recommendation_id")
    parser.add_argument(
        "--followed", required=True, choices=[m.value for m in OutcomeFollowed],
        help="What the user did relative to the recommendation.",
    )
    parser.add_argument(
        "--result", required=True, choices=[m.value for m in OutcomeResult],
        help="How things turned out (a classification, not a causal claim).",
    )
    parser.add_argument("--source", required=True, help='Where this outcome came from (e.g. "app_event").')
    parser.add_argument("--user-feedback", default=None, dest="user_feedback")
    parser.add_argument("--measured-result", default=None, dest="measured_result")
    parser.add_argument(
        "--observed-at", default=None, dest="observed_at",
        help="ISO 8601 datetime the outcome was observed (optional).",
    )
    parser.add_argument(
        "--created-at", default=None, dest="created_at",
        help="ISO 8601 datetime for the record; defaults to now (UTC).",
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

    try:
        created_at = _parse_timestamp(args.created_at) if args.created_at else datetime.now(timezone.utc)
        observed_at = _parse_timestamp(args.observed_at) if args.observed_at else None
    except ValueError as exc:
        print(f"Invalid timestamp: {exc}", file=sys.stderr)
        return 1

    outcome = RecommendationOutcome(
        outcome_id=args.outcome_id,
        recommendation_id=args.recommendation_id,
        followed=OutcomeFollowed(args.followed),
        result=OutcomeResult(args.result),
        user_feedback=args.user_feedback,
        measured_result=args.measured_result,
        source=args.source,
        observed_at=observed_at,
        created_at=created_at,
        **VERSION_FIELDS,
    )

    repo = Repository.at_path(args.db)
    try:
        try:
            repo.insert_recommendation_outcome(outcome)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except sqlite3.IntegrityError as exc:
            print(
                f"Duplicate outcome_id {args.outcome_id!r} (already stored in {args.db!r}): {exc}",
                file=sys.stderr,
            )
            return 1
    finally:
        repo.close()

    print(outcome.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

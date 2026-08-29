#!/usr/bin/env python3
"""Manual review workflow for outcome-learning signal promotion.

Records one reviewer decision (``approved`` / ``rejected``) on an
outcome-learning signal, making promotion auditable rather than a one-off
CLI run. On ``--decision approved`` you may additionally pass ``--promote``
(and, with it, ``--recompute``) to have the sanctioned
``promote_outcome_learning_signal`` run and its result recorded on the
review. ``--decision rejected`` promotes nothing.

A signal that already has an approved review cannot be re-approved unless
``--allow-duplicate`` is given; when it is, the new review is still stored
and the audit trail (``Repository.list_outcome_learning_signal_reviews``)
shows both. Promotion is independently idempotent, so an allowed duplicate
approval never double-counts evidence.

The decision and reviewer identity always come from these CLI flags -- a
human/backend actor -- never from a model-supplied object.

Run:
    python tools/review_outcome_learning_signal.py --db events.sqlite3 \\
        --signal-id ols-... --review-id rev_1 --reviewer-id alice \\
        --decision approved --notes "repeated positive trials, low risk" --promote --recompute

    python tools/review_outcome_learning_signal.py --db events.sqlite3 \\
        --signal-id ols-... --review-id rev_2 --reviewer-id bob --decision rejected \\
        --notes "outcomes are self-report only"
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.enums import OutcomeLearningReviewDecision  # noqa: E402
from src.recommendations.review import review_outcome_learning_signal  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a manual review of an outcome-learning signal; optionally promote it "
            "(and recompute) when approved. Rejected reviews promote nothing."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--signal-id", required=True, dest="signal_id")
    parser.add_argument("--review-id", required=True, dest="review_id")
    parser.add_argument("--reviewer-id", required=True, dest="reviewer_id")
    parser.add_argument(
        "--decision", required=True, choices=[m.value for m in OutcomeLearningReviewDecision],
    )
    parser.add_argument("--notes", default=None, help="Free-text reviewer notes.")
    parser.add_argument(
        "--promote", action="store_true",
        help="When approved, promote the signal's proposed evidence into belief_evidence.",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="With --promote, recompute affected beliefs; otherwise they are left locked.",
    )
    parser.add_argument(
        "--allow-duplicate", action="store_true", dest="allow_duplicate",
        help="Record this review even if the signal already has an approved review.",
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime for the review (and any promotion); defaults to now (UTC).",
    )
    return parser.parse_args(argv)


def _parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.recompute and not args.promote:
        print("--recompute requires --promote.", file=sys.stderr)
        return 1
    if args.promote and args.decision == OutcomeLearningReviewDecision.REJECTED.value:
        print("--promote cannot be combined with --decision rejected.", file=sys.stderr)
        return 1

    try:
        as_of = _parse_timestamp(args.as_of) if args.as_of else datetime.now(timezone.utc)
    except ValueError as exc:
        print(f"Invalid --as-of {args.as_of!r}: {exc}", file=sys.stderr)
        return 1

    if not Path(args.db).is_file():
        print(f"no such database file: {args.db!r}", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db)
    try:
        try:
            outcome = review_outcome_learning_signal(
                repo,
                review_id=args.review_id,
                signal_id=args.signal_id,
                reviewer_id=args.reviewer_id,
                decision=args.decision,
                as_of=as_of,
                notes=args.notes,
                promote=args.promote,
                recompute=args.recompute,
                allow_duplicate=args.allow_duplicate,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except sqlite3.IntegrityError as exc:
            print(f"Duplicate review_id {args.review_id!r}: {exc}", file=sys.stderr)
            return 1
    finally:
        repo.close()

    if not outcome.stored:
        print(json.dumps({"stored": False, "blocked_reason": outcome.blocked_reason}, indent=2))
        print(f"not recorded: {outcome.blocked_reason}", file=sys.stderr)
        return 1

    payload = {
        "stored": True,
        "review": json.loads(outcome.review.model_dump_json()),
    }
    if outcome.promotion is not None:
        payload["promotion"] = dataclasses.asdict(outcome.promotion)
        payload["promotion"]["inserted_evidence_ids"] = outcome.promotion.inserted_evidence_ids
    print(json.dumps(payload, indent=2, default=str))

    review = outcome.review
    if review.decision is OutcomeLearningReviewDecision.REJECTED:
        print(f"recorded rejection {review.review_id!r}; nothing promoted.", file=sys.stderr)
    elif not review.promotion_requested:
        print(f"recorded approval {review.review_id!r}; --promote not passed, nothing promoted.", file=sys.stderr)
    elif review.recomputed_belief_ids:
        print(
            f"recorded approval {review.review_id!r}; promoted "
            f"{review.promoted_evidence_ids or '0'} evidence row(s); "
            f"recomputed {review.recomputed_belief_ids}.",
            file=sys.stderr,
        )
    elif review.promoted_evidence_ids:
        print(
            f"recorded approval {review.review_id!r}; promoted "
            f"{review.promoted_evidence_ids}; affected belief(s) left locked until recompute.",
            file=sys.stderr,
        )
    else:
        print(
            f"recorded approval {review.review_id!r}; promotion requested but nothing new to "
            f"promote (already promoted or gate not met).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

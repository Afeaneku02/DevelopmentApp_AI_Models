#!/usr/bin/env python3
"""Manual belief-evidence invalidation CLI for the Better You adaptive user
model.

Marks one ``BeliefEvidence`` row inactive via the existing
``Repository.mark_evidence_inactive()`` -- no direct SQL, and no separate
"what does invalidation mean" logic: that method already applies
``invalidate_evidence()`` (the same function used everywhere else in the
pipeline) and, per blueprint section 6.0.2's fail-closed guarantee, locks the
belief's latest saved row (``locked_until_recompute=True``) if one exists, so
a stale confidence is never silently served as current.

This is invalidation only: it never recomputes the belief (that is
``tools/recompute_belief.py``, a separate, deliberate step a caller runs
afterward against the now-inactive evidence), and it never modifies events or
observations.

Run:
    python tools/invalidate_belief_evidence.py --db events.sqlite3 \\
        --evidence-id bev_1 --reason deletion
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.repository import Repository  # noqa: E402

_REASON_CHOICES = ("deletion", "reset", "duplicate_suppression", "policy_invalidation", "manual_review")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mark a single BeliefEvidence row inactive. Invalidation only -- never "
            "recomputes the belief, and never modifies events or observations."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--evidence-id", required=True, dest="evidence_id")
    parser.add_argument("--reason", required=True, choices=_REASON_CHOICES)
    parser.add_argument(
        "--invalidated-at", default=None, dest="invalidated_at",
        help="ISO 8601 datetime for the invalidation; defaults to now (UTC).",
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

    if args.invalidated_at is not None:
        try:
            invalidated_at = _parse_timestamp(args.invalidated_at)
        except ValueError as exc:
            print(f"Invalid --invalidated-at {args.invalidated_at!r}: {exc}", file=sys.stderr)
            return 1
    else:
        invalidated_at = datetime.now(timezone.utc)

    repo = Repository.at_path(args.db)
    try:
        try:
            evidence = repo.mark_evidence_inactive(
                args.evidence_id, reason=args.reason, invalidated_at=invalidated_at,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        latest_belief = repo.get_latest_belief(user_id=evidence.user_id, belief_id=evidence.belief_id)
    finally:
        repo.close()

    output = {
        "evidence": json.loads(evidence.model_dump_json()),
        "latest_belief": json.loads(latest_belief.model_dump_json()) if latest_belief is not None else None,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

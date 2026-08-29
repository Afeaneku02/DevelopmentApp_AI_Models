#!/usr/bin/env python3
"""Promote one outcome-learning signal's proposed evidence into the real
belief_evidence ledger, after deterministic backend checks (blueprint
sections 6.1.2, 12.6).

This is a manual, one-signal-at-a-time authorization step -- not a learner.
It loads the signal named by ``--signal-id`` and runs the deterministic
gate in ``src.recommendations.promotion``: ``causal_claim`` must be false,
``kind`` must be ``support`` or ``weak_contradiction``, ``proposed_evidence``
must exist, and the stored trial breakdown must still meet policy.

Each surviving proposal is re-provenanced from the target belief's current
active evidence leaves, then authorized via ``authorize_evidence`` with the
signal's ``independence_group`` (so repeated analyses of the same pattern
share one independence group and cannot corroborate each other) and
``source_type=repeated_pattern_summary``. A proposal whose belief already
has an equivalent active promoted row, or has no leaf evidence to trace to,
is skipped.

Nothing is written unless ``--persist``. When evidence is added but
``--recompute`` is not passed, the affected beliefs are locked
(``locked_until_recompute=true``). ``--recompute`` instead runs the real
recompute path and saves fresh beliefs.

Run:
    python tools/promote_outcome_learning_signal.py --db events.sqlite3 --signal-id ols-...          # dry run
    python tools/promote_outcome_learning_signal.py --db events.sqlite3 --signal-id ols-... --persist
    python tools/promote_outcome_learning_signal.py --db events.sqlite3 --signal-id ols-... --persist --recompute
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendations.promotion import promote_outcome_learning_signal  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote one outcome-learning signal's proposed evidence into belief_evidence "
            "after deterministic checks. Dry run unless --persist."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--signal-id", required=True, dest="signal_id")
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime for the authorized evidence / recompute; defaults to now (UTC).",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Write the authorized belief_evidence rows (default: dry run, nothing written).",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="After persisting, recompute affected beliefs; without it they are left locked.",
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

    if args.recompute and not args.persist:
        print("--recompute requires --persist (there is nothing to recompute in a dry run).", file=sys.stderr)
        return 1

    try:
        as_of = _parse_timestamp(args.as_of) if args.as_of else datetime.now(timezone.utc)
    except ValueError as exc:
        print(f"Invalid --as-of {args.as_of!r}: {exc}", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db) if args.persist else Repository.readonly_at_path(args.db)
    try:
        try:
            result = promote_outcome_learning_signal(
                repo, signal_id=args.signal_id, as_of=as_of,
                persist=args.persist, recompute=args.recompute,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        repo.close()

    payload = dataclasses.asdict(result)
    payload["inserted_evidence_ids"] = result.inserted_evidence_ids
    print(json.dumps(payload, indent=2, default=str))

    if not result.authorized:
        print(f"not promoted: {result.rejected_reason}", file=sys.stderr)
        return 1
    if not args.persist:
        print("dry run: nothing was written.", file=sys.stderr)
    elif result.recomputed:
        print(
            f"persisted {len(result.inserted_evidence_ids)} evidence row(s); "
            f"recomputed {len(result.recomputed_beliefs)} belief(s).",
            file=sys.stderr,
        )
    elif result.inserted_evidence_ids:
        print(
            f"persisted {len(result.inserted_evidence_ids)} evidence row(s); "
            f"locked {len(result.locked_belief_ids)} belief(s) until recompute.",
            file=sys.stderr,
        )
    else:
        print("nothing new to persist (all proposals skipped).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

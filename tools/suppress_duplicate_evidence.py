#!/usr/bin/env python3
"""Manual duplicate-evidence suppression CLI for the Better You adaptive
user model.

Finds and marks duplicate ``BeliefEvidence`` rows for one (user_id,
belief_id) scope via the existing ``Repository.suppress_duplicate_evidence()``
-- no direct SQL, and no separate "what counts as a duplicate" logic: that
method already applies ``find_duplicate_evidence()``/
``suppress_duplicate_evidence()`` (backend-owned, never an LLM/proposal
decision -- see ``src/beliefs/models.py``) and, per blueprint section
6.0.2's fail-closed guarantee, locks the belief's latest saved row
(``locked_until_recompute=True``) if any row was newly suppressed, so a
stale, evidence-inflated confidence is never silently served as current.

This never deletes evidence -- a suppressed row remains in the ledger for
audit, only ``is_duplicate_suppressed``/``suppression_reason`` change -- and
it never recomputes the belief (that is ``tools/recompute_belief.py``, a
separate, deliberate step a caller runs afterward against the now-reduced
active evidence), and never modifies events or observations.

Run:
    python tools/suppress_duplicate_evidence.py --db events.sqlite3 \\
        --user-id usr_17 --belief-id bel_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beliefs.models import DEFAULT_DUPLICATE_SUPPRESSION_REASON  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and mark duplicate BeliefEvidence rows for one (user_id, belief_id) scope. "
            "Suppression only -- never deletes evidence, never recomputes the belief, and "
            "never modifies events or observations."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument("--belief-id", required=True, dest="belief_id")
    parser.add_argument(
        "--reason", default=DEFAULT_DUPLICATE_SUPPRESSION_REASON,
        help=f"Suppression reason to record (default: {DEFAULT_DUPLICATE_SUPPRESSION_REASON!r}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    repo = Repository.at_path(args.db)
    try:
        newly_suppressed = repo.suppress_duplicate_evidence(
            user_id=args.user_id, belief_id=args.belief_id, reason=args.reason,
        )
        latest_belief = repo.get_latest_belief(user_id=args.user_id, belief_id=args.belief_id)
    finally:
        repo.close()

    output = {
        "suppressed_evidence": [json.loads(row.model_dump_json()) for row in newly_suppressed],
        "latest_belief": json.loads(latest_belief.model_dump_json()) if latest_belief is not None else None,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

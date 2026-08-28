#!/usr/bin/env python3
"""Manual belief-key canonicalization CLI for the Better You adaptive user
model.

Proposes and authorizes a single belief-key canonicalization decision via
the existing ``src.beliefs.canonicalization.authorize_belief_key_canonicalization()``
-- no separate "is this an alias" logic here: that function is the only
sanctioned path from a proposal to a persisted decision, and it always
checks the canonical belief_key registry (backend policy data) before ever
considering what this CLI's own flags suggest. Saves the decision via
``Repository.insert_belief_key_canonicalization()`` (an append-only audit
log; a re-resolution of the same proposed_key never overwrites an earlier
decision) and prints it as pretty JSON.

The printed ``canonical_key`` is what a caller should actually pass to
``tools/recompute_belief.py --belief-key`` next -- using the raw
``--proposed-key`` instead would defeat the whole point of canonicalization
by letting a semantically duplicate spelling fragment evidence across a
second ``UserBelief`` row.

This never rewrites or deletes any existing belief/evidence row, and it
never recomputes a belief itself (that is ``tools/recompute_belief.py``, a
separate, deliberate step a caller runs afterward with the authorized
canonical_key).

Run:
    python tools/resolve_belief_key.py --db events.sqlite3 \\
        --canonicalization-id can_1 --user-id usr_17 \\
        --belief-type behavioral_tendency \\
        --proposed-key prefers_evening_exercise_sessions
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beliefs.canonicalization import (  # noqa: E402
    BeliefKeyCanonicalizationProposal,
    authorize_belief_key_canonicalization,
)
from src.common.enums import BeliefType, CanonicalizationDecision  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propose and authorize a single belief-key canonicalization decision, then save it. "
            "Never rewrites/deletes existing belief or evidence rows, and never recomputes a belief."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--canonicalization-id", required=True, dest="canonicalization_id")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--belief-type", required=True, dest="belief_type",
        choices=[member.value for member in BeliefType],
    )
    parser.add_argument("--proposed-key", required=True, dest="proposed_key")
    parser.add_argument(
        "--proposed-canonical-key", default=None, dest="proposed_canonical_key",
        help="The LLM/extractor's own guess at what canonical key this should alias to, if any.",
    )
    parser.add_argument(
        "--proposed-decision", default=CanonicalizationDecision.KEEP_SEPARATE.value, dest="proposed_decision",
        choices=[member.value for member in CanonicalizationDecision],
        help="The LLM/extractor's own suggested decision (default: keep_separate, the safest one).",
    )
    parser.add_argument("--reason", default=None, help="The LLM/extractor's own stated reason, if any.")
    parser.add_argument(
        "--authorized-at", default=None, dest="authorized_at",
        help="ISO 8601 datetime for the authorization; defaults to now (UTC).",
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

    if args.authorized_at is not None:
        try:
            authorized_at = _parse_timestamp(args.authorized_at)
        except ValueError as exc:
            print(f"Invalid --authorized-at {args.authorized_at!r}: {exc}", file=sys.stderr)
            return 1
    else:
        authorized_at = datetime.now(timezone.utc)

    proposal = BeliefKeyCanonicalizationProposal(
        user_id=args.user_id,
        belief_type=BeliefType(args.belief_type),
        proposed_key=args.proposed_key,
        proposed_canonical_key=args.proposed_canonical_key,
        proposed_decision=CanonicalizationDecision(args.proposed_decision),
        reason=args.reason,
        **VERSION_FIELDS,
    )

    decision = authorize_belief_key_canonicalization(
        proposal, canonicalization_id=args.canonicalization_id, authorized_at=authorized_at,
    )

    repo = Repository.at_path(args.db)
    try:
        try:
            repo.insert_belief_key_canonicalization(decision)
        except sqlite3.IntegrityError as exc:
            print(
                f"Duplicate canonicalization_id {args.canonicalization_id!r} "
                f"(already stored in {args.db!r}): {exc}",
                file=sys.stderr,
            )
            return 1
    finally:
        repo.close()

    print(decision.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

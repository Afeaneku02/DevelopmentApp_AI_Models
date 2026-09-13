#!/usr/bin/env python3
"""Process one user's stored ``user_observations`` into conservative
``belief_evidence`` (blueprint section 6, create-belief-evidence's runtime
slice).

This is the CLI for
``src.beliefs.process_observations.process_user_observations`` -- a
deterministic, allowlisted mapping from a handful of low-risk observation
categories to fixed belief mappings, using only the sanctioned
``propose_evidence_from_observation_validated()`` / ``authorize_evidence()``
path. See that module's docstring for the full scope and guarantees. In
short: no LLM, no raw/private text, provenance always traces to real
``observation_events``/``source_event_ids``, and idempotent -- rerunning
against the same (or a superset of) observations creates no duplicate
evidence.

Nothing is written unless ``--persist``, and no belief is recomputed unless
``--recompute`` (which requires ``--persist``) -- matching
``tools/promote_outcome_learning_signal.py``'s exact conventions. Without
``--recompute``, a belief that received new evidence is left
``locked_until_recompute`` rather than silently kept serving a now-stale
cached confidence.

Run:
    # Every unprocessed observation for a user (dry run -- nothing written):
    python tools/process_observations_to_evidence.py --db canonical.sqlite3 --user-id usr_17

    # Persist the evidence, leaving affected beliefs locked:
    python tools/process_observations_to_evidence.py --db canonical.sqlite3 --user-id usr_17 --persist

    # Persist and also recompute affected beliefs:
    python tools/process_observations_to_evidence.py --db canonical.sqlite3 --user-id usr_17 \\
        --persist --recompute

    # Only specific observations:
    python tools/process_observations_to_evidence.py --db canonical.sqlite3 --user-id usr_17 \\
        --observation-id obs_1 --observation-id obs_2 --persist
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

from src.beliefs.process_observations import process_user_observations  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

_CREATED_ACTIONS = {"created", "would_create"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn one user's stored user_observations into conservative belief_evidence, "
            "using only the sanctioned propose/authorize path. Dry run unless --persist."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--observation-id", action="append", default=None, dest="observation_id",
        help=(
            "Restrict processing to this observation (repeatable). Default: every "
            "stored observation for --user-id that doesn't already have evidence."
        ),
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime stamped on any created evidence / recompute; defaults to now (UTC).",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Write the new belief_evidence rows (default: dry run, nothing written).",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="After persisting, recompute beliefs that received new evidence; without it they are left locked.",
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

    # Reject a missing database before opening it, in every mode: a typoed
    # --db must never silently create a new empty database (which
    # ``Repository.at_path`` would do under --persist) or produce an opaque
    # traceback (which ``readonly_at_path`` would raise for a dry run).
    if not Path(args.db).is_file():
        print(f"no such database file: {args.db!r}", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db) if args.persist else Repository.readonly_at_path(args.db)
    try:
        result = process_user_observations(
            repo, user_id=args.user_id, observation_ids=args.observation_id,
            as_of=as_of, persist=args.persist, recompute=args.recompute,
        )
    finally:
        repo.close()

    payload = dataclasses.asdict(result)
    payload["created_evidence_ids"] = result.created_evidence_ids
    print(json.dumps(payload, indent=2, default=str))

    created = [o for o in result.outcomes if o.action in _CREATED_ACTIONS]
    skipped = [o for o in result.outcomes if o.action not in _CREATED_ACTIONS]
    if not args.persist:
        print(
            f"dry run: nothing was written ({len(created)} evidence row(s) would be created; "
            f"{len(skipped)} observation(s) skipped).",
            file=sys.stderr,
        )
    elif result.recomputed:
        print(
            f"persisted {len(created)} evidence row(s); {len(skipped)} observation(s) skipped; "
            f"recomputed {len(result.recomputed_beliefs)} belief(s).",
            file=sys.stderr,
        )
    elif result.locked_belief_ids:
        print(
            f"persisted {len(created)} evidence row(s); {len(skipped)} observation(s) skipped; "
            f"locked {len(result.locked_belief_ids)} belief(s) until recompute.",
            file=sys.stderr,
        )
    else:
        print(
            f"persisted {len(created)} evidence row(s); {len(skipped)} observation(s) skipped.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

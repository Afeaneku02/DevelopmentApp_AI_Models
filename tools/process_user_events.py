#!/usr/bin/env python3
"""Process one user's stored ``user_events`` into conservative
``user_observations`` (blueprint section 5/6.2's structural half).

This is the CLI for ``src.observations.process_events.process_user_events``
-- a deterministic, allowlisted mapping from a handful of low-risk Better
You event types to fixed observation templates. See that module's docstring
for the full scope and guarantees. In short: no LLM, no free text, no
belief_evidence or beliefs, and idempotent -- rerunning against the same (or
a superset of) events creates no duplicate observations.

Nothing is written unless ``--persist``, matching
``tools/promote_outcome_learning_signal.py``'s convention: a dry run always
prints exactly what *would* be created first, so an operator can review it
before committing.

Run:
    # Every unprocessed event for a user (dry run -- nothing written):
    python tools/process_user_events.py --db canonical.sqlite3 --user-id usr_17

    # Same, actually writing the observations:
    python tools/process_user_events.py --db canonical.sqlite3 --user-id usr_17 --persist

    # Only specific events:
    python tools/process_user_events.py --db canonical.sqlite3 --user-id usr_17 \\
        --event-id evt_1 --event-id evt_2 --persist
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

from src.observations.process_events import process_user_events  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

_CREATED_ACTIONS = {"created", "would_create"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn one user's stored user_events into conservative user_observations. "
            "Creates no belief_evidence or beliefs. Dry run unless --persist."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--event-id", action="append", default=None, dest="event_id",
        help=(
            "Restrict processing to this event (repeatable). Default: every "
            "stored event for --user-id that isn't already covered by an "
            "observation."
        ),
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime stamped on any created observation; defaults to now (UTC).",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Write the new observations (default: dry run, nothing written).",
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
        result = process_user_events(
            repo, user_id=args.user_id, event_ids=args.event_id, as_of=as_of, persist=args.persist,
        )
    finally:
        repo.close()

    payload = dataclasses.asdict(result)
    payload["created_observation_ids"] = result.created_observation_ids
    print(json.dumps(payload, indent=2, default=str))

    created = [o for o in result.outcomes if o.action in _CREATED_ACTIONS]
    skipped = [o for o in result.outcomes if o.action not in _CREATED_ACTIONS]
    if not args.persist:
        print(
            f"dry run: nothing was written ({len(created)} observation(s) would be created; "
            f"{len(skipped)} event(s) skipped).",
            file=sys.stderr,
        )
    else:
        print(
            f"persisted {len(created)} observation(s); {len(skipped)} event(s) skipped.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

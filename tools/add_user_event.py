#!/usr/bin/env python3
"""Manual event intake CLI for the Better You adaptive user model.

Inserts exactly one ``UserEvent`` into a ``Repository.at_path(db)`` SQLite
database, validating it through the same ``src.events.models.UserEvent``
model used everywhere else in this project -- no separate ad hoc validation.

This is event intake only (blueprint section 5, the append-only
ground-truth table): it never creates observations, belief_evidence, or
beliefs. Those are separate, already-built pipeline steps
(``src.observations.create_observation``, ``src.beliefs.propose_evidence``,
``src.beliefs.recompute``) that a caller runs afterward against the events
this CLI has stored.

Run:
    python tools/add_user_event.py --db events.sqlite3 \\
        --user-id usr_17 --event-id evt_1 --event-type goal_completed \\
        --source app --structured-data '{"goal": "workout"}'

``--structured-data-file PATH`` reads the JSON object from a file instead of
a command-line string -- useful on shells (e.g. PowerShell) where quoting a
JSON literal inline is error-prone. It is mutually exclusive with
``--structured-data``.
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

from pydantic import ValidationError  # noqa: E402

from src.events.models import UserEvent  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

# Matches the version literals used elsewhere in this project (e.g.
# tools/demo_user_model.py) -- reusing the current tags, not inventing a new
# versioning scheme for event-intake records.
VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert a single UserEvent into a Repository. Event intake only "
            "-- creates no observations, evidence, or beliefs."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument("--event-id", required=True, dest="event_id")
    parser.add_argument("--event-type", required=True, dest="event_type")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--timestamp", default=None,
        help="ISO 8601 datetime for the event; defaults to now (UTC).",
    )
    parser.add_argument(
        "--structured-data", default=None, dest="structured_data",
        help="JSON object string, e.g. '{\"goal\": \"workout\"}'.",
    )
    parser.add_argument(
        "--structured-data-file", default=None, dest="structured_data_file",
        help=(
            "Path to a file containing a JSON object; an alternative to "
            "--structured-data for payloads a shell would mangle. Mutually "
            "exclusive with --structured-data."
        ),
    )
    parser.add_argument("--raw-content", default=None, dest="raw_content")
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

    if args.structured_data is not None and args.structured_data_file is not None:
        print(
            "Cannot use both --structured-data and --structured-data-file in the same command.",
            file=sys.stderr,
        )
        return 1

    structured_data_text: str | None = None
    structured_data_source: str = "--structured-data"
    if args.structured_data is not None:
        structured_data_text = args.structured_data
    elif args.structured_data_file is not None:
        structured_data_source = f"--structured-data-file {args.structured_data_file!r}"
        try:
            structured_data_text = Path(args.structured_data_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Cannot read --structured-data-file {args.structured_data_file!r}: {exc}", file=sys.stderr)
            return 1

    if structured_data_text is not None:
        try:
            structured_data = json.loads(structured_data_text)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON in {structured_data_source}: {exc}", file=sys.stderr)
            return 1
    else:
        structured_data = None

    if args.timestamp is not None:
        try:
            timestamp = _parse_timestamp(args.timestamp)
        except ValueError as exc:
            print(f"Invalid --timestamp {args.timestamp!r}: {exc}", file=sys.stderr)
            return 1
    else:
        timestamp = datetime.now(timezone.utc)

    try:
        event = UserEvent(
            event_id=args.event_id,
            user_id=args.user_id,
            event_type=args.event_type,
            timestamp=timestamp,
            raw_content=args.raw_content,
            structured_data=structured_data,
            source=args.source,
            **VERSION_FIELDS,
        )
    except ValidationError as exc:
        print(f"Event validation failed:\n{exc}", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db)
    try:
        repo.insert_event(event)
    except sqlite3.IntegrityError as exc:
        print(f"Duplicate event_id {args.event_id!r} (already stored in {args.db!r}): {exc}", file=sys.stderr)
        return 1
    finally:
        repo.close()

    print(event.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Manual observation intake CLI for the Better You adaptive user model.

Inserts exactly one ``UserObservation`` plus its ``observation_events`` links
into a ``Repository.at_path(db)`` SQLite database, validating both through
the same ``src.observations.models`` classes used everywhere else in this
project -- no separate ad hoc validation, and no second "does this link
provenance check out" implementation: ``Repository.insert_observation()``
already re-checks referential integrity
(``src.common.provenance.validate_observation_provenance()``) against the
actual stored events, so this CLI's own upfront event lookup exists only to
give a clearer, per-event-id error message before that write is attempted.

This is observation intake only (blueprint section 5): it never creates
belief_evidence or beliefs. Those are separate, already-built pipeline steps
(``src.beliefs.propose_evidence``, ``src.beliefs.recompute``) that a caller
runs afterward against the observations this CLI has stored.

Run:
    python tools/add_user_observation.py --db events.sqlite3 \\
        --observation-id obs_1 --user-id usr_17 --event-id evt_1 \\
        --category routine --observation "User completed a workout." \\
        --importance 0.6 --confidence 0.6

With more than one --event-id, pick which one is primary explicitly:
    python tools/add_user_observation.py --db events.sqlite3 \\
        --observation-id obs_2 --user-id usr_17 \\
        --event-id evt_1 --event-id evt_2 --primary-event-id evt_1 \\
        --category routine --observation "..." --importance 0.6 --confidence 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from src.common.enums import LinkRole  # noqa: E402
from src.observations.models import ObservationEvent, UserObservation  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

# Matches the version literals used elsewhere in this project (e.g.
# tools/demo_user_model.py, tools/add_user_event.py) -- reusing the current
# tags, not inventing a new versioning scheme for observation-intake records.
VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert a single UserObservation and its observation_events "
            "links into a Repository. Observation intake only -- creates "
            "no belief evidence or beliefs."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--observation-id", required=True, dest="observation_id")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--event-id", action="append", default=[], dest="event_id",
        help="Event this observation links to. Repeatable; at least one is required.",
    )
    parser.add_argument(
        "--primary-event-id", default=None, dest="primary_event_id",
        help=(
            "Which --event-id is the primary link. Required when more than "
            "one --event-id is given; optional when exactly one is given "
            "(that one is used as primary)."
        ),
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--observation", required=True, help="The observation text.")
    parser.add_argument("--importance", required=True, type=float)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument(
        "--created-at", default=None, dest="created_at",
        help="ISO 8601 datetime for the observation; defaults to now (UTC).",
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


def _resolve_primary_event_id(event_ids: list[str], primary_event_id: str | None) -> str:
    """Raises ``ValueError`` with a message ready to print to stderr if the
    primary link cannot be determined unambiguously.
    """
    if primary_event_id is not None:
        if primary_event_id not in event_ids:
            raise ValueError(
                f"--primary-event-id {primary_event_id!r} is not one of the --event-id values {event_ids!r}."
            )
        return primary_event_id
    if len(event_ids) == 1:
        return event_ids[0]
    raise ValueError(
        "--primary-event-id is required when more than one --event-id is given "
        f"(got {event_ids!r})."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.event_id:
        print("At least one --event-id is required.", file=sys.stderr)
        return 1

    duplicate_event_ids = sorted({
        event_id for event_id in set(args.event_id) if args.event_id.count(event_id) > 1
    })
    if duplicate_event_ids:
        print(
            "Duplicate --event-id value(s): " + ", ".join(repr(event_id) for event_id in duplicate_event_ids)
            + ". Each --event-id must be given at most once.",
            file=sys.stderr,
        )
        return 1

    try:
        primary_event_id = _resolve_primary_event_id(args.event_id, args.primary_event_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

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
        errors: list[str] = []
        for event_id in args.event_id:  # duplicates already rejected above
            event = repo.get_event(event_id)
            if event is None:
                errors.append(f"Unknown event_id {event_id!r}: not found in {args.db!r}.")
            elif event.user_id != args.user_id:
                errors.append(
                    f"event_id {event_id!r} belongs to a different user "
                    f"(event.user_id={event.user_id!r} != --user-id={args.user_id!r})."
                )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1

        try:
            observation = UserObservation(
                observation_id=args.observation_id,
                user_id=args.user_id,
                category=args.category,
                observation=args.observation,
                importance=args.importance,
                confidence=args.confidence,
                created_at=created_at,
                **VERSION_FIELDS,
            )
            links = [
                ObservationEvent(
                    observation_id=args.observation_id,
                    event_id=event_id,
                    link_role=LinkRole.PRIMARY if event_id == primary_event_id else LinkRole.SUPPORTING,
                    created_at=created_at,
                    **VERSION_FIELDS,
                )
                for event_id in args.event_id
            ]
        except ValidationError as exc:
            print(f"Observation validation failed:\n{exc}", file=sys.stderr)
            return 1

        try:
            repo.insert_observation(observation, links)
        except ValueError as exc:
            print(f"Cannot insert observation {args.observation_id!r}: {exc}", file=sys.stderr)
            return 1
    finally:
        repo.close()

    output = {
        "observation": json.loads(observation.model_dump_json()),
        "observation_events": [json.loads(link.model_dump_json()) for link in links],
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

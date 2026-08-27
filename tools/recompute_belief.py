#!/usr/bin/env python3
"""Manual belief-recompute CLI for the Better You adaptive user model.

Loads the active belief_evidence ledger for one belief scope from
``Repository.at_path(db)`` (``repo.list_active_evidence()``, the same
``is_active and not is_duplicate_suppressed`` definition used everywhere else
in this project), recomputes a ``UserBelief`` from it via the existing
``src.beliefs.recompute.recompute_belief()`` -- no separate scoring formula
-- and saves the result via ``repo.save_belief()``.

This is belief recompute only (blueprint section 6.0.1-6.0.2): it never
creates or modifies events, observations, or belief_evidence. Those are
separate, already-built intake steps (``tools/add_user_event.py``,
``tools/add_user_observation.py``, ``tools/add_belief_evidence.py``) that a
caller runs beforehand to populate what this CLI reads.

Run:
    python tools/recompute_belief.py --db events.sqlite3 \\
        --belief-id bel_1 --user-id usr_17 --belief-type behavioral_tendency \\
        --belief-key higher_adherence_after_work --belief-value-json true

After invalidating all of a belief's evidence (e.g. via a deletion request),
recompute with the matching reason so the zero-active-evidence branch is
reached deliberately rather than by accident (see
``src.beliefs.recompute.recompute_belief``'s own docstring):

    python tools/recompute_belief.py --db events.sqlite3 \\
        --belief-id bel_1 --user-id usr_17 --belief-type behavioral_tendency \\
        --belief-key higher_adherence_after_work --belief-value-json true \\
        --recompute-reason deletion --allow-no-evidence

``--belief-value-json-file PATH`` reads belief_value's JSON from a file
instead of a command-line string -- useful on shells where quoting a JSON
literal inline is error-prone. Unlike observation intake's
--structured-data-file, the file may hold any JSON value (bool, string,
number, object, array, or null), not just an object, since belief_value
itself is untyped. It is mutually exclusive with --belief-value-json, and
exactly one of the two is required.
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

from src.beliefs.recompute import recompute_belief  # noqa: E402
from src.common.enums import BeliefType  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

# Matches the version literals used elsewhere in this project (e.g.
# tools/demo_user_model.py, tools/add_user_event.py, tools/add_user_observation.py,
# tools/add_belief_evidence.py) -- reusing the current tags, not inventing a
# new versioning scheme.
VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute and save a single UserBelief from its stored active belief_evidence. "
            "Recompute only -- creates or modifies no events, observations, or evidence."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--belief-id", required=True, dest="belief_id")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--belief-type", required=True, dest="belief_type",
        choices=[member.value for member in BeliefType],
    )
    parser.add_argument("--belief-key", required=True, dest="belief_key")
    belief_value_group = parser.add_mutually_exclusive_group(required=True)
    belief_value_group.add_argument(
        "--belief-value-json", default=None, dest="belief_value_json",
        help="JSON value for belief_value, e.g. 'true' or '\"some string\"'.",
    )
    belief_value_group.add_argument(
        "--belief-value-json-file", default=None, dest="belief_value_json_file",
        help=(
            "Path to a file containing a JSON value for belief_value (any JSON type: "
            "bool, string, number, object, array, or null). Mutually exclusive with "
            "--belief-value-json."
        ),
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime to recompute as of; defaults to now (UTC).",
    )
    parser.add_argument(
        "--first-observed", default=None, dest="first_observed",
        help=(
            "ISO 8601 datetime; if omitted, inferred from the earliest active evidence's "
            "observed_at, or from --as-of when there is no active evidence."
        ),
    )
    parser.add_argument("--recompute-reason", default=None, dest="recompute_reason")
    parser.add_argument(
        "--allow-no-evidence", action="store_true", dest="allow_no_evidence",
        help="Recompute (to confidence 0.0) even when there is no active evidence.",
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

    if args.belief_value_json is not None:
        belief_value_text = args.belief_value_json
        belief_value_source = "--belief-value-json"
    else:
        belief_value_source = f"--belief-value-json-file {args.belief_value_json_file!r}"
        try:
            belief_value_text = Path(args.belief_value_json_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"Cannot read --belief-value-json-file {args.belief_value_json_file!r}: {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        belief_value = json.loads(belief_value_text)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {belief_value_source}: {exc}", file=sys.stderr)
        return 1

    if args.as_of is not None:
        try:
            as_of = _parse_timestamp(args.as_of)
        except ValueError as exc:
            print(f"Invalid --as-of {args.as_of!r}: {exc}", file=sys.stderr)
            return 1
    else:
        as_of = datetime.now(timezone.utc)

    if args.first_observed is not None:
        try:
            first_observed = _parse_timestamp(args.first_observed)
        except ValueError as exc:
            print(f"Invalid --first-observed {args.first_observed!r}: {exc}", file=sys.stderr)
            return 1
    else:
        first_observed = None  # resolved below once active evidence is known

    repo = Repository.at_path(args.db)
    try:
        active = repo.list_active_evidence(user_id=args.user_id, belief_id=args.belief_id)

        if not active and not args.allow_no_evidence:
            print(
                f"No active evidence found for belief_id={args.belief_id!r}, user_id={args.user_id!r}; "
                "pass --allow-no-evidence to recompute anyway (this will produce confidence 0.0).",
                file=sys.stderr,
            )
            return 1

        if first_observed is None:
            first_observed = min(row.observed_at for row in active) if active else as_of

        try:
            belief = recompute_belief(
                belief_id=args.belief_id,
                user_id=args.user_id,
                belief_type=BeliefType(args.belief_type),
                belief_key=args.belief_key,
                belief_value=belief_value,
                evidence=active,
                as_of=as_of,
                first_observed=first_observed,
                recompute_reason=args.recompute_reason,
                **VERSION_FIELDS,
            )
        except ValidationError as exc:
            print(f"Belief validation failed:\n{exc}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        repo.save_belief(belief)
    finally:
        repo.close()

    print(belief.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

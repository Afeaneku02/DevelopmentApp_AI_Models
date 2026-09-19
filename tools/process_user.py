#!/usr/bin/env python3
"""Run one user through the whole existing adaptive-user-model pipeline in a
single command --

    unprocessed user_events -> user_observations -> belief_evidence
    -> belief recomputation

This is the CLI for
``src.orchestration.process_user_pipeline.process_user_full_pipeline`` --
a coordinator, not a new drafting mechanism. It reuses
``tools/process_user_events.py``'s and
``tools/process_observations_to_evidence.py``'s own underlying functions
exactly (``process_user_events`` / ``process_user_observations``); running
those two CLIs back to back for one user is equivalent to one run of this
one, just scoped and reported together. See that module's docstring for the
full scope and guarantees: idempotent reruns, one record's failure isolated
from every other record, and a mandatory (not caller-opt-in) belief
recompute for every belief that receives new evidence this run.

Nothing is written to ``--db`` unless ``--persist``. Without it, this is a
dry run that accurately simulates the complete chain -- including a stage-
one observation this same run would create feeding stage two, and the
belief recomputation that would follow -- against a disposable in-memory
clone of ``--db``'s data; see ``process_user_full_pipeline``'s module
docstring for how and why. Every count and previewed belief in a dry run's
output reflects what ``--persist`` would actually do right now, not a
partial guess.

Exit codes: ``0`` every considered record succeeded or was safely skipped
(the normal case for a dry run too); ``1`` this command could not even
start (``--db``/``--as-of`` itself is invalid); ``2`` the run started and
``result.failures`` is nonempty -- at least one record hit an unexpected
error (see ``RecordFailure``) even though every other record still
succeeded or was safely skipped. Automation that only checks for exit ``0``
would otherwise mistake a partially failed run for a fully successful one.

Run:
    # Dry run -- nothing written to --db, full chain simulated:
    python tools/process_user.py --db canonical.sqlite3 --user-id usr_17

    # Persist observations, evidence, and recomputed beliefs:
    python tools/process_user.py --db canonical.sqlite3 --user-id usr_17 --persist
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

from src.orchestration.process_user_pipeline import process_user_full_pipeline  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one user's stored, unprocessed user_events through the whole existing pipeline "
            "(-> observations -> belief_evidence -> belief recomputation) in one command. "
            "Dry run unless --persist."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", required=True, dest="user_id")
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime stamped on any created observation/evidence/recompute; defaults to now (UTC).",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Write the new observations, evidence, and recomputed beliefs (default: dry run, nothing written).",
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

    # Reject a missing database before opening it, in every mode -- see
    # tools/process_observations_to_evidence.py for why: a typoed --db must
    # never silently create a new empty database under --persist, or produce
    # an opaque traceback under a dry run.
    if not Path(args.db).is_file():
        print(f"no such database file: {args.db!r}", file=sys.stderr)
        return 1

    repo = Repository.at_path(args.db) if args.persist else Repository.readonly_at_path(args.db)
    try:
        result = process_user_full_pipeline(
            repo, user_id=args.user_id, as_of=as_of, persist=args.persist,
        )
    finally:
        repo.close()

    payload = dataclasses.asdict(result)
    payload["counts"] = result.counts()
    print(json.dumps(payload, indent=2, default=str))

    counts = result.counts()
    if not args.persist:
        print(
            f"dry run: nothing was written to {args.db!r} ({counts['created']} record(s) would be created; "
            f"{counts['skipped']} skipped; {counts['failed']} failed; "
            f"would recompute {counts['recomputed']} belief(s)).",
            file=sys.stderr,
        )
    else:
        print(
            f"persisted {counts['created']} record(s); {counts['skipped']} skipped; "
            f"{counts['failed']} failed; recomputed {counts['recomputed']} belief(s).",
            file=sys.stderr,
        )
    for failure in result.failures:
        print(f"  failed: {failure.stage} {failure.record_id}: {failure.error}", file=sys.stderr)
    # A nonempty result.failures means at least one record hit an unexpected
    # error (see ProcessUserPipelineResult/RecordFailure) -- every other
    # record in the run still succeeded or was safely skipped, and stdout's
    # JSON payload already reports exactly which records and why. But this
    # command's exit code is what automation actually branches on, and 0
    # historically means "nothing went wrong"; returning 0 here would let a
    # scripted caller mistake a partially failed run for a fully successful
    # one. Exit 2 (distinct from argument/IO errors' exit 1 above) so a
    # caller can tell "this run itself found a problem" apart from "this run
    # couldn't even start."
    return 2 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Outcome-learning MVP CLI for the Better You adaptive user model.

Reads persisted recommendations and their outcomes, groups them by
recommendation pattern (context + beliefs used), and -- only where there
have been enough repeated trials -- prints a conservative
``OutcomeLearningSignal`` per pattern: weak ``repeated_pattern_summary``
belief_evidence *proposals* for or against the beliefs the recommendation
used, plus the full trial breakdown and provenance
(``recommendation_ids`` / ``outcome_ids`` / ``independence_group``).

This never claims causality (the evidence is about the belief, framed as
correlation), never mutates a belief, and never recomputes.

By default it only prints. Pass ``--persist`` to append the signals to the
``outcome_learning_signals`` table; without it the database is opened
strictly read-only, so it cannot be modified. Promoting a proposal into the
real belief_evidence ledger is a separate, later, backend-authorized step
that this tool does not perform.

Run:
    python tools/learn_from_recommendation_outcomes.py --db events.sqlite3
    python tools/learn_from_recommendation_outcomes.py --db events.sqlite3 --user-id usr_31 --persist
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

from src.common.registry import OUTCOME_LEARNING_POLICY  # noqa: E402
from src.recommendations.outcome_learning import (  # noqa: E402
    DEFAULT_MODEL_VERSION,
    analyze_recommendation_outcomes,
)
from src.storage.repository import Repository  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse repeated recommendation outcomes into conservative belief_evidence "
            "proposals. Prints proposals; --persist appends the analysis signals."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", default=None, dest="user_id", help="Restrict analysis to this user.")
    parser.add_argument(
        "--min-trials", type=int, default=OUTCOME_LEARNING_POLICY.min_trials, dest="min_trials",
        help=f"Repeated outcomes required before any signal (default: {OUTCOME_LEARNING_POLICY.min_trials}).",
    )
    parser.add_argument(
        "--model-version", default=DEFAULT_MODEL_VERSION, dest="model_version",
        help=f"Version tag recorded on signals/proposals (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--as-of", default=None, dest="as_of",
        help="ISO 8601 datetime recorded as the signal's created_at; defaults to now (UTC).",
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Append the analysis signals to the outcome_learning_signals table (default: print only).",
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


def _belief_source_events(repo: Repository, recommendations) -> dict[str, list[str]]:
    """For every belief any recommendation used, collect the real
    ``source_event_ids`` from its active belief_evidence, so learning
    proposals trace back to leaf events where possible."""
    mapping: dict[str, set[str]] = {}
    for rec in recommendations:
        for belief_id in rec.belief_ids_used:
            if belief_id in mapping:
                continue
            events: set[str] = set()
            try:
                rows = repo.list_active_evidence(user_id=rec.user_id, belief_id=belief_id)
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                events.update(row.source_event_ids)
            mapping[belief_id] = events
    return {belief_id: sorted(events) for belief_id, events in mapping.items() if events}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        as_of = _parse_timestamp(args.as_of) if args.as_of else datetime.now(timezone.utc)
    except ValueError as exc:
        print(f"Invalid --as-of {args.as_of!r}: {exc}", file=sys.stderr)
        return 1

    try:
        repo = Repository.at_path(args.db) if args.persist else Repository.readonly_at_path(args.db)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        recommendations = repo.list_recommendations(user_id=args.user_id)
        shown_ids = {rec.recommendation_id for rec in recommendations}
        outcomes = [
            outcome
            for outcome in repo.list_recommendation_outcomes()
            if outcome.recommendation_id in shown_ids
        ]
        signals = analyze_recommendation_outcomes(
            recommendations,
            outcomes,
            as_of=as_of,
            min_trials=args.min_trials,
            model_version=args.model_version,
            belief_source_events=_belief_source_events(repo, recommendations),
        )

        print(json.dumps([json.loads(signal.model_dump_json()) for signal in signals], indent=2))

        if args.persist:
            appended = 0
            for signal in signals:
                try:
                    repo.insert_outcome_learning_signal(signal)
                    appended += 1
                except sqlite3.IntegrityError:
                    pass  # identical analysis already stored -- safe no-op
            print(
                f"persisted {appended} new signal(s) of {len(signals)} analysed "
                f"(proposals are NOT belief_evidence rows and did not change any belief).",
                file=sys.stderr,
            )
    finally:
        repo.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only inspection CLI for the Better You adaptive user model.

Dumps whatever is currently stored in a ``Repository.readonly_at_path(db)``
SQLite database as one JSON document, using the repository's own read/list
helpers (``list_events``, ``list_observations``,
``list_observation_events_for``, ``list_all_evidence``,
``list_latest_beliefs``, ``list_belief_key_canonicalizations``,
``list_recommendations``, ``list_recommendation_outcomes``,
``list_outcome_learning_signals``) -- never direct SQL, and never a write
path. This CLI performs no inserts, updates, invalidations, recomputes,
recommendation issuance, or outcome-learning evidence promotion; it exists
purely to answer "what is actually in this database right now" after using
the other CLIs. A database opened before a given table existed simply
reports that key as an empty list.

``readonly_at_path()`` (not the writable ``at_path()`` every intake CLI
uses) is what actually makes this true rather than merely intended: it opens
the database via SQLite's own URI ``mode=ro``, so an accidental write call
would fail at the SQLite layer, and it never creates a database file or
schema -- a typoed ``--db`` path fails loudly (nonzero exit, clear stderr)
instead of silently creating an empty database and printing empty results.

Confidence is never computed here -- ``beliefs`` shows exactly the latest
already-saved ``UserBelief`` row per (user_id, belief_id) scope
(``repo.list_latest_beliefs()``), including its ``locked_until_recompute``
flag as-is; this CLI does not call ``recompute_belief`` and does not judge
whether a shown belief is stale.

Run:
    python tools/inspect_user_model.py --db events.sqlite3 --pretty

    python tools/inspect_user_model.py --db events.sqlite3 \\
        --user-id usr_17 --belief-id bel_1 --include-inactive-evidence --pretty
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beliefs.models import active_evidence  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _list_or_empty(list_fn: Callable[..., list[Any]], **kwargs: Any) -> list[Any]:
    """A ``Repository.list_*`` call that treats a missing table (a database
    created before that table existed, opened read-only so the schema never
    ran) as an empty result rather than an error."""
    try:
        return list_fn(**kwargs)
    except sqlite3.OperationalError:
        return []


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump events, observations, observation_events, evidence, the latest beliefs, "
            "belief-key canonicalization decisions, recommendations, recommendation outcomes, "
            "and outcome-learning signals currently stored in a Repository. Read-only -- makes "
            "no changes."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--user-id", default=None, dest="user_id", help="Restrict output to this user.")
    parser.add_argument(
        "--belief-id", default=None, dest="belief_id",
        help="Restrict evidence and beliefs to this belief_id.",
    )
    parser.add_argument(
        "--include-inactive-evidence", action="store_true", dest="include_inactive_evidence",
        help="Include inactive/duplicate-suppressed evidence too (active only by default).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        repo = Repository.readonly_at_path(args.db)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        events = repo.list_events(user_id=args.user_id)
        observations = repo.list_observations(user_id=args.user_id)
        observation_events = repo.list_observation_events_for(
            [observation.observation_id for observation in observations]
        )
        evidence = repo.list_all_evidence(user_id=args.user_id, belief_id=args.belief_id)
        if not args.include_inactive_evidence:
            evidence = active_evidence(evidence)
        beliefs = repo.list_latest_beliefs(user_id=args.user_id, belief_id=args.belief_id)
        canonicalizations = _list_or_empty(
            repo.list_belief_key_canonicalizations, user_id=args.user_id
        )
        recommendations = _list_or_empty(repo.list_recommendations, user_id=args.user_id)
        shown_recommendation_ids = {r.recommendation_id for r in recommendations}
        recommendation_outcomes = [
            o
            for o in _list_or_empty(repo.list_recommendation_outcomes)
            if o.recommendation_id in shown_recommendation_ids
        ]
        outcome_learning_signals = _list_or_empty(
            repo.list_outcome_learning_signals, user_id=args.user_id
        )
        # Mark which signals have been promoted: a repeated_pattern_summary
        # belief_evidence row carrying the signal's independence_group exists.
        promoted_by_group: dict[str, list[str]] = {}
        for row in _list_or_empty(repo.list_all_evidence, user_id=args.user_id):
            if row.source_type.value == "repeated_pattern_summary":
                promoted_by_group.setdefault(row.independence_group, []).append(row.evidence_id)
    finally:
        repo.close()

    signal_dicts = []
    for signal in outcome_learning_signals:
        payload = json.loads(signal.model_dump_json())
        promoted_ids = sorted(promoted_by_group.get(signal.independence_group, []))
        payload["promoted"] = bool(promoted_ids)
        payload["promoted_evidence_ids"] = promoted_ids
        signal_dicts.append(payload)

    output = {
        "events": [json.loads(event.model_dump_json()) for event in events],
        "observations": [json.loads(observation.model_dump_json()) for observation in observations],
        "observation_events": [json.loads(link.model_dump_json()) for link in observation_events],
        "evidence": [json.loads(row.model_dump_json()) for row in evidence],
        "beliefs": [json.loads(belief.model_dump_json()) for belief in beliefs],
        "canonicalizations": [json.loads(c.model_dump_json()) for c in canonicalizations],
        "recommendations": [json.loads(r.model_dump_json()) for r in recommendations],
        "recommendation_outcomes": [
            json.loads(o.model_dump_json()) for o in recommendation_outcomes
        ],
        "outcome_learning_signals": signal_dicts,
    }
    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

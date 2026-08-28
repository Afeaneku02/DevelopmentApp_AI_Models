#!/usr/bin/env python3
"""Manual belief-evidence intake CLI for the Better You adaptive user model.

Loads an already-stored ``UserObservation`` (and the real ``UserEvent`` rows
its ``observation_events`` links claim to reference) from
``Repository.at_path(db)``, proposes a ``BeliefEvidenceProposal`` from them
via the existing ``propose_evidence_from_observation_validated()`` (which
re-checks that provenance against those real events -- no separate ad hoc
check here), authorizes it via the existing ``authorize_evidence()`` (the
only sanctioned proposal-to-persisted-record constructor), and saves the
result via ``repo.insert_evidence()``.

This CLI deliberately narrows what can be set: no CLI flag exists for
``authorized_aggregation_mode``, ``confidence``, ``status``, or any
``UserBelief`` field -- those are backend-owned or belong to a different
record entirely, and ``authorize_evidence()`` is the only place
``authorized_aggregation_mode`` can become anything other than
``leaf_default`` (never this CLI's own judgment call).

Controlled evidence replacement: ``--proposed-aggregation-mode
aggregate_replacement`` lets an operator *propose* that new evidence replace
one or more existing rows (``--replaces-evidence-id``, repeatable), and
``--backend-validation-passed`` stands in for a real backend policy
component asserting the replacement is legitimate -- see
``authorize_evidence()``'s own docstring for why that flag is scaffold-only,
not a production authorization guarantee. Either way, this CLI never decides
the outcome itself: it loads the replaced rows from the database and hands
them to ``authorize_evidence(..., replaced_evidence=...)``, which runs its
own deterministic checks (same user, same belief, coverage not narrowed) and
may still refuse the replacement (falling back to ``leaf_default`` with
``aggregation_review_status=rejected``) even when
``--backend-validation-passed`` is given. This CLI only ever creates the new
replacement evidence row -- it does not suppress, invalidate, or otherwise
modify the rows named in ``--replaces-evidence-id``.

This is belief-evidence intake only (blueprint section 6): it never
recomputes or saves a ``UserBelief``. That is a separate, already-built
pipeline step (``src.beliefs.recompute.recompute_belief``) that a caller
runs afterward against the evidence this CLI has stored.

Run:
    python tools/add_belief_evidence.py --db events.sqlite3 \\
        --evidence-id bev_1 --observation-id obs_1 --belief-id bel_1 \\
        --belief-type behavioral_tendency --direction support \\
        --source-type recorded_event --context-key fitness \\
        --strength 0.9 --model-version demo-0.1
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from src.beliefs.models import authorize_evidence  # noqa: E402
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated  # noqa: E402
from src.common.enums import AggregationMode, BeliefType, Direction, SourceType  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

# Matches the version literals used elsewhere in this project (e.g.
# tools/demo_user_model.py, tools/add_user_event.py, tools/add_user_observation.py)
# -- reusing the current tags, not inventing a new versioning scheme.
VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propose and authorize a single BeliefEvidence row from an already-stored "
            "UserObservation, then save it into a Repository. Evidence intake only -- "
            "never recomputes or saves a UserBelief."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--evidence-id", required=True, dest="evidence_id")
    parser.add_argument("--observation-id", required=True, dest="observation_id")
    parser.add_argument("--belief-id", required=True, dest="belief_id")
    parser.add_argument(
        "--belief-type", required=True, dest="belief_type",
        choices=[member.value for member in BeliefType],
    )
    parser.add_argument("--direction", required=True, choices=[member.value for member in Direction])
    parser.add_argument(
        "--source-type", required=True, dest="source_type",
        choices=[member.value for member in SourceType],
    )
    parser.add_argument("--context-key", required=True, dest="context_key")
    parser.add_argument("--strength", required=True, type=float)
    parser.add_argument("--model-version", required=True, dest="model_version")
    parser.add_argument(
        "--created-at", default=None, dest="created_at",
        help="ISO 8601 datetime for the evidence; defaults to now (UTC).",
    )
    parser.add_argument(
        "--proposed-aggregation-mode", default="leaf_default", dest="proposed_aggregation_mode",
        choices=[member.value for member in AggregationMode],
        help="Propose aggregate_replacement to request replacing existing evidence (default: leaf_default).",
    )
    parser.add_argument(
        "--replaces-evidence-id", action="append", default=[], dest="replaces_evidence_id",
        help=(
            "An evidence_id this new row proposes to replace. Repeatable. Only allowed "
            "with --proposed-aggregation-mode aggregate_replacement, where at least one is required."
        ),
    )
    parser.add_argument(
        "--backend-validation-passed", action="store_true", dest="backend_validation_passed",
        help=(
            "Assert that a real backend policy check approved this replacement (see "
            "authorize_evidence()'s docstring: this is scaffold-only, not a production "
            "authorization guarantee). Only allowed with --proposed-aggregation-mode "
            "aggregate_replacement; even then, authorize_evidence() may still refuse the "
            "replacement based on its own deterministic checks."
        ),
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

    if args.created_at is not None:
        try:
            created_at = _parse_timestamp(args.created_at)
        except ValueError as exc:
            print(f"Invalid --created-at {args.created_at!r}: {exc}", file=sys.stderr)
            return 1
    else:
        created_at = datetime.now(timezone.utc)

    is_replacement = args.proposed_aggregation_mode == AggregationMode.AGGREGATE_REPLACEMENT.value
    if not is_replacement:
        if args.replaces_evidence_id:
            print(
                "--replaces-evidence-id is only allowed with "
                "--proposed-aggregation-mode aggregate_replacement.",
                file=sys.stderr,
            )
            return 1
        if args.backend_validation_passed:
            print(
                "--backend-validation-passed is only allowed with "
                "--proposed-aggregation-mode aggregate_replacement.",
                file=sys.stderr,
            )
            return 1
    elif not args.replaces_evidence_id:
        print(
            "At least one --replaces-evidence-id is required with "
            "--proposed-aggregation-mode aggregate_replacement.",
            file=sys.stderr,
        )
        return 1
    else:
        duplicate_replaced_ids = sorted({
            evidence_id for evidence_id in set(args.replaces_evidence_id)
            if args.replaces_evidence_id.count(evidence_id) > 1
        })
        if duplicate_replaced_ids:
            print(
                "Duplicate --replaces-evidence-id value(s): "
                + ", ".join(repr(evidence_id) for evidence_id in duplicate_replaced_ids)
                + ". Each --replaces-evidence-id must be given at most once.",
                file=sys.stderr,
            )
            return 1

    repo = Repository.at_path(args.db)
    try:
        observation = repo.get_observation(args.observation_id)
        if observation is None:
            print(f"Unknown observation_id {args.observation_id!r}: not found in {args.db!r}.", file=sys.stderr)
            return 1

        links = repo.list_observation_events(args.observation_id)
        if not links:
            print(
                f"Observation {args.observation_id!r} has no observation_events links; "
                "cannot propose evidence with no provenance to draw from.",
                file=sys.stderr,
            )
            return 1

        referenced_event_ids = sorted({link.event_id for link in links})
        events = []
        missing_event_ids = []
        for event_id in referenced_event_ids:
            event = repo.get_event(event_id)
            if event is None:
                missing_event_ids.append(event_id)
            else:
                events.append(event)
        if missing_event_ids:
            for event_id in missing_event_ids:
                print(
                    f"Linked event_id {event_id!r} (from observation {args.observation_id!r}'s "
                    f"observation_events) not found in {args.db!r}.",
                    file=sys.stderr,
                )
            return 1

        replaced_evidence = None
        if is_replacement:
            replaced_evidence = {}
            missing_replaced_ids = []
            for replaced_id in args.replaces_evidence_id:  # duplicates already rejected above
                replaced_row = repo.get_evidence(replaced_id)
                if replaced_row is None:
                    missing_replaced_ids.append(replaced_id)
                else:
                    replaced_evidence[replaced_id] = replaced_row
            if missing_replaced_ids:
                for replaced_id in missing_replaced_ids:
                    print(
                        f"Unknown --replaces-evidence-id {replaced_id!r}: not found in {args.db!r}.",
                        file=sys.stderr,
                    )
                return 1

        try:
            proposal = propose_evidence_from_observation_validated(
                observation, links, events,
                belief_id=args.belief_id,
                direction=Direction(args.direction),
                source_type=SourceType(args.source_type),
                context_key=args.context_key,
                strength=args.strength,
                model_version=args.model_version,
                belief_type=BeliefType(args.belief_type),
                proposed_aggregation_mode=AggregationMode(args.proposed_aggregation_mode),
                replaces_evidence_ids=args.replaces_evidence_id,
                **VERSION_FIELDS,
            )
        except ValidationError as exc:
            print(f"Evidence proposal validation failed:\n{exc}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        evidence = authorize_evidence(
            proposal, evidence_id=args.evidence_id, created_at=created_at,
            aggregation_policy_version=AGGREGATION_POLICY_VERSION,
            backend_validation_passed=args.backend_validation_passed,
            replaced_evidence=replaced_evidence,
        )

        try:
            repo.insert_evidence(evidence)
        except sqlite3.IntegrityError as exc:
            print(
                f"Duplicate evidence_id {args.evidence_id!r} (already stored in {args.db!r}): {exc}",
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(f"Cannot insert evidence {args.evidence_id!r}: {exc}", file=sys.stderr)
            return 1
    finally:
        repo.close()

    print(evidence.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

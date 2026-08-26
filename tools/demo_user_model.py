#!/usr/bin/env python3
"""Deterministic demo of the Better You adaptive user model, end to end.

Runs one small, fixed scenario through the already-built ``src/`` modules
only -- no new scoring formula, no shortcut around any validator or
repository built across Phases 1-4:

    UserEvent (x3) -> UserObservation -> BeliefEvidenceProposal
    -> authorized BeliefEvidence -> Repository.in_memory()
    -> UserBelief recompute -> saved belief

Scenario: three after-work workout completions build a
``higher_adherence_after_work`` belief. All backing evidence is then
invalidated (e.g. a user-requested deletion), and the demo shows the belief
correctly LOCKED before a fresh recompute runs (blueprint section 6.0.2's
fail-closed guarantee, enforced by ``Repository.mark_evidence_inactive()``),
then correctly reset to ``confidence=0.0``/``status=outdated`` after one.

Run:
    python tools/demo_user_model.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beliefs.models import authorize_evidence  # noqa: E402
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated  # noqa: E402
from src.beliefs.recompute import recompute_belief  # noqa: E402
from src.events.models import UserEvent  # noqa: E402
from src.observations.create_observation import create_observation_from_event  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

USER_ID = "usr_17"
BELIEF_ID = "bel_88"
BELIEF_TYPE = "behavioral_tendency"
BELIEF_KEY = "higher_adherence_after_work"

AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)
EVENT_OFFSETS_DAYS = (5, 3, 1)


def _kv(label: str, value: object) -> str:
    return f"  {label}: {value}"


def _belief_snapshot(label: str, belief) -> list[str]:
    return [
        f"-- {label} --",
        _kv("confidence", f"{belief.confidence:.6f}"),
        _kv("status", belief.status.value),
        _kv("effective_support_count", f"{belief.effective_support_count:.6f}"),
        _kv("locked_until_recompute", belief.locked_until_recompute),
    ]


def run() -> str:
    """Runs the full scenario and returns the printable summary as one
    string (also printed to stdout by ``main()``). Returning the text
    rather than only printing it is what lets tests assert on the exact
    output.
    """
    lines: list[str] = ["=== Better You adaptive user model: deterministic demo ===", ""]
    repo = Repository.in_memory()

    # --- 1. Three after-work workout completion events. ---
    lines.append("Scenario: three after-work workout completions.")
    lines.append("")
    events = []
    for index, days_ago in enumerate(EVENT_OFFSETS_DAYS, start=1):
        event = UserEvent(
            event_id=f"evt_104{index}", user_id=USER_ID, event_type="goal_completed",
            timestamp=AS_OF - timedelta(days=days_ago),
            structured_data={"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:20"},
            source="app", goal_id="goal_8", **VERSION_FIELDS,
        )
        repo.insert_event(event)
        events.append(event)
        lines.append(_kv(f"UserEvent {event.event_id}", f"{event.event_type} @ {event.timestamp.isoformat()}"))

    # --- 2 & 3. Observation + proposed/authorized evidence per event. ---
    lines.append("")
    lines.append("-- Observations and authorized belief_evidence --")
    evidence_rows = []
    for event in events:
        observation, links = create_observation_from_event(
            event, observation_id=f"obs_{event.event_id}", category="routine",
            observation_text="User completed an after-work workout.",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        repo.insert_observation(observation, links)

        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="demo-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id=f"bev_{event.event_id}", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        repo.insert_evidence(evidence)
        evidence_rows.append(evidence)
        lines.append(_kv(
            f"BeliefEvidence {evidence.evidence_id}",
            f"source_event_ids={evidence.source_event_ids} "
            f"source_type={evidence.source_type.value} source_reliability={evidence.source_reliability} "
            f"authorized_aggregation_mode={evidence.authorized_aggregation_mode.value}",
        ))

    # --- 4 & 5. Recompute from active evidence, save the belief. ---
    lines.append("")
    active = repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID)
    first_belief = recompute_belief(
        belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
        belief_value=True, evidence=active, as_of=AS_OF, first_observed=events[0].timestamp, **VERSION_FIELDS,
    )
    repo.save_belief(first_belief)
    lines.extend(_belief_snapshot("Recompute #1: from active evidence, saved", repo.get_latest_belief(
        user_id=USER_ID, belief_id=BELIEF_ID,
    )))

    # --- 6. Invalidate all evidence; the belief must lock, not go stale. ---
    lines.append("")
    lines.append("-- Invalidating all evidence (e.g. a user-requested deletion) --")
    for evidence in evidence_rows:
        repo.mark_evidence_inactive(evidence.evidence_id, reason="deletion", invalidated_at=AS_OF)
    remaining_active = repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID)
    lines.append(_kv("active evidence remaining", len(remaining_active)))
    lines.append("")
    lines.extend(_belief_snapshot(
        "Latest belief BEFORE recompute (must be locked, not silently stale)",
        repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID),
    ))

    # --- 7. Recompute after invalidation: confidence resets, lock clears. ---
    lines.append("")
    recomputed = recompute_belief(
        belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
        belief_value=True, evidence=repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
        as_of=AS_OF, first_observed=events[0].timestamp, recompute_reason="deletion", **VERSION_FIELDS,
    )
    repo.save_belief(recomputed)
    lines.extend(_belief_snapshot(
        "Recompute #2: after invalidation, saved",
        repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID),
    ))

    repo.close()
    return "\n".join(lines) + "\n"


def main() -> int:
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fixture builder for the mentor-feedback tests.

Builds real event -> observation -> evidence -> belief chains through the
sanctioned functions (never by hand-constructing a UserBelief), so the
beliefs the mentor engine sees are genuinely scored.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.beliefs.models import authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.common.enums import BeliefStatus
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)
AS_OF = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
_AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"

_SOURCE_TYPES = ("recorded_event", "explicit_user_statement", "model_observation", "explicit_user_correction")


def seed_belief(
    repo: Repository,
    *,
    user_id: str,
    belief_id: str,
    belief_key: str = "higher_adherence_after_work",
    belief_value: object = True,
    belief_type: str = "behavioral_tendency",
    support: int = 3,
    contradict: int = 0,
    strength: float = 0.9,
    recompute_reason: str | None = None,
    no_active_evidence_status: BeliefStatus | None = None,
    as_of: datetime = AS_OF,
):
    """Build ``support`` supporting + ``contradict`` contradicting evidence
    rows for one belief and recompute it. Returns the saved ``UserBelief``."""
    specs = [("support", i) for i in range(support)] + [("contradict", i) for i in range(contradict)]
    for direction, index in specs:
        tag = f"{belief_id}_{direction}_{index}"
        observed = as_of - timedelta(days=1 + index)
        event = UserEvent(
            event_id=f"evt_{tag}", user_id=user_id, event_type="goal_completed",
            timestamp=observed, source="app", **VERSION_FIELDS,
        )
        repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id=f"obs_{tag}", category="routine",
            observation_text="recorded activity", importance=0.6, confidence=0.6,
            created_at=observed, **VERSION_FIELDS,
        )
        repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=belief_id, direction=direction,
            source_type=_SOURCE_TYPES[index % len(_SOURCE_TYPES)], context_key="fitness",
            strength=strength, model_version="mentor-test", belief_type=belief_type,
            **VERSION_FIELDS,
        )
        repo.insert_evidence(authorize_evidence(
            proposal, evidence_id=f"bev_{tag}", created_at=observed,
            aggregation_policy_version=_AGGREGATION_POLICY_VERSION,
        ))

    active = repo.list_active_evidence(user_id=user_id, belief_id=belief_id)
    extra = {}
    if no_active_evidence_status is not None:
        extra["no_active_evidence_status"] = no_active_evidence_status
    belief = recompute_belief(
        belief_id=belief_id, user_id=user_id, belief_type=belief_type,
        belief_key=belief_key, belief_value=belief_value,
        evidence=active, as_of=as_of,
        first_observed=min((e.observed_at for e in active), default=as_of),
        recompute_reason=recompute_reason, **VERSION_FIELDS, **extra,
    )
    repo.save_belief(belief)
    return belief

"""The bridge between an API request model and the existing sanctioned
model/repository functions.

Nothing here scores, recomputes, ranks, analyses outcomes, or promotes --
each function builds a real domain record the same way the matching
``tools/add_*`` / ``tools/recompute_belief`` / ``tools/make_recommendation``
CLI does (inject the backend-owned version tags, call the one sanctioned
constructor) and persists it through ``Repository``. The read helpers just
call the viewer/eval collectors.

Errors are raised, not translated to HTTP here (``src.api.app`` maps them):

- ``LookupError``  -- a referenced record (event, observation) does not exist
- ``ValueError``   -- a domain rule rejected the request
- ``pydantic.ValidationError`` -- a constructed record failed model validation
- ``sqlite3.IntegrityError``  -- the id is already used
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.beliefs.models import authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.common.enums import LinkRole
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation
from src.recommendations.engine import generate_recommendation
from src.recommendations.models import RecommendationOutcome
from src.viewer.evals_view import collect_eval_report
from src.viewer.reviews_view import collect_review_queue
from src.viewer.user_model_view import collect_view_model

from src.api.models import (
    BeliefEvidenceIn,
    EventIn,
    ObservationIn,
    RecommendationIn,
    RecommendationOutcomeIn,
    RecomputeIn,
)

# The version literals used by every intake tool in this repo (see
# tools/add_user_event.py). Backend-owned: clients never supply these.
VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)
AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------ writes --


def create_event(repo: Any, payload: EventIn) -> dict[str, Any]:
    event = UserEvent(
        event_id=payload.event_id,
        user_id=payload.user_id,
        event_type=payload.event_type,
        timestamp=_utc(payload.timestamp),
        raw_content=payload.raw_content,
        structured_data=payload.structured_data,
        source=payload.source,
        **VERSION_FIELDS,
    )
    repo.insert_event(event)
    return event.model_dump(mode="json")


def _resolve_primary_event_id(event_ids: list[str], primary: str | None) -> str:
    if primary is not None:
        if primary not in event_ids:
            raise ValueError(
                f"primary_event_id {primary!r} is not one of event_ids {event_ids!r}"
            )
        return primary
    if len(event_ids) == 1:
        return event_ids[0]
    raise ValueError("primary_event_id is required when more than one event_id is given")


def create_observation(repo: Any, payload: ObservationIn) -> dict[str, Any]:
    event_ids = payload.event_ids
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event_ids must not contain duplicates")

    primary = _resolve_primary_event_id(event_ids, payload.primary_event_id)

    for event_id in event_ids:
        event = repo.get_event(event_id)
        if event is None:
            raise LookupError(f"unknown event_id {event_id!r}")
        if event.user_id != payload.user_id:
            raise ValueError(
                f"event_id {event_id!r} belongs to a different user than {payload.user_id!r}"
            )

    created_at = _utc(payload.created_at)
    observation = UserObservation(
        observation_id=payload.observation_id,
        user_id=payload.user_id,
        category=payload.category,
        observation=payload.observation,
        importance=payload.importance,
        confidence=payload.confidence,
        created_at=created_at,
        **VERSION_FIELDS,
    )
    links = [
        ObservationEvent(
            observation_id=payload.observation_id,
            event_id=event_id,
            link_role=LinkRole.PRIMARY if event_id == primary else LinkRole.SUPPORTING,
            created_at=created_at,
            **VERSION_FIELDS,
        )
        for event_id in event_ids
    ]
    repo.insert_observation(observation, links)
    return {
        "observation": observation.model_dump(mode="json"),
        "observation_events": [link.model_dump(mode="json") for link in links],
    }


def create_belief_evidence(repo: Any, payload: BeliefEvidenceIn) -> dict[str, Any]:
    observation = repo.get_observation(payload.observation_id)
    if observation is None:
        raise LookupError(f"unknown observation_id {payload.observation_id!r}")

    links = repo.list_observation_events(payload.observation_id)
    if not links:
        raise ValueError(
            f"observation {payload.observation_id!r} has no provenance links to draw evidence from"
        )

    events = []
    for event_id in sorted({link.event_id for link in links}):
        event = repo.get_event(event_id)
        if event is None:
            raise LookupError(
                f"linked event_id {event_id!r} (from observation {payload.observation_id!r}) not found"
            )
        events.append(event)

    proposal = propose_evidence_from_observation_validated(
        observation,
        links,
        events,
        belief_id=payload.belief_id,
        direction=payload.direction,
        source_type=payload.source_type,
        context_key=payload.context_key,
        strength=payload.strength,
        model_version=payload.model_version,
        belief_type=payload.belief_type,
        **VERSION_FIELDS,
    )
    # The only sanctioned proposal -> BeliefEvidence path. No
    # backend_validation_passed, no replaced_evidence: aggregation stays
    # leaf_default. The API deliberately exposes no aggregate-replacement.
    evidence = authorize_evidence(
        proposal,
        evidence_id=payload.evidence_id,
        created_at=_utc(payload.created_at),
        aggregation_policy_version=AGGREGATION_POLICY_VERSION,
    )
    repo.insert_evidence(evidence)
    return evidence.model_dump(mode="json")


def recompute(repo: Any, belief_id: str, payload: RecomputeIn) -> dict[str, Any]:
    active = repo.list_active_evidence(user_id=payload.user_id, belief_id=belief_id)
    if not active and not payload.allow_no_evidence:
        raise ValueError(
            f"no active evidence for belief_id={belief_id!r}, user_id={payload.user_id!r}; "
            "set allow_no_evidence=true to recompute to confidence 0.0"
        )

    as_of = _utc(payload.as_of)
    if payload.first_observed is not None:
        first_observed = _utc(payload.first_observed)
    else:
        first_observed = min((row.observed_at for row in active), default=as_of)

    belief = recompute_belief(
        belief_id=belief_id,
        user_id=payload.user_id,
        belief_type=payload.belief_type,
        belief_key=payload.belief_key,
        belief_value=payload.belief_value,
        evidence=active,
        as_of=as_of,
        first_observed=first_observed,
        recompute_reason=payload.recompute_reason,
        **VERSION_FIELDS,
    )
    repo.save_belief(belief)
    return belief.model_dump(mode="json")


def make_recommendation(repo: Any, payload: RecommendationIn) -> dict[str, Any]:
    beliefs = repo.list_latest_beliefs(user_id=payload.user_id)
    recommendation = generate_recommendation(
        recommendation_id=payload.recommendation_id,
        user_id=payload.user_id,
        context_key=payload.context_key,
        beliefs=beliefs,
        created_at=_utc(payload.created_at),
        goal=payload.goal,
        expected_outcome=payload.expected_outcome,
    )
    repo.insert_recommendation(recommendation)
    return recommendation.model_dump(mode="json")


def record_recommendation_outcome(
    repo: Any, payload: RecommendationOutcomeIn
) -> dict[str, Any]:
    outcome = RecommendationOutcome(
        outcome_id=payload.outcome_id,
        recommendation_id=payload.recommendation_id,
        followed=payload.followed,
        result=payload.result,
        user_feedback=payload.user_feedback,
        measured_result=payload.measured_result,
        source=payload.source,
        observed_at=_utc(payload.observed_at) if payload.observed_at is not None else None,
        created_at=_utc(payload.created_at),
        **VERSION_FIELDS,
    )
    # Append-only and purely descriptive. Recording an outcome never analyses
    # outcomes, never creates an outcome-learning signal, and never promotes
    # anything -- that stays a separate, explicit CLI step.
    repo.insert_recommendation_outcome(outcome)
    return outcome.model_dump(mode="json")


# ------------------------------------------------------------------- reads --


def read_user_model(repo: Any, *, db_path: str, user_id: str) -> dict[str, Any]:
    view_model = collect_view_model(repo, db_path=db_path, user_id=user_id)
    return {
        "user_id": user_id,
        "generated_at": view_model.generated_at.isoformat(),
        "summary": view_model.summary(),
        "events": view_model.events,
        "observations": view_model.observations,
        "observation_events": view_model.observation_events,
        "evidence": view_model.evidence,
        "beliefs": view_model.beliefs,
        "canonicalizations": view_model.canonicalizations,
        "recommendations": view_model.recommendations,
        "recommendation_outcomes": view_model.recommendation_outcomes,
        "outcome_learning_signals": view_model.outcome_learning_signals,
        "outcome_learning_signal_reviews": view_model.outcome_learning_signal_reviews,
    }


def read_user_reviews(repo: Any, *, db_path: str, user_id: str) -> dict[str, Any]:
    queue = collect_review_queue(repo, db_path=db_path, user_id=user_id)
    return {
        "user_id": user_id,
        "generated_at": queue.generated_at.isoformat(),
        "summary": queue.summary(),
        "pending_signals": queue.pending_signals,
        "reviewed_signals": queue.reviewed_signals,
        "reviews": queue.reviews,
        "note": (
            "read-only: approve/reject a pending signal from the CLI "
            "(tools/review_outcome_learning_signal.py); the API does not promote or review"
        ),
    }


def read_evals() -> dict[str, Any]:
    report, resolved_dir = collect_eval_report()
    return {"manifest_dir": str(resolved_dir), **report.to_dict()}

"""Deterministic provenance validators shared across models.

These check referential integrity that a single Pydantic model cannot check
about itself (whether an observation has a linked primary event, whether
evidence points at real events) -- the same class of check as
``skills/shared/extract-user-observations/scripts/check_evidence_links.py``,
generalized to operate on the typed models in ``src/`` instead of raw dicts,
and extended to belief_evidence provenance.
"""
from __future__ import annotations

from src.beliefs.models import BeliefEvidence
from src.common.enums import LinkRole
from src.observations.models import ObservationEvent, UserObservation


def validate_observation_provenance(
    observations: list[UserObservation],
    links: list[ObservationEvent],
    known_event_ids: set[str] | None = None,
) -> list[str]:
    """Every observation needs >=1 primary link; every link must reference a
    real observation and (when known_event_ids is supplied) a real event."""
    errors: list[str] = []
    observation_ids = {o.observation_id for o in observations}
    if len(observation_ids) != len(observations):
        errors.append("Duplicate observation_id values in observations batch.")

    primary_count: dict[str, int] = {oid: 0 for oid in observation_ids}
    for link in links:
        if link.observation_id not in observation_ids:
            errors.append(f"observation_events references unknown observation_id: {link.observation_id!r}")
            continue
        if known_event_ids is not None and link.event_id not in known_event_ids:
            errors.append(f"observation_events references unknown event_id: {link.event_id!r}")
        if link.link_role == LinkRole.PRIMARY:
            primary_count[link.observation_id] += 1

    for observation_id in observation_ids:
        if primary_count[observation_id] == 0:
            errors.append(f"observation {observation_id!r} has no primary observation_events link.")
    return errors


def validate_belief_evidence_provenance(
    evidence_items: list[BeliefEvidence],
    known_event_ids: set[str] | None = None,
) -> list[str]:
    """Every evidence row's source_event_ids must be non-empty and, when
    known_event_ids is supplied, must reference real events."""
    errors: list[str] = []
    for evidence in evidence_items:
        if not evidence.source_event_ids:
            errors.append(f"belief_evidence {evidence.evidence_id!r} has empty source_event_ids.")
            continue
        if known_event_ids is not None:
            for event_id in evidence.source_event_ids:
                if event_id not in known_event_ids:
                    errors.append(
                        f"belief_evidence {evidence.evidence_id!r} references unknown "
                        f"event_id {event_id!r} in source_event_ids."
                    )
    return errors

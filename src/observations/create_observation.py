"""Runtime pipeline: build a UserObservation (plus its primary
observation_events link) from a single UserEvent -- extract-user-
observations's runtime implementation for the narrowest possible case: one
event, one observation.

Deliberately narrow: a real extraction pipeline looks for *patterns* across
many events (blueprint section 6.2 cold-start bands) and needs semantic
judgment -- an LLM or a human -- to decide what an event means. This function
supplies neither: ``category`` and ``observation_text`` are caller-supplied,
exactly as ``direction``/``source_type``/``strength`` are caller-supplied to
``propose_evidence_from_event()`` in ``src/beliefs/propose_evidence.py``.
What this function does do deterministically is the structural half:
assigning identifiers, generating the mandatory ``observation_events``
primary link (a single-event observation has exactly one, trivially), and
populating version fields -- so provenance is never an afterthought.
"""
from __future__ import annotations

from datetime import datetime

from src.common.enums import LinkRole
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation


def create_observation_from_event(
    event: UserEvent,
    *,
    observation_id: str,
    category: str,
    observation_text: str,
    importance: float,
    confidence: float,
    created_at: datetime,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
) -> tuple[UserObservation, list[ObservationEvent]]:
    """Returns ``(observation, [primary_link])``. The link list always has
    exactly one row with ``link_role=primary``, since a single-event
    observation has nothing else to link to; extract-user-observations's own
    completion check ("every observation has >=1 primary observation_events
    link") is satisfied trivially by construction here, not by a separate
    validation pass a caller has to remember to run.
    """
    observation = UserObservation(
        observation_id=observation_id,
        user_id=event.user_id,
        category=category,
        observation=observation_text,
        importance=importance,
        confidence=confidence,
        created_at=created_at,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )
    primary_link = ObservationEvent(
        observation_id=observation_id,
        event_id=event.event_id,
        link_role=LinkRole.PRIMARY,
        created_at=created_at,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )
    return observation, [primary_link]

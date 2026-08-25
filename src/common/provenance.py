"""Deterministic cross-record provenance validators.

These check referential integrity that a single Pydantic model cannot check
about itself (whether an observation's links point at events that actually
exist, whether a link crosses users, whether evidence points at real
same-user events) -- the same class of check as
``skills/shared/extract-user-observations/scripts/check_evidence_links.py``,
generalized to operate on the typed models in ``src/`` against real
``UserEvent`` records instead of a bare set of IDs.

Both validators take the actual ``UserEvent`` rows being referenced (not just
their IDs) specifically so they can check ``user_id`` consistency, not only
existence -- a link or a piece of evidence that resolves to a real event
belonging to a *different* user is exactly as dangerous as one that resolves
to no event at all, and both must fail closed the same way.
"""
from __future__ import annotations

from src.beliefs.models import BeliefEvidence
from src.common.enums import LinkRole
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation


def validate_observation_provenance(
    observations: list[UserObservation],
    links: list[ObservationEvent],
    events: list[UserEvent],
) -> list[str]:
    """Checks, for every ``ObservationEvent`` link:

    - ``observation_id`` references a real observation in ``observations``.
    - ``event_id`` references a real event in ``events``.
    - the linked event's ``user_id`` matches the observation's ``user_id``.

    And, for every observation in ``observations``:

    - it has at least one link with ``link_role="primary"``.
    """
    errors: list[str] = []
    observation_by_id = {observation.observation_id: observation for observation in observations}
    if len(observation_by_id) != len(observations):
        errors.append("Duplicate observation_id values in observations batch.")
    event_by_id = {event.event_id: event for event in events}
    if len(event_by_id) != len(events):
        errors.append("Duplicate event_id values in events batch.")

    primary_count: dict[str, int] = {observation_id: 0 for observation_id in observation_by_id}
    for link in links:
        observation = observation_by_id.get(link.observation_id)
        if observation is None:
            errors.append(f"observation_events references unknown observation_id: {link.observation_id!r}")
            continue

        event = event_by_id.get(link.event_id)
        if event is None:
            errors.append(f"observation_events references unknown event_id: {link.event_id!r}")
        elif event.user_id != observation.user_id:
            errors.append(
                f"observation_events link {link.observation_id!r} -> {link.event_id!r} crosses users: "
                f"event.user_id={event.user_id!r} != observation.user_id={observation.user_id!r}"
            )

        if link.link_role == LinkRole.PRIMARY:
            primary_count[link.observation_id] += 1

    for observation_id in observation_by_id:
        if primary_count[observation_id] == 0:
            errors.append(f"observation {observation_id!r} has no primary observation_events link.")
    return errors


def validate_belief_evidence_provenance(
    evidence_items: list[BeliefEvidence],
    events: list[UserEvent],
) -> list[str]:
    """Checks, for every ``BeliefEvidence`` row:

    - ``source_event_ids`` is non-empty.
    - every id in ``source_event_ids`` references a real event in ``events``.
    - every referenced event's ``user_id`` matches the evidence row's
      ``user_id``.
    - ``event_id``, when present, is included in ``source_event_ids``.

    And, for ``events`` itself:

    - it contains no duplicate ``event_id`` values.

    The last check matters specifically because this function builds
    ``event_by_id = {event.event_id: event for event in events}``: a naive
    dict comprehension silently collapses duplicate keys to whichever event
    happens to appear last, so a caller-supplied ``events`` batch containing
    two different ``UserEvent`` rows under the same ``event_id`` (for
    example, belonging to two different users) would make the subsequent
    user_id check order-dependent -- passing or failing based on iteration
    order rather than on anything real about the data. Flagging the
    duplicate directly, matching ``validate_observation_provenance()``'s own
    duplicate-``event_id`` check, means the overall result is always a
    failure in that case regardless of which row the dict happened to keep.

    The ``event_id``-inclusion check above is defense-in-depth:
    ``BeliefEvidenceProposal`` (which ``BeliefEvidence`` extends) already
    enforces this at construction time via its own model validator, so it
    should be unreachable through normal construction. It is re-checked here
    anyway for a row that reached this validator by some other path (e.g.
    ``model_construct()``, which skips validation, or deserialization of
    data this process did not itself validate) -- this validator's job is to
    trust nothing about how the ``BeliefEvidence`` objects it receives came
    to exist.
    """
    errors: list[str] = []
    event_by_id = {event.event_id: event for event in events}
    if len(event_by_id) != len(events):
        errors.append("Duplicate event_id values in events batch.")
    for evidence in evidence_items:
        if not evidence.source_event_ids:
            errors.append(f"belief_evidence {evidence.evidence_id!r} has empty source_event_ids.")
            continue

        for event_id in evidence.source_event_ids:
            event = event_by_id.get(event_id)
            if event is None:
                errors.append(
                    f"belief_evidence {evidence.evidence_id!r} references unknown event_id {event_id!r} "
                    "in source_event_ids."
                )
            elif event.user_id != evidence.user_id:
                errors.append(
                    f"belief_evidence {evidence.evidence_id!r} references event {event_id!r} belonging to "
                    f"a different user (event.user_id={event.user_id!r} != evidence.user_id={evidence.user_id!r})."
                )

        if evidence.event_id is not None and evidence.event_id not in evidence.source_event_ids:
            errors.append(
                f"belief_evidence {evidence.evidence_id!r} has event_id={evidence.event_id!r} not included "
                "in its own source_event_ids."
            )
    return errors

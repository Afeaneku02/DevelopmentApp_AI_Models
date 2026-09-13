"""Deterministic pipeline step: turn stored, low-risk Better You
``user_events`` into conservative ``user_observations`` (blueprint section
5/6.2's structural half -- see ``src.observations.create_observation`` for
the single-event primitive this builds on).

This is a batch wrapper around ``create_observation_from_event``, not a new
extraction mechanism: for each event it decides *whether* a safe, structured
observation can be drafted, and if so *what* it says, using nothing but a
fixed, reviewed mapping from an allowlisted ``event_type`` and its
``structured_data`` shape to a fixed observation template. There is no LLM,
no free-text summarization, and no semantic judgment call at runtime --
every wording is a compile-time constant selected by an enum-like structured
field (e.g. a check-in's ``response``), never composed from event content.

Deliberately conservative in scope:

- **Allowlisted event types only** (``_DRAFTERS``). Every other event_type --
  including every other real Better You ``ActivityEventType`` this project
  does not yet have a reviewed mapping for (``goal_created``, ``goal_paused``,
  ``goal_resumed``, ``goal_completed``, ``goal_archived``,
  ``roadmap_generated``) -- is skipped, not guessed at. Extending the
  allowlist is a deliberate, reviewed code change, not something this
  pipeline infers on its own.
- **Only ``structured_data`` is ever read.** ``raw_content`` is never
  inspected and never required -- an event missing it, or carrying it, is
  processed identically. Nothing free-text ever reaches an observation's
  wording; the text is always one of a handful of fixed templates.
- **No sensitive or high-impact claims.** Every mapping describes a single,
  already-happened, low-stakes behavioral fact ("a check-in happened", "a
  roadmap step was completed") -- never a diagnosis, a prediction, a
  causal claim, or anything about health, finances, or emotional state.
- **No belief_evidence or beliefs.** This stops at ``user_observations`` --
  ``src.beliefs.propose_evidence`` and ``src.beliefs.recompute`` remain
  separate, already-built, explicitly-invoked steps a caller runs
  afterward.
- **Idempotent.** An event that already has an ``observation_events`` link
  (from any prior run) is always skipped -- rerunning against the same
  events, or a superset of them, creates no duplicate observations. As a
  second, independent layer, the derived ``observation_id`` (``f"obs_{event.
  event_id}"``, the same convention already used in
  ``tools/demo_user_model.py``) is deterministic, so even a bug that let a
  duplicate slip past the first check would still hit a primary-key
  collision in storage rather than silently double-inserting.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.observations.models import ObservationEvent, UserObservation

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)


@dataclass(frozen=True)
class ObservationDraft:
    """What a drafter decided to say about one event, before any id/version
    fields are attached."""

    category: str
    observation_text: str
    importance: float
    confidence: float


def _is_nonblank_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# Better You's own wire format (services/mentor-feedback's sibling,
# services/activity's HttpActivityEventSyncClient -- ADR 0027 in that
# project) sends structured_data with these exact camelCase keys, unchanged
# by this project's EventIn/UserEvent intake. Response wording is looked up
# by the reported enum value, never composed -- a response value outside
# this table is treated as malformed data (skipped), not guessed at.
_CHECK_IN_RESPONSE_TEXT = {
    "yes": "User completed a scheduled check-in.",
    "no": "User recorded a missed check-in.",
    "partly": "User recorded a partially completed check-in.",
    "skipped": "User skipped a scheduled check-in.",
}


def _draft_check_in_recorded(structured_data: dict[str, Any]) -> ObservationDraft | None:
    if not _is_nonblank_str(structured_data.get("goalId")):
        return None
    if not _is_nonblank_str(structured_data.get("checkInId")):
        return None
    response = structured_data.get("response")
    if not isinstance(response, str) or response not in _CHECK_IN_RESPONSE_TEXT:
        return None
    return ObservationDraft(
        category="check_in",
        observation_text=_CHECK_IN_RESPONSE_TEXT[response],
        # A single check-in is a frequent, routine data point -- deliberately
        # low importance regardless of which response was reported. Treating
        # e.g. "no"/"skipped" as more important would be an inference about
        # significance this pipeline is not authorized to make.
        importance=0.3,
        # High confidence: this is a directly reported structured fact, not
        # an inference.
        confidence=0.9,
    )


def _draft_roadmap_step_completed(structured_data: dict[str, Any]) -> ObservationDraft | None:
    required_keys = ("goalId", "roadmapId", "milestoneId", "actionStepId")
    if not all(_is_nonblank_str(structured_data.get(key)) for key in required_keys):
        return None
    return ObservationDraft(
        category="goal_progress",
        observation_text="User completed a roadmap action step.",
        # A step completion is a more meaningful progress signal than a
        # single check-in, but still a routine, low-stakes fact.
        importance=0.5,
        confidence=0.9,
    )


# The reviewed allowlist. Adding an event type here is a deliberate code
# change (see the module docstring) -- never inferred from what a stored
# event happens to contain.
_DRAFTERS: dict[str, Callable[[dict[str, Any]], ObservationDraft | None]] = {
    "check_in_recorded": _draft_check_in_recorded,
    "roadmap_step_completed": _draft_roadmap_step_completed,
}


def _draft_observation(event: UserEvent) -> ObservationDraft | None:
    """``None`` covers both "not on the allowlist" and "on the allowlist but
    ``structured_data`` doesn't match the expected shape" -- callers that
    need to distinguish the two check ``event.event_type in _DRAFTERS``
    themselves, as ``process_user_events`` does for its outcome reporting.
    Never raises: a malformed or unrecognized event is always "cannot safely
    process," not an error.
    """
    drafter = _DRAFTERS.get(event.event_type)
    if drafter is None:
        return None
    if not isinstance(event.structured_data, dict):
        return None
    return drafter(event.structured_data)


# Action taken for one event, always one of:
#   "created"                    - a new observation was written (persist=True)
#   "would_create"                - a new observation would be written (persist=False)
#   "skipped_already_processed"  - a prior run (or a primary-key collision
#                                    caught at insert time) already covered
#                                    this event
#   "skipped_unsupported_type"   - event_type is not on the allowlist
#   "skipped_malformed_data"     - event_type is allowlisted but
#                                    structured_data doesn't match the
#                                    expected shape
#   "skipped_not_found"          - an explicitly requested event_id does not
#                                    resolve to a stored event owned by this
#                                    user (unknown id, or another user's)
@dataclass(frozen=True)
class EventProcessingOutcome:
    event_id: str
    event_type: str | None
    action: str
    observation_id: str | None = None


@dataclass(frozen=True)
class ProcessEventsResult:
    user_id: str
    persisted: bool
    outcomes: list[EventProcessingOutcome]

    @property
    def created_observation_ids(self) -> list[str]:
        action = "created" if self.persisted else "would_create"
        return [o.observation_id for o in self.outcomes if o.action == action and o.observation_id]


def process_user_events(
    repo: Any,
    *,
    user_id: str,
    event_ids: list[str] | None = None,
    as_of: datetime,
    persist: bool = False,
) -> ProcessEventsResult:
    """Draft (and, if ``persist``, write) observations for ``user_id``'s
    events.

    ``event_ids=None`` (the default) considers every stored event for
    ``user_id`` -- in practice, only the ones without an existing
    ``observation_events`` link do anything, since every already-linked
    event is reported as ``skipped_already_processed``. Passing
    ``event_ids`` restricts consideration to exactly those ids; an id that
    is not a stored event owned by ``user_id`` is reported as
    ``skipped_not_found`` rather than raising, so one typo in a batch does
    not abort the rest.

    ``repo`` is only read from unless ``persist=True``. Never raises for
    anything about the events themselves -- every event gets an
    ``EventProcessingOutcome``, whatever it says.
    """
    outcomes: list[EventProcessingOutcome] = []
    events: list[UserEvent] = []

    if event_ids is not None:
        for event_id in event_ids:
            event = repo.get_event(event_id)
            if event is None or event.user_id != user_id:
                outcomes.append(EventProcessingOutcome(event_id, None, "skipped_not_found"))
                continue
            events.append(event)
    else:
        events = repo.list_events(user_id=user_id)

    events = sorted(events, key=lambda e: (e.timestamp, e.event_id))

    # Idempotency: every event_id already linked to any stored observation
    # for this user (from any prior run, any link_role) is treated as
    # already processed. This is the primary defense against duplicates;
    # the deterministic observation_id derived below is the second,
    # independent one (see module docstring).
    existing_observations = repo.list_observations(user_id=user_id)
    existing_links = repo.list_observation_events_for(
        [observation.observation_id for observation in existing_observations]
    )
    already_linked_event_ids = {link.event_id for link in existing_links}

    pending: list[tuple[int, UserEvent, UserObservation, list[ObservationEvent]]] = []

    for event in events:
        if event.event_id in already_linked_event_ids:
            outcomes.append(
                EventProcessingOutcome(event.event_id, event.event_type, "skipped_already_processed")
            )
            continue

        draft = _draft_observation(event)
        if draft is None:
            reason = (
                "skipped_unsupported_type"
                if event.event_type not in _DRAFTERS
                else "skipped_malformed_data"
            )
            outcomes.append(EventProcessingOutcome(event.event_id, event.event_type, reason))
            continue

        observation_id = f"obs_{event.event_id}"
        observation, links = create_observation_from_event(
            event,
            observation_id=observation_id,
            category=draft.category,
            observation_text=draft.observation_text,
            importance=draft.importance,
            confidence=draft.confidence,
            created_at=as_of,
            **_VERSION_FIELDS,
        )
        # Placeholder at this index, filled in below once we know whether
        # (and how) each pending observation actually got written -- keeps
        # the outcome list in the same order as the events it describes,
        # regardless of dry-run vs. persist.
        pending.append((len(outcomes), event, observation, links))
        outcomes.append(EventProcessingOutcome(event.event_id, event.event_type, "pending"))

    for index, event, observation, links in pending:
        if not persist:
            outcomes[index] = EventProcessingOutcome(
                event.event_id, event.event_type, "would_create", observation.observation_id
            )
            continue
        try:
            repo.insert_observation(observation, links)
            outcomes[index] = EventProcessingOutcome(
                event.event_id, event.event_type, "created", observation.observation_id
            )
        except sqlite3.IntegrityError:
            # Another process (or an earlier, interrupted run) already wrote
            # this exact deterministic observation_id -- treat as already
            # processed, not a failure.
            outcomes[index] = EventProcessingOutcome(
                event.event_id, event.event_type, "skipped_already_processed", observation.observation_id
            )

    return ProcessEventsResult(user_id=user_id, persisted=persist, outcomes=outcomes)

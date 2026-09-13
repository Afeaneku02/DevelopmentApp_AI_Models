"""Deterministic pipeline step: turn stored, low-risk ``user_observations``
into conservative ``belief_evidence`` proposals (blueprint section 6,
create-belief-evidence's runtime slice -- see
``src.beliefs.propose_evidence`` for the sanctioned proposal builder and
``src.beliefs.models.authorize_evidence`` for the sanctioned
proposal-to-persisted-record constructor this module builds on).

This is a batch wrapper around ``propose_evidence_from_observation_validated``
and ``authorize_evidence``, not a new evidence mechanism: for each
observation it decides *whether* a safe, structured piece of evidence can be
drafted, and if so *what it says*, using nothing but a fixed, reviewed
mapping from an allowlisted observation ``category`` -- corroborated against
the observation's own linked source events, never the observation's prose
text -- to a fixed ``(belief_type, belief_key, context_key, direction,
strength)`` tuple. There is no LLM and no semantic judgment call at runtime.

Deliberately conservative in scope:

- **Allowlisted observation categories only** (``_DRAFTERS``), each
  corroborated against its own linked source event(s)' ``event_type`` and
  ``structured_data`` -- never the observation's free-form ``observation``
  text, which exists for human readability, not as this module's input.
  This also means an observation is handled by what its *underlying events*
  actually are, not merely by trusting whatever ``category`` string it was
  given: a mismatched or malformed observation is skipped, not guessed at
  (see ``_draft_check_in``/``_draft_roadmap_step_completed``).
- **Only two low-risk belief mappings, to start** (the task's own examples):
  repeated ``check_in`` observations toward a ``routine_or_preference``
  belief about check-in consistency (``checkin_consistency``), and
  ``goal_progress`` (roadmap-step-completion) observations toward a
  ``behavioral_tendency`` belief about following through on roadmap action
  steps (``completes_roadmap_action_steps``). Both are ordinary, low-stakes
  behavioral tendencies -- never a diagnosis, prediction, causal claim, or
  anything about health, finances, or emotional state. Extending the
  allowlist is a deliberate, reviewed code change.
- **``source_type=recorded_event`` always** -- every proposal here traces to
  a directly recorded structured event, never an inference
  (``model_observation``/``llm_inference`` would need extra justification
  this module has no basis for).
- **Every belief_id is deterministic**: ``f"bel_{user_id}_{belief_key}"``, so
  every observation that maps to the same belief_key always contributes
  evidence to the same belief, and reruns resolve to the same id rather than
  fragmenting evidence across accidental duplicates.
- **No recompute, unless a caller explicitly opts in** (``recompute=True``).
  By default, evidence is added and any affected belief is left
  ``locked_until_recompute`` (the same "evidence changed through a path
  other than invalidation" contract ``src.recommendations.promotion``
  already uses) -- this module never silently changes a belief's cached
  confidence/status itself.
- **Idempotent.** An observation that already produced evidence (from any
  prior run) is always skipped -- rerunning against the same observations,
  or a superset of them, creates no duplicate evidence. As a second,
  independent layer, the derived ``evidence_id``
  (``f"bev_{observation.observation_id}"``) is deterministic, so even a bug
  that let a duplicate slip past the first check would still hit a
  primary-key collision in storage rather than silently double-inserting.
- **Provenance always traces to real events.** ``source_event_ids`` is
  derived from the observation's own ``observation_events`` links by
  ``propose_evidence_from_observation_validated``, which also re-validates
  every link against the real, stored ``UserEvent`` rows before a proposal
  is even built -- an observation whose links do not check out is rejected,
  not silently trusted.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from src.beliefs.models import BeliefEvidence, authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.common.enums import BeliefType, Direction, SourceType
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)
AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"
MODEL_VERSION = "deterministic-observation-evidence-0.1"
AUTHORIZED_BY = "observation_evidence_pipeline"

_CHECK_IN_BELIEF_TYPE = BeliefType.ROUTINE_OR_PREFERENCE
_CHECK_IN_BELIEF_KEY = "checkin_consistency"
_CHECK_IN_CONTEXT_KEY = "engagement"
_CHECK_IN_BELIEF_VALUE = True

_ROADMAP_STEP_BELIEF_TYPE = BeliefType.BEHAVIORAL_TENDENCY
_ROADMAP_STEP_BELIEF_KEY = "completes_roadmap_action_steps"
_ROADMAP_STEP_CONTEXT_KEY = "goal_progress"
_ROADMAP_STEP_BELIEF_VALUE = True


@dataclass(frozen=True)
class EvidenceDraft:
    """What a drafter decided to claim about one observation, before any
    id/version/authorization fields are attached."""

    belief_type: BeliefType
    belief_key: str
    belief_value: Any
    context_key: str
    direction: Direction
    strength: float


def _draft_check_in(observation: UserObservation, source_events: list[UserEvent]) -> EvidenceDraft | None:
    # Corroborated against the linked source event(s), never the
    # observation's own prose text (see module docstring). Exactly one
    # matching check_in_recorded response is required: zero means this
    # observation's links do not actually support a check-in claim; more
    # than one is a multi-event observation this narrow mapper does not
    # know how to combine, so it declines rather than guessing.
    responses = {
        event.structured_data.get("response")
        for event in source_events
        if event.event_type == "check_in_recorded" and isinstance(event.structured_data, dict)
    }
    if len(responses) != 1:
        return None
    response = responses.pop()

    if response == "yes":
        direction, strength = Direction.SUPPORT, 0.7
    elif response == "partly":
        direction, strength = Direction.SUPPORT, 0.4
    elif response == "no":
        # An explicit, directly reported miss - low-risk to record as
        # contradicting evidence for the same consistency belief support
        # is recorded for, at the mirrored conservative strength.
        direction, strength = Direction.CONTRADICT, 0.5
    else:
        # "skipped" (or anything else) is genuinely ambiguous - it could be
        # a deliberate, reasonable day off rather than a lapse - so no
        # directional claim is made either way.
        return None

    return EvidenceDraft(
        belief_type=_CHECK_IN_BELIEF_TYPE,
        belief_key=_CHECK_IN_BELIEF_KEY,
        belief_value=_CHECK_IN_BELIEF_VALUE,
        context_key=_CHECK_IN_CONTEXT_KEY,
        direction=direction,
        strength=strength,
    )


def _draft_roadmap_step_completed(
    observation: UserObservation, source_events: list[UserEvent]
) -> EvidenceDraft | None:
    if not any(event.event_type == "roadmap_step_completed" for event in source_events):
        return None
    return EvidenceDraft(
        belief_type=_ROADMAP_STEP_BELIEF_TYPE,
        belief_key=_ROADMAP_STEP_BELIEF_KEY,
        belief_value=_ROADMAP_STEP_BELIEF_VALUE,
        context_key=_ROADMAP_STEP_CONTEXT_KEY,
        direction=Direction.SUPPORT,
        strength=0.6,
    )


# The reviewed allowlist, keyed by UserObservation.category - the same
# categories src.observations.process_events assigns, but this module does
# not import or depend on that one: any observation with a matching
# category and a genuinely matching linked event is handled the same way,
# regardless of what created it (see module docstring).
_DRAFTERS: dict[str, Callable[[UserObservation, list[UserEvent]], EvidenceDraft | None]] = {
    "check_in": _draft_check_in,
    "goal_progress": _draft_roadmap_step_completed,
}


# Action taken for one observation, always one of:
#   "created"                     - new belief_evidence was written (persist=True)
#   "would_create"                 - new belief_evidence would be written (persist=False)
#   "skipped_already_processed"   - a prior run (or a primary-key collision
#                                     caught at insert time) already covered
#                                     this observation
#   "skipped_unsupported_category" - observation.category is not on the
#                                     allowlist
#   "skipped_no_signal"            - category is allowlisted but the linked
#                                     source event(s) don't yield a confident
#                                     directional claim (e.g. a "skipped"
#                                     check-in, or zero/multiple matching
#                                     events)
#   "skipped_malformed_source"     - an observation_events link points to an
#                                     event that is not actually stored (a
#                                     store-level inconsistency)
#   "skipped_not_found"            - an explicitly requested observation_id
#                                     does not resolve to a stored
#                                     observation owned by this user
@dataclass(frozen=True)
class ObservationProcessingOutcome:
    observation_id: str
    category: str | None
    action: str
    evidence_id: str | None = None
    belief_id: str | None = None


@dataclass(frozen=True)
class ProcessObservationsResult:
    user_id: str
    persisted: bool
    recomputed: bool
    outcomes: list[ObservationProcessingOutcome]
    locked_belief_ids: list[str]
    recomputed_beliefs: list[dict[str, Any]]

    @property
    def created_evidence_ids(self) -> list[str]:
        action = "created" if self.persisted else "would_create"
        return [o.evidence_id for o in self.outcomes if o.action == action and o.evidence_id]


def _belief_id_for(user_id: str, belief_key: str) -> str:
    return f"bel_{user_id}_{belief_key}"


def process_user_observations(
    repo: Any,
    *,
    user_id: str,
    observation_ids: list[str] | None = None,
    as_of: datetime,
    persist: bool = False,
    recompute: bool = False,
) -> ProcessObservationsResult:
    """Draft (and, if ``persist``, write) belief_evidence for ``user_id``'s
    observations.

    ``observation_ids=None`` (the default) considers every stored
    observation for ``user_id`` -- in practice, only the ones without
    existing evidence do anything, since every already-covered observation
    is reported as ``skipped_already_processed``. Passing ``observation_ids``
    restricts consideration to exactly those ids; an id that is not a stored
    observation owned by ``user_id`` is reported as ``skipped_not_found``
    rather than raising, so one typo in a batch does not abort the rest.

    ``recompute`` only has any effect when ``persist=True`` (a dry run
    recomputes nothing, matching ``tools/promote_outcome_learning_signal.py``'s
    own ``--recompute requires --persist`` rule, enforced by the CLI). When
    evidence is persisted without ``recompute``, every belief that received
    new evidence this run is left ``locked_until_recompute`` instead.

    ``repo`` is only read from unless ``persist=True``. Never raises for
    anything about the observations themselves -- every observation gets an
    ``ObservationProcessingOutcome``, whatever it says.
    """
    outcomes: list[ObservationProcessingOutcome] = []
    observations: list[UserObservation] = []

    if observation_ids is not None:
        for observation_id in observation_ids:
            observation = repo.get_observation(observation_id)
            if observation is None or observation.user_id != user_id:
                outcomes.append(ObservationProcessingOutcome(observation_id, None, "skipped_not_found"))
                continue
            observations.append(observation)
    else:
        observations = repo.list_observations(user_id=user_id)

    observations = sorted(observations, key=lambda o: (o.created_at, o.observation_id))

    # Idempotency: every observation_id that already has a belief_evidence
    # row (active or inactive, from any prior run) is treated as already
    # processed. This is the primary defense against duplicates; the
    # deterministic evidence_id derived below is the second, independent
    # one (see module docstring). Only the belief_ids this module could
    # possibly have written to need checking - it never touches any other
    # belief_id.
    known_belief_ids = {_belief_id_for(user_id, _CHECK_IN_BELIEF_KEY), _belief_id_for(user_id, _ROADMAP_STEP_BELIEF_KEY)}
    already_evidenced_observation_ids: set[str] = set()
    for belief_id in known_belief_ids:
        for row in repo.list_evidence(user_id=user_id, belief_id=belief_id):
            if row.observation_id:
                already_evidenced_observation_ids.add(row.observation_id)

    all_links = repo.list_observation_events_for([o.observation_id for o in observations])
    links_by_observation: dict[str, list[ObservationEvent]] = {}
    for link in all_links:
        links_by_observation.setdefault(link.observation_id, []).append(link)

    pending: list[tuple[int, UserObservation, str, BeliefEvidence]] = []

    for observation in observations:
        if observation.observation_id in already_evidenced_observation_ids:
            outcomes.append(
                ObservationProcessingOutcome(observation.observation_id, observation.category, "skipped_already_processed")
            )
            continue

        drafter = _DRAFTERS.get(observation.category)
        if drafter is None:
            outcomes.append(
                ObservationProcessingOutcome(observation.observation_id, observation.category, "skipped_unsupported_category")
            )
            continue

        own_links = links_by_observation.get(observation.observation_id, [])
        source_event_ids = sorted({link.event_id for link in own_links})
        source_events = [event for event in (repo.get_event(eid) for eid in source_event_ids) if event is not None]
        if len(source_events) != len(source_event_ids):
            outcomes.append(
                ObservationProcessingOutcome(observation.observation_id, observation.category, "skipped_malformed_source")
            )
            continue

        draft = drafter(observation, source_events)
        if draft is None:
            outcomes.append(
                ObservationProcessingOutcome(observation.observation_id, observation.category, "skipped_no_signal")
            )
            continue

        belief_id = _belief_id_for(user_id, draft.belief_key)
        proposal = propose_evidence_from_observation_validated(
            observation,
            own_links,
            source_events,
            belief_id=belief_id,
            direction=draft.direction,
            source_type=SourceType.RECORDED_EVENT,
            context_key=draft.context_key,
            strength=draft.strength,
            belief_type=draft.belief_type,
            model_version=MODEL_VERSION,
            **_VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal,
            evidence_id=f"bev_{observation.observation_id}",
            created_at=as_of,
            aggregation_policy_version=AGGREGATION_POLICY_VERSION,
            aggregation_authorized_by=AUTHORIZED_BY,
        )
        # Placeholder at this index, filled in below once we know whether
        # (and how) each pending evidence row actually got written - keeps
        # the outcome list in the same order as the observations it
        # describes, regardless of dry-run vs. persist.
        pending.append((len(outcomes), observation, belief_id, evidence))
        outcomes.append(ObservationProcessingOutcome(observation.observation_id, observation.category, "pending"))

    newly_inserted_belief_ids: set[str] = set()

    for index, observation, belief_id, evidence in pending:
        if not persist:
            outcomes[index] = ObservationProcessingOutcome(
                observation.observation_id, observation.category, "would_create", evidence.evidence_id, belief_id
            )
            continue
        try:
            repo.insert_evidence(evidence)
            newly_inserted_belief_ids.add(belief_id)
            outcomes[index] = ObservationProcessingOutcome(
                observation.observation_id, observation.category, "created", evidence.evidence_id, belief_id
            )
        except sqlite3.IntegrityError:
            # Another process (or an earlier, interrupted run) already
            # wrote this exact deterministic evidence_id - treat as already
            # processed, not a failure.
            outcomes[index] = ObservationProcessingOutcome(
                observation.observation_id, observation.category, "skipped_already_processed",
                evidence.evidence_id, belief_id,
            )

    if not persist or not newly_inserted_belief_ids:
        return ProcessObservationsResult(
            user_id=user_id, persisted=persist, recomputed=False, outcomes=outcomes,
            locked_belief_ids=[], recomputed_beliefs=[],
        )

    locked_belief_ids: list[str] = []
    recomputed_beliefs: list[dict[str, Any]] = []

    if recompute:
        belief_specs = {
            _belief_id_for(user_id, _CHECK_IN_BELIEF_KEY): (
                _CHECK_IN_BELIEF_TYPE, _CHECK_IN_BELIEF_KEY, _CHECK_IN_BELIEF_VALUE,
            ),
            _belief_id_for(user_id, _ROADMAP_STEP_BELIEF_KEY): (
                _ROADMAP_STEP_BELIEF_TYPE, _ROADMAP_STEP_BELIEF_KEY, _ROADMAP_STEP_BELIEF_VALUE,
            ),
        }
        for belief_id in sorted(newly_inserted_belief_ids):
            belief_type, belief_key, belief_value = belief_specs[belief_id]
            recomputed_beliefs.append(
                _recompute_one(
                    repo, user_id=user_id, belief_id=belief_id, belief_type=belief_type,
                    belief_key=belief_key, belief_value=belief_value, as_of=as_of,
                )
            )
    else:
        for belief_id in sorted(newly_inserted_belief_ids):
            if repo.lock_belief_until_recompute(user_id=user_id, belief_id=belief_id):
                locked_belief_ids.append(belief_id)

    return ProcessObservationsResult(
        user_id=user_id, persisted=True, recomputed=bool(recomputed_beliefs), outcomes=outcomes,
        locked_belief_ids=locked_belief_ids, recomputed_beliefs=recomputed_beliefs,
    )


def _recompute_one(
    repo: Any, *, user_id: str, belief_id: str, belief_type: BeliefType, belief_key: str,
    belief_value: Any, as_of: datetime,
) -> dict[str, Any]:
    latest = repo.get_latest_belief(user_id=user_id, belief_id=belief_id)
    active = repo.list_active_evidence(user_id=user_id, belief_id=belief_id)
    first_observed = min(
        (row.observed_at for row in active),
        default=(latest.first_observed if latest is not None else as_of),
    )
    belief = recompute_belief(
        belief_id=belief_id, user_id=user_id, belief_type=belief_type, belief_key=belief_key,
        belief_value=belief_value, evidence=active, as_of=as_of, first_observed=first_observed,
        **_VERSION_FIELDS,
    )
    repo.save_belief(belief)
    return {
        "belief_id": belief_id,
        "recomputed": True,
        "confidence": belief.confidence,
        "status": belief.status.value,
        "locked_until_recompute": belief.locked_until_recompute,
    }

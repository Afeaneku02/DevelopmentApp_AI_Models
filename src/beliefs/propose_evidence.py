"""Runtime pipeline: build a BeliefEvidenceProposal from a UserEvent or a
UserObservation. This is create-belief-evidence's runtime implementation --
the deterministic step that turns "this happened" into "here is a candidate
belief_evidence row," per skills/shared/create-belief-evidence/SKILL.md.

Deliberately narrow scope (this is the first runtime slice over the Phase 1
contracts, not the whole belief engine):

- Provenance derivation (source_event_ids, event_id) from the actual event(s)
  behind the evidence being proposed.
- source_reliability defaulting from the canonical registry, reusing exactly
  the deviation-justification contract BeliefEvidenceProposal itself already
  enforces -- this module does not duplicate that validation, it just
  supplies a sensible default and lets the model validate the result.
- decay_lambda defaulting from the belief_type registry when a caller
  supplies belief_type instead of an explicit value.
- Always produces a *proposal* and nothing more. Deciding the belief_id a
  proposal is attached to, choosing source_type/direction/strength, and
  authorizing aggregation are all deliberately left to the caller (a skill,
  an extractor, or -- for authorization -- authorize_evidence() in
  src/beliefs/models.py) rather than guessed here.

NOT built here, on purpose: belief-key canonicalization (canonicalize-belief-
key, not yet implemented), any database/persistence layer, the recommendation
engine, or deletion/reset cascade execution (recompute-user-model, not yet
implemented).
"""
from __future__ import annotations

from datetime import datetime

from src.beliefs.models import BeliefEvidenceProposal
from src.common.enums import AggregationMode, BeliefType, Direction, LinkRole, SourceType
from src.common.provenance import validate_observation_provenance
from src.common.registry import BELIEF_TYPE_REGISTRY, SOURCE_TYPE_RELIABILITY
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation


def _resolve_decay_lambda(decay_lambda: float | None, belief_type: BeliefType | None) -> float:
    if decay_lambda is not None:
        return decay_lambda
    if belief_type is not None:
        return BELIEF_TYPE_REGISTRY[belief_type].default_decay_lambda
    raise ValueError(
        "decay_lambda must be supplied explicitly, or belief_type must be given so it can be "
        "defaulted from the belief_type_registry"
    )


def _build_proposal(
    *,
    user_id: str,
    belief_id: str,
    direction: Direction,
    event_id: str | None,
    observation_id: str | None,
    source_event_ids: list[str],
    observed_at: datetime,
    source_type: SourceType,
    context_key: str,
    strength: float,
    source_reliability: float | None,
    reliability_deviation_reason: str | None,
    decay_lambda: float | None,
    belief_type: BeliefType | None,
    model_version: str,
    prompt_version: str | None,
    proposed_aggregation_mode: AggregationMode,
    replaces_evidence_ids: list[str] | None,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
) -> BeliefEvidenceProposal:
    resolved_reliability = (
        source_reliability if source_reliability is not None else SOURCE_TYPE_RELIABILITY[source_type]
    )
    return BeliefEvidenceProposal(
        belief_id=belief_id,
        user_id=user_id,
        direction=direction,
        event_id=event_id,
        observation_id=observation_id,
        source_event_ids=source_event_ids,
        proposed_aggregation_mode=proposed_aggregation_mode,
        replaces_evidence_ids=replaces_evidence_ids or [],
        source_type=source_type,
        context_key=context_key,
        strength=strength,
        source_reliability=resolved_reliability,
        reliability_deviation_reason=reliability_deviation_reason,
        observed_at=observed_at,
        decay_lambda=_resolve_decay_lambda(decay_lambda, belief_type),
        model_version=model_version,
        prompt_version=prompt_version,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )


def propose_evidence_from_event(
    event: UserEvent,
    *,
    belief_id: str,
    direction: Direction,
    source_type: SourceType,
    context_key: str,
    strength: float,
    model_version: str,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
    prompt_version: str | None = None,
    source_reliability: float | None = None,
    reliability_deviation_reason: str | None = None,
    decay_lambda: float | None = None,
    belief_type: BeliefType | None = None,
    proposed_aggregation_mode: AggregationMode = AggregationMode.LEAF_DEFAULT,
    replaces_evidence_ids: list[str] | None = None,
) -> BeliefEvidenceProposal:
    """Propose belief_evidence directly from a single UserEvent.

    ``source_event_ids`` is always exactly ``[event.event_id]``: a proposal
    built from one event has exactly one event behind it, so ``event_id`` can
    safely be set to the same id (trivially satisfying
    ``BeliefEvidenceProposal``'s own event_id-must-be-covered-by-
    source_event_ids guard).

    ``source_reliability``, when not supplied, defaults to the canonical
    registry value for ``source_type`` (blueprint section 6.2.1); passing an
    explicit value that deviates beyond tolerance without
    ``reliability_deviation_reason`` still fails, because that check lives on
    ``BeliefEvidenceProposal`` itself and this function does not bypass it.
    """
    return _build_proposal(
        user_id=event.user_id,
        belief_id=belief_id,
        direction=direction,
        event_id=event.event_id,
        observation_id=None,
        source_event_ids=[event.event_id],
        observed_at=event.timestamp,
        source_type=source_type,
        context_key=context_key,
        strength=strength,
        source_reliability=source_reliability,
        reliability_deviation_reason=reliability_deviation_reason,
        decay_lambda=decay_lambda,
        belief_type=belief_type,
        model_version=model_version,
        prompt_version=prompt_version,
        proposed_aggregation_mode=proposed_aggregation_mode,
        replaces_evidence_ids=replaces_evidence_ids,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )


def propose_evidence_from_observation_validated(
    observation: UserObservation,
    observation_events: list[ObservationEvent],
    source_events: list[UserEvent],
    *,
    belief_id: str,
    direction: Direction,
    source_type: SourceType,
    context_key: str,
    strength: float,
    model_version: str,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
    prompt_version: str | None = None,
    source_reliability: float | None = None,
    reliability_deviation_reason: str | None = None,
    decay_lambda: float | None = None,
    belief_type: BeliefType | None = None,
    proposed_aggregation_mode: AggregationMode = AggregationMode.LEAF_DEFAULT,
    replaces_evidence_ids: list[str] | None = None,
) -> BeliefEvidenceProposal:
    """Cross-record-integrity-checked wrapper around
    ``propose_evidence_from_observation()``.

    ``propose_evidence_from_observation()`` itself is deliberately pure and
    narrow: given an observation and its links, it trusts that every linked
    event genuinely exists and belongs to the same user, and derives
    ``source_event_ids`` accordingly. That trust is exactly the gap this
    wrapper closes -- it additionally requires ``source_events`` (the actual
    ``UserEvent`` rows the links claim to reference) and runs
    ``validate_observation_provenance()`` against
    ``[observation], observation_events, source_events`` *before* calling the
    underlying function. If validation finds anything wrong -- a link to a
    missing event, a link to another user's event, or no primary link at all
    -- this raises ``ValueError`` with every problem found, and
    ``propose_evidence_from_observation()`` is never called.

    This is the only place in the pipeline that checks cross-record
    integrity against real event data; the lower-level function stays pure
    and unaware of any validation layer, so it remains directly composable
    and testable on its own (see ``tests/beliefs/test_propose_evidence.py``).
    """
    errors = validate_observation_provenance([observation], observation_events, source_events)
    if errors:
        raise ValueError(
            f"cannot propose evidence from observation {observation.observation_id!r}: " + "; ".join(errors)
        )
    return propose_evidence_from_observation(
        observation,
        observation_events,
        belief_id=belief_id,
        direction=direction,
        source_type=source_type,
        context_key=context_key,
        strength=strength,
        model_version=model_version,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
        prompt_version=prompt_version,
        source_reliability=source_reliability,
        reliability_deviation_reason=reliability_deviation_reason,
        decay_lambda=decay_lambda,
        belief_type=belief_type,
        proposed_aggregation_mode=proposed_aggregation_mode,
        replaces_evidence_ids=replaces_evidence_ids,
    )


def propose_evidence_from_observation(
    observation: UserObservation,
    observation_events: list[ObservationEvent],
    *,
    belief_id: str,
    direction: Direction,
    source_type: SourceType,
    context_key: str,
    strength: float,
    model_version: str,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
    prompt_version: str | None = None,
    source_reliability: float | None = None,
    reliability_deviation_reason: str | None = None,
    decay_lambda: float | None = None,
    belief_type: BeliefType | None = None,
    proposed_aggregation_mode: AggregationMode = AggregationMode.LEAF_DEFAULT,
    replaces_evidence_ids: list[str] | None = None,
) -> BeliefEvidenceProposal:
    """Propose belief_evidence from a UserObservation, deriving provenance
    from its ``observation_events`` links rather than any ad hoc field on the
    observation itself -- ``UserObservation`` deliberately has no event_id or
    event_ids field at all (blueprint section 18 Definition of Done).

    ``source_event_ids`` is every event_id linked to this observation, both
    ``primary`` and ``supporting``: the observation's own justification rests
    on all of them, not only the single designated primary one, so all are
    genuine provenance for evidence built from it. This is a deliberate
    choice beyond what the blueprint's own illustrative example shows (which
    only carries the primary event forward for brevity) -- full traceable
    provenance is safer and matches section 6's "populate source_event_ids
    with the real, traceable event ID(s)" (plural) more literally.

    ``event_id`` is set to the primary-linked event only when there is
    exactly one; left null when there are zero (raises instead, see below) or
    multiple primary links, since a single ad hoc pointer would be either
    meaningless or misleading in the multi-primary case -- ``source_event_ids``
    remains correct and complete regardless of what ``event_id`` is set to.

    Raises ``ValueError`` if the observation has no linked events at all, or
    no ``primary`` link -- extract-user-observations's own completion check
    already requires every observation to have a primary link before
    treating extraction as done, so an observation missing one here indicates
    upstream state that should not be feeding the belief engine yet.
    """
    own_links = [link for link in observation_events if link.observation_id == observation.observation_id]
    if not own_links:
        raise ValueError(
            f"observation {observation.observation_id!r} has no observation_events links; "
            "cannot derive source_event_ids with no provenance to draw from"
        )
    event_ids = sorted({link.event_id for link in own_links})
    primary_event_ids = sorted({link.event_id for link in own_links if link.link_role == LinkRole.PRIMARY})
    if not primary_event_ids:
        raise ValueError(
            f"observation {observation.observation_id!r} has no primary observation_events link; "
            "extract-user-observations requires one before evidence can be proposed from it"
        )
    resolved_event_id = primary_event_ids[0] if len(primary_event_ids) == 1 else None

    return _build_proposal(
        user_id=observation.user_id,
        belief_id=belief_id,
        direction=direction,
        event_id=resolved_event_id,
        observation_id=observation.observation_id,
        source_event_ids=event_ids,
        observed_at=observation.created_at,
        source_type=source_type,
        context_key=context_key,
        strength=strength,
        source_reliability=source_reliability,
        reliability_deviation_reason=reliability_deviation_reason,
        decay_lambda=decay_lambda,
        belief_type=belief_type,
        model_version=model_version,
        prompt_version=prompt_version,
        proposed_aggregation_mode=proposed_aggregation_mode,
        replaces_evidence_ids=replaces_evidence_ids,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )

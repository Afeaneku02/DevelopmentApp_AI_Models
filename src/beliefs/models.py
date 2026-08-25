"""belief_evidence and user_beliefs (blueprint section 5, section 6).

The model-proposes/backend-authorizes boundary (section 6.1.2, section 22
Decision Log) is enforced structurally, not just by convention:

- ``BeliefEvidenceProposal`` is the complete set of fields a skill, extractor,
  or LLM output may set. It has no aggregation-authorization, suppression, or
  active/invalidation fields at all.
- ``BeliefEvidence`` extends it with exactly the backend-owned fields. It can
  only be constructed by hand (e.g. when deserializing an existing persisted
  record) or via ``authorize_evidence()`` below.
- ``authorize_evidence()`` is the single sanctioned path from a proposal to a
  persisted record, and it is the only place ``authorized_aggregation_mode``
  is allowed to become anything other than ``leaf_default``.

Because ``BeliefEvidenceProposal`` uses ``extra="forbid"`` (via
``VersionedModel``), constructing one from a dict that contains
``authorized_aggregation_mode`` or any other backend-owned field raises a
``pydantic.ValidationError`` immediately -- this is the deterministic test in
``tests/beliefs/test_belief_evidence_authorization.py``.

Every contract-governed numeric field (``strength``, ``source_reliability``,
``decay_lambda`` on the proposal; ``confidence`` and every evidence-count
field on ``UserBelief``) uses ``strict=True``. Pydantic's default lax mode
would otherwise coerce ``True``/``False`` into ``1.0``/``0.0`` (bool is a
subclass of int in Python) and coerce a numeric string like ``"0.95"`` into
a real float, both of which are real JSON-type violations a proposal or a
persisted record should never get away with silently. Strict mode still
accepts a plain JSON int for a float field (e.g. ``1`` for ``strength``),
since that is a genuine number -- only bools and strings are rejected.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from src.common.enums import (
    AggregationMode,
    AggregationReviewStatus,
    BeliefStatus,
    BeliefType,
    Direction,
    PersistencePolicy,
    SensitivityClass,
    SourceType,
)
from src.common.registry import SOURCE_RELIABILITY_TOLERANCE, SOURCE_TYPE_RELIABILITY
from src.common.versioning import VersionedModel


class BeliefEvidenceProposal(VersionedModel):
    """Fields a skill/extractor may propose (blueprint section 5, belief_evidence,
    minus every field marked backend-owned in that section).

    ``event_id`` is a nullable *direct pointer* the blueprint keeps for
    convenience/readability (section 5: "event_id # nullable direct event
    pointer"), but ``source_event_ids`` is the sole *authoritative* provenance
    field (section 6.1.1). To prevent the two from silently diverging, this
    model requires that whenever ``event_id`` is set, it also appears in
    ``source_event_ids`` -- evidence may never rely on ``event_id`` alone.

    ``source_reliability`` is similarly constrained rather than left free:
    the create-belief-evidence skill's own completion check requires it to
    match (or explicitly justify a deviation from) the versioned default in
    ``SOURCE_TYPE_RELIABILITY`` for the chosen ``source_type``. A value more
    than ``SOURCE_RELIABILITY_TOLERANCE`` away from that default requires a
    non-blank ``reliability_deviation_reason``; an unjustified deviation is
    rejected rather than silently accepted, since an inflated or deflated
    reliability directly changes downstream confidence math.
    """

    belief_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    direction: Direction
    event_id: str | None = None
    observation_id: str | None = None
    source_event_ids: list[str] = Field(min_length=1)
    proposed_aggregation_mode: AggregationMode = AggregationMode.LEAF_DEFAULT
    replaces_evidence_ids: list[str] = Field(default_factory=list)
    source_type: SourceType
    context_key: str = Field(min_length=1)
    strength: float = Field(ge=0.0, le=1.0, strict=True)
    source_reliability: float = Field(ge=0.0, le=1.0, strict=True)
    reliability_deviation_reason: str | None = None
    observed_at: datetime
    decay_lambda: float = Field(ge=0.0, strict=True)
    model_version: str = Field(min_length=1)
    prompt_version: str | None = None

    @field_validator("source_event_ids")
    @classmethod
    def _no_blank_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("source_event_ids must not contain blank entries")
        return value

    @model_validator(mode="after")
    def _event_id_must_be_covered_by_source_event_ids(self) -> "BeliefEvidenceProposal":
        if self.event_id is not None and self.event_id not in self.source_event_ids:
            raise ValueError(
                "event_id, when present, must also appear in source_event_ids "
                f"(got event_id={self.event_id!r}, source_event_ids={self.source_event_ids!r}); "
                "source_event_ids is the sole authoritative provenance field and evidence "
                "must never rely on event_id alone"
            )
        return self

    @model_validator(mode="after")
    def _reliability_matches_default_or_is_justified(self) -> "BeliefEvidenceProposal":
        expected = SOURCE_TYPE_RELIABILITY[self.source_type]
        deviation = abs(self.source_reliability - expected)
        # A tiny epsilon absorbs float-precision noise at the tolerance
        # boundary itself (e.g. 1.0 - 0.95 == 0.050000000000000044 in binary
        # floating point, not exactly 0.05), so a value landing exactly on
        # the boundary is not spuriously rejected as "beyond tolerance."
        if deviation > SOURCE_RELIABILITY_TOLERANCE + 1e-9 and not (
            self.reliability_deviation_reason and self.reliability_deviation_reason.strip()
        ):
            raise ValueError(
                f"source_reliability {self.source_reliability!r} deviates from the versioned "
                f"default {expected!r} for source_type {self.source_type.value!r} by {deviation:.2f}, "
                f"more than the {SOURCE_RELIABILITY_TOLERANCE} tolerance; either use the default or "
                "set reliability_deviation_reason to justify the override"
            )
        return self


class BeliefEvidence(BeliefEvidenceProposal):
    """The full persisted belief_evidence record: the proposal fields plus
    every backend-owned field. Do not construct this directly from untrusted
    input -- use ``authorize_evidence()``."""

    evidence_id: str = Field(min_length=1)
    created_at: datetime
    independence_group: str = Field(min_length=1)

    authorized_aggregation_mode: AggregationMode = AggregationMode.LEAF_DEFAULT
    aggregation_authorized_by: str | None = None
    aggregation_authorized_at: datetime | None = None
    aggregation_policy_version: str | None = None
    aggregation_review_required: bool = False
    aggregation_review_status: AggregationReviewStatus = AggregationReviewStatus.NOT_REQUIRED

    replaces_evidence_ids: list[str] = Field(default_factory=list)

    is_duplicate_suppressed: bool = False
    suppression_reason: str | None = None
    is_active: bool = True
    invalidated_at: datetime | None = None
    invalidation_reason: str | None = None


def _validate_replacement(
    proposal: BeliefEvidenceProposal, replaced_evidence: dict[str, "BeliefEvidence"]
) -> list[str]:
    """Deterministic checks this helper can make when the caller supplies the
    rows a replacement proposal claims to replace: that they exist, belong to
    the same user/belief, and that the replacement's source_event_ids covers
    everything the rows it replaces were already covering (so a replacement
    can compact evidence but can never silently narrow its provenance).

    This is intentionally narrow. It is not the full aggregation-policy
    engine the blueprint describes (section 6.1.2) -- it cannot, for example,
    confirm the replacement is semantically non-duplicative in the way a real
    dedup/independence-grouping pass over the whole ledger could. Treat an
    empty return as "nothing checkable was found wrong," not as "a policy
    engine approved this."
    """
    errors: list[str] = []
    missing_ids = set(proposal.replaces_evidence_ids) - set(replaced_evidence)
    if missing_ids:
        errors.append(
            f"replaces_evidence_ids references rows not supplied for validation: {sorted(missing_ids)}"
        )

    covered_source_event_ids: set[str] = set()
    for evidence_id in proposal.replaces_evidence_ids:
        row = replaced_evidence.get(evidence_id)
        if row is None:
            continue
        if row.user_id != proposal.user_id:
            errors.append(
                f"replaced row {evidence_id!r} belongs to a different user_id "
                f"({row.user_id!r} != {proposal.user_id!r})"
            )
        if row.belief_id != proposal.belief_id:
            errors.append(
                f"replaced row {evidence_id!r} belongs to a different belief_id "
                f"({row.belief_id!r} != {proposal.belief_id!r})"
            )
        covered_source_event_ids.update(row.source_event_ids)

    missing_coverage = covered_source_event_ids - set(proposal.source_event_ids)
    if missing_coverage:
        errors.append(
            "replacement does not cover all source_event_ids of the rows it replaces "
            f"(missing from the proposal's own source_event_ids: {sorted(missing_coverage)})"
        )
    return errors


def authorize_evidence(
    proposal: BeliefEvidenceProposal,
    *,
    evidence_id: str,
    created_at: datetime,
    aggregation_policy_version: str,
    aggregation_authorized_by: str = "evidence_policy",
    independence_group: str | None = None,
    backend_validation_passed: bool = False,
    replaced_evidence: dict[str, "BeliefEvidence"] | None = None,
) -> BeliefEvidence:
    """The only sanctioned constructor from a proposal to a persisted
    BeliefEvidence record (blueprint section 6.1.2).

    This function is a safe *constructor*, not the aggregation-policy engine
    itself. It never grants ``aggregate_replacement`` on its own judgment:

    - ``backend_validation_passed`` must be explicitly set True by a caller
      standing in for real backend aggregation-policy code -- never by an
      extractor asserting its own proposal is fine. The name is deliberately
      literal about what it means (validation happened, and passed), not a
      vague "approve" flag that invites being set reflexively.
    - When ``replaced_evidence`` (a mapping of evidence_id -> the existing
      ``BeliefEvidence`` rows named in ``replaces_evidence_ids``) is supplied,
      this function *also* runs the deterministic checks in
      ``_validate_replacement`` and refuses replacement if any fail, even if
      ``backend_validation_passed=True`` -- a caller cannot assert past
      verifiable data.

    Any proposal that does not clear this bar remains ``leaf_default``, per
    the blueprint's "leaf_default is the safe fallback; human review is
    optional because the safe behavior is to avoid suppression/replacement"
    decision (section 22).

    INTERIM SCHEMA-LAYER BOUNDARY, SCAFFOLD-ONLY -- read before wiring a real
    backend to this function: when ``replaced_evidence`` is omitted, this
    function has no way to check anything and falls back to trusting
    ``backend_validation_passed`` at face value. That is acceptable for this
    module's own tests and for exploratory/offline use, but it is not a real
    authorization guarantee -- it is exactly as trustworthy as whatever
    already validated the claim before calling this function, which is
    nothing, unless the caller did real work.

    TODO(recompute-user-model / backend aggregation-policy, not yet
    implemented): once a real backend policy component exists, it should
    call this function with one of (a) ``replaced_evidence`` populated from
    the actual ledger, or (b) a typed backend-policy decision artifact
    (e.g. a future ``AggregationPolicyDecision`` result carrying its own
    provenance/authority, rather than a bare bool) in place of
    ``backend_validation_passed``. Until then,
    ``test_bare_backend_validation_passed_without_replaced_evidence_is_scaffold_only_not_production_safe``
    in ``tests/beliefs/test_belief_evidence_authorization.py`` pins down
    today's permissive scaffold behavior; if that test starts failing after a
    change here, update it deliberately -- it means this boundary moved, and
    the docstring/decision log should move with it, not silently.
    """
    if independence_group is None:
        independence_group = (
            proposal.source_event_ids[0]
            if len(proposal.source_event_ids) == 1
            else "|".join(sorted(proposal.source_event_ids))
        )

    wants_replacement = (
        proposal.proposed_aggregation_mode == AggregationMode.AGGREGATE_REPLACEMENT
        and bool(proposal.replaces_evidence_ids)
    )

    replacement_errors: list[str] = []
    if wants_replacement and replaced_evidence is not None:
        replacement_errors = _validate_replacement(proposal, replaced_evidence)

    grant_replacement = wants_replacement and backend_validation_passed and not replacement_errors

    if grant_replacement:
        authorized_mode = AggregationMode.AGGREGATE_REPLACEMENT
        review_status = AggregationReviewStatus.APPROVED
        review_required = False
    elif wants_replacement and replacement_errors:
        # We have concrete proof this replacement is invalid: reject outright
        # rather than merely leaving it pending, regardless of the caller's
        # backend_validation_passed claim.
        authorized_mode = AggregationMode.LEAF_DEFAULT
        review_status = AggregationReviewStatus.REJECTED
        review_required = False
    elif wants_replacement:
        # Uncertain/unapproved aggregate proposal: remains leaf_default and
        # is flagged for review rather than silently granted or discarded.
        authorized_mode = AggregationMode.LEAF_DEFAULT
        review_status = AggregationReviewStatus.PENDING
        review_required = True
    else:
        authorized_mode = AggregationMode.LEAF_DEFAULT
        review_status = AggregationReviewStatus.NOT_REQUIRED
        review_required = False

    return BeliefEvidence(
        **proposal.model_dump(),
        evidence_id=evidence_id,
        created_at=created_at,
        independence_group=independence_group,
        authorized_aggregation_mode=authorized_mode,
        aggregation_authorized_by=aggregation_authorized_by,
        aggregation_authorized_at=created_at,
        aggregation_policy_version=aggregation_policy_version,
        aggregation_review_required=review_required,
        aggregation_review_status=review_status,
        is_duplicate_suppressed=False,
        suppression_reason=None,
        is_active=True,
        invalidated_at=None,
        invalidation_reason=None,
    )


def invalidate_evidence(evidence: BeliefEvidence, *, reason: str, invalidated_at: datetime) -> BeliefEvidence:
    """Mark a persisted BeliefEvidence row inactive (blueprint section 6.0.1:
    ``is_active`` becomes false after deletion/reset/policy invalidation).

    This never deletes the row -- it remains for audit -- it only flips the
    fields that ``active_evidence()`` filters on. Actually recomputing the
    owning belief after invalidation is out of scope here; that workflow
    belongs to the planned ``recompute-user-model`` skill, which will call
    this and then ``src.beliefs.scoring.compute_confidence`` over whatever
    ``active_evidence()`` returns.
    """
    return evidence.model_copy(
        update={"is_active": False, "invalidated_at": invalidated_at, "invalidation_reason": reason}
    )


def active_evidence(evidence_items: list[BeliefEvidence]) -> list[BeliefEvidence]:
    """Active evidence = ``is_active=True`` and ``is_duplicate_suppressed=False``
    (blueprint section 6.0.1, verbatim definition)."""
    return [item for item in evidence_items if item.is_active and not item.is_duplicate_suppressed]


class UserBelief(VersionedModel):
    """Flexible AI-learned knowledge about an individual (blueprint section 5).

    ``supporting_evidence_count`` through ``evidence_against`` are
    derived/cache fields only: the authoritative source is the
    ``belief_evidence`` ledger, and these must be reproducible from it under
    the recorded ``scoring_version`` (section 5, section 18). This class does
    not enforce that reproducibility itself -- ``update-user-beliefs``'s
    ``compute_confidence()`` (bridged in ``src/beliefs/scoring.py``) is the
    only sanctioned source for these values; treat direct construction with
    arbitrary counts as valid only for deserializing an already-computed
    record, not for hand-authoring one.
    """

    belief_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    belief_type: BeliefType
    belief_type_registry_version: str = Field(min_length=1)
    belief_key: str = Field(min_length=1)
    belief_value: Any

    confidence: float = Field(ge=0.0, le=0.98, strict=True)
    supporting_evidence_count: int = Field(ge=0, strict=True)
    contradicting_evidence_count: int = Field(ge=0, strict=True)
    total_evidence_count: int = Field(ge=0, strict=True)
    effective_support_count: float = Field(ge=0.0, strict=True)
    effective_evidence_count: float = Field(ge=0.0, strict=True)
    evidence_for: int = Field(ge=0, strict=True)
    evidence_against: int = Field(ge=0, strict=True)

    allowed_contexts: list[str] = Field(default_factory=list)
    disallowed_contexts: list[str] = Field(default_factory=list)
    sensitivity_class: SensitivityClass
    persistence_policy: PersistencePolicy

    first_observed: datetime
    last_validated: datetime
    status: BeliefStatus
    reasoning_summary: str | None = None

    locked_until_recompute: bool = False
    last_recompute_attempt_id: str | None = None
    last_successful_recompute_at: datetime | None = None

    @field_validator("confidence")
    @classmethod
    def _confidence_band(cls, value: float) -> float:
        # Section 3.4.1: confidence is either exactly 0.0 (no evidence / the
        # no-active-evidence-after-invalidation branch) or clamped to
        # [0.02, 0.98]. Never a value inside (0, 0.02) -- that band does not
        # correspond to any code path in the scoring algorithm -- and never
        # 1.0, which the 0.98 ceiling exists specifically to prevent.
        if value == 0.0:
            return value
        if not (0.02 <= value <= 0.98):
            raise ValueError(
                "confidence must be exactly 0.0 (no evidence) or within [0.02, 0.98]; "
                f"got {value!r}"
            )
        return value

    @field_validator("total_evidence_count")
    @classmethod
    def _total_matches_parts(cls, value: int, info) -> int:
        supporting = info.data.get("supporting_evidence_count")
        contradicting = info.data.get("contradicting_evidence_count")
        if supporting is not None and contradicting is not None and value != supporting + contradicting:
            raise ValueError(
                "total_evidence_count must equal supporting_evidence_count + "
                f"contradicting_evidence_count (got {value}, expected "
                f"{supporting + contradicting})"
            )
        return value

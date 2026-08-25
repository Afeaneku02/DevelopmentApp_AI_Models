"""Runtime pipeline: recompute a UserBelief from its active belief_evidence
ledger -- update-user-beliefs's runtime implementation.

Bridges ``src/beliefs/scoring.py``'s ``compute_confidence()``/
``determine_status()`` (already the skill's own reference implementation,
imported from ``skills/shared/update-user-beliefs/scripts/compute_confidence.py``
rather than re-derived) to a concrete ``UserBelief``, adding only:

- Active-evidence filtering via ``active_evidence()`` in
  ``src/beliefs/models.py`` (the same "is_active and not
  is_duplicate_suppressed" definition, blueprint section 6.0.1), applied
  before anything is handed to the scoring function.
- Fail-closed validation of that active evidence before it reaches the
  scoring function at all: every active row must actually belong to the
  belief/user being recomputed (a foreign ``belief_id``/``user_id`` would
  otherwise silently corrupt confidence rather than error), and none may be
  dated after ``as_of`` (which would otherwise produce a negative
  ``age_days`` -- decay > 1 -- and inflate confidence instead of raising).
- Raw/effective count computation, independent of which of
  ``compute_confidence()``'s three branches fired -- its two early-return
  branches intentionally omit those fields (there is no evidence to count),
  so this module computes them once, uniformly, from the same active,
  deduplicated evidence list the scoring function itself saw.
- belief_type-registry defaulting for ``expected_source_types``,
  ``sensitivity_class``, and ``persistence_policy``.
- Constructing a ``UserBelief`` that always passes its own confidence-band
  and total-evidence-count-consistency validators by construction, since the
  values are derived the same way in both places.

Deliberately narrow: this is a pure, deterministic transform from
"caller-supplied evidence list + as_of timestamp" to "one UserBelief." It
does not read from or write to anything itself -- there is no cache for a
stale value to hide in, so passing an evidence list with zero active rows
always produces confidence 0.0, never a leftover prior value.

On successful completion this function always represents blueprint section
6.0.2's *success* path: ``locked_until_recompute`` is cleared and
``last_successful_recompute_at`` is set to ``as_of``. Fail-closed locking for
a recompute attempt that itself fails (raises, times out) is a caller/
orchestration concern -- there is no partial or stale ``UserBelief`` this
function can return, only a complete one or an exception. Tracking an actual
failed attempt belongs on ``UserModelDeletionRequest``/``UserModelResetEvent``
(``src/common/deletion.py``), not here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.beliefs.models import BeliefEvidence, UserBelief, active_evidence
from src.beliefs.scoring import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    EvidenceItem,
    compute_confidence,
    determine_status,
)
from src.common.enums import BeliefStatus, BeliefType, PersistencePolicy, SensitivityClass
from src.common.registry import BELIEF_TYPE_REGISTRY, BELIEF_TYPE_REGISTRY_VERSION


def _to_evidence_item(evidence: BeliefEvidence, as_of: datetime) -> EvidenceItem:
    age_days = (as_of - evidence.observed_at).total_seconds() / 86400.0
    return EvidenceItem(
        direction=evidence.direction.value,
        strength=evidence.strength,
        source_reliability=evidence.source_reliability,
        decay_lambda=evidence.decay_lambda,
        age_days=age_days,
        source_type=evidence.source_type.value,
    )


def _validate_active_evidence(
    active: list[BeliefEvidence], *, belief_id: str, user_id: str, as_of: datetime
) -> None:
    """Fail closed rather than silently corrupt confidence (blueprint
    section 6.0.1's "derived belief caches must be recomputed from active...
    belief_evidence only" implies that evidence, not just its count).

    Both checks run only against the *active* subset (matching what actually
    reaches ``compute_confidence()``); an inactive/suppressed row with a
    foreign scope or a future timestamp does not affect the result and is
    left alone.
    """
    foreign_belief = [row.evidence_id for row in active if row.belief_id != belief_id]
    if foreign_belief:
        raise ValueError(
            f"active evidence contains row(s) with belief_id != {belief_id!r}: {foreign_belief!r}. "
            "recompute_belief() only accepts evidence already scoped to the belief being recomputed."
        )
    foreign_user = [row.evidence_id for row in active if row.user_id != user_id]
    if foreign_user:
        raise ValueError(
            f"active evidence contains row(s) with user_id != {user_id!r}: {foreign_user!r}. "
            "recompute_belief() only accepts evidence already scoped to the user being recomputed."
        )
    future_dated = [row.evidence_id for row in active if row.observed_at > as_of]
    if future_dated:
        raise ValueError(
            f"active evidence contains row(s) observed after as_of={as_of.isoformat()!r}: "
            f"{future_dated!r}. age_days cannot be computed for evidence dated after the recompute itself."
        )


def _compute_counts(items: list[EvidenceItem]) -> dict[str, float | int]:
    supporting = [item for item in items if item.direction == "support"]
    contradicting = [item for item in items if item.direction == "contradict"]
    return {
        "supporting_evidence_count": len(supporting),
        "contradicting_evidence_count": len(contradicting),
        "total_evidence_count": len(items),
        "effective_support_count": sum(item.count_weight() for item in supporting),
        "effective_evidence_count": sum(item.count_weight() for item in items),
        "evidence_for": len(supporting),
        "evidence_against": len(contradicting),
    }


def recompute_belief(
    *,
    belief_id: str,
    user_id: str,
    belief_type: BeliefType,
    belief_key: str,
    belief_value: Any,
    evidence: list[BeliefEvidence],
    as_of: datetime,
    first_observed: datetime,
    schema_version: str,
    scoring_version: str,
    canonicalizer_version: str,
    policy_version: str,
    recompute_reason: str | None = None,
    no_active_evidence_status: BeliefStatus = BeliefStatus.OUTDATED,
    sensitivity_class: SensitivityClass | None = None,
    persistence_policy: PersistencePolicy | None = None,
    allowed_contexts: list[str] | None = None,
    disallowed_contexts: list[str] | None = None,
    reasoning_summary: str | None = None,
    last_recompute_attempt_id: str | None = None,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> UserBelief:
    """Recompute ``belief_id`` from ``evidence`` as of ``as_of`` (blueprint
    section 3.4.1, section 6.0.1-6.0.2).

    ``evidence`` should already be scoped to this belief (every row sharing
    ``belief_id`` and ``user_id``); this function filters by active-evidence
    status via ``active_evidence()`` and then validates the result -- any
    active row carrying a different ``belief_id`` or ``user_id``, or an
    ``observed_at`` after ``as_of``, raises ``ValueError`` rather than
    silently feeding a corrupted or negatively-aged item into
    ``compute_confidence()``. There is still no database here to re-derive
    correct scope from, so a caller passing the wrong evidence in the first
    place is still possible -- this only guarantees the function fails
    closed instead of silently producing a wrong confidence when it does.

    ``recompute_reason`` and ``no_active_evidence_status`` map directly onto
    ``compute_confidence()``'s own invalidation branch: pass
    ``recompute_reason`` as one of ``"deletion"``, ``"reset"``,
    ``"duplicate_suppression"``, or ``"policy_invalidation"`` when this
    recompute is happening *because* evidence was just invalidated, so the
    zero-active-evidence branch is reached deliberately rather than by
    accident (the two branches produce the same ``confidence=0.0`` but a
    different ``status``/``reason``, and branch order is part of the
    contract -- see ``tests/beliefs/test_deletion_reset_recompute.py``).
    """
    active = active_evidence(evidence)
    _validate_active_evidence(active, belief_id=belief_id, user_id=user_id, as_of=as_of)
    items = [_to_evidence_item(row, as_of) for row in active]
    counts = _compute_counts(items)

    registry_defaults = BELIEF_TYPE_REGISTRY[belief_type]
    result = compute_confidence(
        items,
        weights or DEFAULT_WEIGHTS,
        expected_source_types=registry_defaults.expected_source_types,
        recompute_reason=recompute_reason,
        active_evidence_count=len(active),
        no_active_evidence_status=no_active_evidence_status.value,
    )

    status_value = result["status"]
    if status_value is None:
        status_value = determine_status(
            result["confidence"], result["D"], counts["effective_support_count"], thresholds or DEFAULT_THRESHOLDS
        )

    return UserBelief(
        belief_id=belief_id,
        user_id=user_id,
        belief_type=belief_type,
        belief_type_registry_version=BELIEF_TYPE_REGISTRY_VERSION,
        belief_key=belief_key,
        belief_value=belief_value,
        confidence=result["confidence"],
        supporting_evidence_count=counts["supporting_evidence_count"],
        contradicting_evidence_count=counts["contradicting_evidence_count"],
        total_evidence_count=counts["total_evidence_count"],
        effective_support_count=counts["effective_support_count"],
        effective_evidence_count=counts["effective_evidence_count"],
        evidence_for=counts["evidence_for"],
        evidence_against=counts["evidence_against"],
        allowed_contexts=allowed_contexts or [],
        disallowed_contexts=disallowed_contexts or [],
        sensitivity_class=sensitivity_class or registry_defaults.default_sensitivity_class,
        persistence_policy=persistence_policy or registry_defaults.default_persistence_policy,
        first_observed=first_observed,
        last_validated=as_of,
        status=status_value,
        reasoning_summary=reasoning_summary,
        locked_until_recompute=False,
        last_recompute_attempt_id=last_recompute_attempt_id,
        last_successful_recompute_at=as_of,
        schema_version=schema_version,
        scoring_version=scoring_version,
        canonicalizer_version=canonicalizer_version,
        policy_version=policy_version,
    )

"""Backend authorization for promoting an outcome-learning signal's
proposed evidence into the real ``belief_evidence`` ledger (blueprint
sections 6.1.2, 12.6).

This is **not** an automatic learner. ``promote_outcome_learning_signal``
runs only when a caller (the CLI) asks it to, for one named
``signal_id``, and every persistence-affecting step is behind an explicit
flag:

- Nothing is written unless ``persist=True``.
- No belief is recomputed unless ``recompute=True``; when evidence was
  added but ``recompute`` is off, the affected beliefs are locked
  (``locked_until_recompute=True``) so their cached confidence/status are
  treated as non-authoritative until a real recompute runs.

Deterministic gate (all must hold, else the signal is rejected and nothing
happens):

- ``causal_claim`` is ``False``;
- ``kind`` is ``support`` or ``weak_contradiction`` (never ``no_signal``);
- ``proposed_evidence`` is non-empty;
- ``trial_count`` still meets ``OUTCOME_LEARNING_POLICY.min_trials`` and
  the stored ``kind`` still matches ``classify_counts`` for the stored
  supportive/adverse/neutral breakdown -- a signal written under a policy
  that has since changed, or a tampered one, does not promote.

Each surviving proposal is:

1. checked against the ledger -- if an *active* ``repeated_pattern_summary``
   row for the same ``(user_id, belief_id, independence_group, direction)``
   already exists, it is skipped (a re-run of promotion for the same signal
   cannot add a second row);
2. re-provenanced -- ``source_event_ids`` is rebuilt from the target
   belief's *current* active evidence leaves, so the promoted row traces to
   real events; a belief with no leaf evidence yet is skipped;
3. authorized via ``authorize_evidence`` with
   ``independence_group=signal.independence_group`` (so repeated analyses of
   the same pattern share one independence group and cannot corroborate
   each other) and ``source_type=repeated_pattern_summary``.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.beliefs.models import BeliefEvidence, BeliefEvidenceProposal, authorize_evidence
from src.beliefs.recompute import recompute_belief
from src.common.enums import Direction, OutcomeLearningSignalKind, SourceType
from src.common.registry import OUTCOME_LEARNING_POLICY, OUTCOME_LEARNING_VERSION
from src.recommendations.models import OutcomeLearningSignal
from src.recommendations.outcome_learning import classify_counts

AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"
PROMOTION_AUTHORIZED_BY = "outcome_learning_promotion"

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)

_PROMOTABLE_KINDS = {OutcomeLearningSignalKind.SUPPORT, OutcomeLearningSignalKind.WEAK_CONTRADICTION}


@dataclass(frozen=True)
class ProposalPromotion:
    """What happened to one of a signal's proposed evidence rows."""

    belief_id: str
    user_id: str
    direction: str
    strength: float
    action: str  # "authorized" | "inserted" | "skipped_existing" | "skipped_no_leaf_provenance"
    evidence_id: str | None = None
    source_event_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromotionResult:
    signal_id: str
    authorized: bool
    rejected_reason: str | None
    persisted: bool
    recomputed: bool
    independence_group: str | None
    proposals: list[ProposalPromotion]
    locked_belief_ids: list[str]
    recomputed_beliefs: list[dict[str, Any]]
    model_version: str = OUTCOME_LEARNING_VERSION

    @property
    def inserted_evidence_ids(self) -> list[str]:
        return [p.evidence_id for p in self.proposals if p.action == "inserted" and p.evidence_id]


def evaluate_signal_for_promotion(
    signal: OutcomeLearningSignal, *, min_trials: int | None = None
) -> tuple[bool, str | None]:
    """The pure deterministic gate. Returns ``(True, None)`` if the signal
    may be promoted, or ``(False, reason)`` otherwise."""
    if signal.causal_claim:
        return False, "causal_claim is true"
    if signal.kind not in _PROMOTABLE_KINDS:
        return False, f"kind {signal.kind.value!r} is not promotable"
    if not signal.proposed_evidence:
        return False, "signal has no proposed_evidence"

    threshold = OUTCOME_LEARNING_POLICY.min_trials if min_trials is None else min_trials
    if signal.trial_count < threshold:
        return False, f"trial_count {signal.trial_count} is below the {threshold}-trial threshold"

    if signal.supportive_count + signal.adverse_count + signal.neutral_count != signal.trial_count:
        return False, "supportive/adverse/neutral counts do not sum to trial_count"

    expected_kind = classify_counts(
        supportive=signal.supportive_count,
        adverse=signal.adverse_count,
        neutral=signal.neutral_count,
        min_trials=threshold,
    )
    if expected_kind is not signal.kind:
        return False, (
            f"stored kind {signal.kind.value!r} no longer matches current policy "
            f"(counts imply {expected_kind.value!r})"
        )

    expected_direction = (
        Direction.SUPPORT if signal.kind is OutcomeLearningSignalKind.SUPPORT else Direction.CONTRADICT
    )
    if signal.direction is not expected_direction or any(
        p.direction is not expected_direction for p in signal.proposed_evidence
    ):
        return False, "signal / proposal direction is inconsistent with kind"

    return True, None


def _equivalent_evidence_exists(
    existing: list[BeliefEvidence], *, independence_group: str, direction: Direction
) -> bool:
    return any(
        row.is_active
        and not row.is_duplicate_suppressed
        and row.independence_group == independence_group
        and row.source_type is SourceType.REPEATED_PATTERN_SUMMARY
        and row.direction is direction
        for row in existing
    )


def _derive_source_event_ids(repo: Any, *, user_id: str, belief_id: str) -> list[str]:
    events: set[str] = set()
    for row in repo.list_active_evidence(user_id=user_id, belief_id=belief_id):
        events.update(row.source_event_ids)
    return sorted(e for e in events if e.strip())


def promote_outcome_learning_signal(
    repo: Any,
    *,
    signal_id: str,
    as_of: datetime,
    persist: bool = False,
    recompute: bool = False,
    min_trials: int | None = None,
) -> PromotionResult:
    """Promote the proposed evidence of one outcome-learning signal.

    ``repo`` is only read from unless ``persist=True``. Returns a
    ``PromotionResult`` describing every decision; raises ``ValueError`` only
    if ``signal_id`` names no stored signal.
    """
    signal = repo.get_outcome_learning_signal(signal_id)
    if signal is None:
        raise ValueError(f"no outcome-learning signal {signal_id!r} is stored")

    ok, reason = evaluate_signal_for_promotion(signal, min_trials=min_trials)
    if not ok:
        return PromotionResult(
            signal_id=signal_id, authorized=False, rejected_reason=reason, persisted=False,
            recomputed=False, independence_group=signal.independence_group, proposals=[],
            locked_belief_ids=[], recomputed_beliefs=[],
        )

    expected_direction = (
        Direction.SUPPORT if signal.kind is OutcomeLearningSignalKind.SUPPORT else Direction.CONTRADICT
    )

    proposals: list[ProposalPromotion] = []
    authorized_rows: list[tuple[str, str, BeliefEvidence]] = []  # (user_id, belief_id, evidence)

    for proposal in signal.proposed_evidence:
        existing = repo.list_evidence(user_id=proposal.user_id, belief_id=proposal.belief_id)
        if _equivalent_evidence_exists(
            existing, independence_group=signal.independence_group, direction=expected_direction
        ):
            proposals.append(
                ProposalPromotion(
                    belief_id=proposal.belief_id, user_id=proposal.user_id,
                    direction=proposal.direction.value, strength=proposal.strength,
                    action="skipped_existing",
                )
            )
            continue

        derived = _derive_source_event_ids(
            repo, user_id=proposal.user_id, belief_id=proposal.belief_id
        )
        if not derived:
            proposals.append(
                ProposalPromotion(
                    belief_id=proposal.belief_id, user_id=proposal.user_id,
                    direction=proposal.direction.value, strength=proposal.strength,
                    action="skipped_no_leaf_provenance",
                )
            )
            continue

        rebuilt = _rebuilt_proposal(proposal, source_event_ids=derived, as_of=as_of)
        evidence_id = f"bev-ols-{signal.signal_id}-{proposal.belief_id}"
        authorized = authorize_evidence(
            rebuilt,
            evidence_id=evidence_id,
            created_at=as_of,
            aggregation_policy_version=AGGREGATION_POLICY_VERSION,
            aggregation_authorized_by=PROMOTION_AUTHORIZED_BY,
            independence_group=signal.independence_group,
        )
        authorized_rows.append((proposal.user_id, proposal.belief_id, authorized))
        proposals.append(
            ProposalPromotion(
                belief_id=proposal.belief_id, user_id=proposal.user_id,
                direction=proposal.direction.value, strength=proposal.strength,
                action="authorized", evidence_id=evidence_id, source_event_ids=derived,
            )
        )

    if not persist:
        return PromotionResult(
            signal_id=signal_id, authorized=True, rejected_reason=None, persisted=False,
            recomputed=False, independence_group=signal.independence_group, proposals=proposals,
            locked_belief_ids=[], recomputed_beliefs=[],
        )

    newly_inserted: set[tuple[str, str]] = set()
    inserted_by_evidence_id: dict[str, tuple[str, str]] = {}
    for user_id, belief_id, authorized in authorized_rows:
        try:
            repo.insert_evidence(authorized)
            newly_inserted.add((user_id, belief_id))
            inserted_by_evidence_id[authorized.evidence_id] = (user_id, belief_id)
        except sqlite3.IntegrityError:
            pass  # a prior promotion already wrote this exact row -- treat as skipped

    updated_proposals: list[ProposalPromotion] = []
    for promotion in proposals:
        if promotion.action == "authorized":
            was_inserted = promotion.evidence_id in inserted_by_evidence_id
            updated_proposals.append(
                ProposalPromotion(
                    belief_id=promotion.belief_id, user_id=promotion.user_id,
                    direction=promotion.direction, strength=promotion.strength,
                    action="inserted" if was_inserted else "skipped_existing",
                    evidence_id=promotion.evidence_id, source_event_ids=promotion.source_event_ids,
                )
            )
        else:
            updated_proposals.append(promotion)

    # Every belief this signal has evidence for (inserted now or on a prior
    # run) whose stored belief is still locked from a persist-without-recompute
    # is "unsettled" -- a later --recompute run must finish the job.
    unsettled: set[tuple[str, str]] = set()
    for promotion in updated_proposals:
        if promotion.action in {"inserted", "skipped_existing"}:
            latest = repo.get_latest_belief(
                user_id=promotion.user_id, belief_id=promotion.belief_id
            )
            if latest is not None and latest.locked_until_recompute:
                unsettled.add((promotion.user_id, promotion.belief_id))

    locked_belief_ids: list[str] = []
    recomputed_beliefs: list[dict[str, Any]] = []

    if recompute:
        for user_id, belief_id in sorted(newly_inserted | unsettled):
            recomputed_beliefs.append(
                _recompute_one(repo, user_id=user_id, belief_id=belief_id, as_of=as_of)
            )
    else:
        for user_id, belief_id in sorted(newly_inserted):
            if repo.lock_belief_until_recompute(user_id=user_id, belief_id=belief_id):
                locked_belief_ids.append(belief_id)

    return PromotionResult(
        signal_id=signal_id, authorized=True, rejected_reason=None, persisted=True,
        recomputed=bool(recomputed_beliefs), independence_group=signal.independence_group,
        proposals=updated_proposals, locked_belief_ids=locked_belief_ids,
        recomputed_beliefs=recomputed_beliefs,
    )


def _rebuilt_proposal(
    proposal: BeliefEvidenceProposal, *, source_event_ids: list[str], as_of: datetime
) -> BeliefEvidenceProposal:
    data = proposal.model_dump()
    data["source_event_ids"] = source_event_ids
    data["event_id"] = None
    # The pattern was observed no later than the moment of promotion; clamp so
    # a subsequent recompute never sees evidence dated after its own as_of.
    if isinstance(data.get("observed_at"), datetime) and data["observed_at"] > as_of:
        data["observed_at"] = as_of
    return BeliefEvidenceProposal.model_validate(data)


def _recompute_one(repo: Any, *, user_id: str, belief_id: str, as_of: datetime) -> dict[str, Any]:
    latest = repo.get_latest_belief(user_id=user_id, belief_id=belief_id)
    active = repo.list_active_evidence(user_id=user_id, belief_id=belief_id)
    if latest is None:
        return {"belief_id": belief_id, "recomputed": False, "reason": "no belief to recompute"}
    first_observed = min((row.observed_at for row in active), default=latest.first_observed)
    belief = recompute_belief(
        belief_id=belief_id,
        user_id=user_id,
        belief_type=latest.belief_type,
        belief_key=latest.belief_key,
        belief_value=latest.belief_value,
        evidence=active,
        as_of=as_of,
        first_observed=first_observed,
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

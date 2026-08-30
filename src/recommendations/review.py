"""Manual review workflow for outcome-learning signal promotion
(blueprint section 6.4's manual-review gate, applied to the section 12.6
learning loop).

``review_outcome_learning_signal()`` records one reviewer decision and, only
when the decision is ``approved`` *and* promotion was explicitly requested,
calls the sanctioned ``promote_outcome_learning_signal`` -- it never
re-implements promotion. A ``rejected`` review promotes nothing. The result
of any promotion is written back onto the stored review so the whole action
is replayable from the audit trail.

Promotion and the review's own audit row are written inside a single
``repo.transaction()``: the promoted ``belief_evidence`` rows and any
recomputed beliefs commit only if the review row commits with them. If
storing the review fails, the promotion is rolled back, so an approved
review can never leave changed belief state behind with no audit record to
explain it.

Duplicate protection: a second ``approved`` review for a signal that already
has one is refused unless ``allow_duplicate=True``; when allowed, the new
review is still stored (and the audit trail shows both, each with its own
``review_id`` and ``created_at``). Promotion itself is independently
idempotent -- ``promote_outcome_learning_signal`` skips a proposal whose
belief already carries an equivalent promoted row -- so an allowed duplicate
approval never double-counts evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.enums import OutcomeLearningReviewDecision
from src.recommendations.models import (
    OutcomeLearningSignalReview,
    OutcomeLearningSignalReviewProposal,
)
from src.recommendations.promotion import PromotionResult, promote_outcome_learning_signal

_VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


@dataclass(frozen=True)
class ReviewOutcome:
    """What ``review_outcome_learning_signal`` did. ``stored`` is False only
    when the review was blocked (``blocked_reason`` says why); otherwise
    ``review`` is the persisted record and ``promotion`` is the
    ``PromotionResult`` if one ran."""

    review: OutcomeLearningSignalReview | None
    stored: bool
    blocked_reason: str | None
    promotion: PromotionResult | None


def review_outcome_learning_signal(
    repo: Any,
    *,
    review_id: str,
    signal_id: str,
    reviewer_id: str,
    decision: OutcomeLearningReviewDecision | str,
    as_of: datetime,
    notes: str | None = None,
    promote: bool = False,
    recompute: bool = False,
    allow_duplicate: bool = False,
    proposal: OutcomeLearningSignalReviewProposal | None = None,
) -> ReviewOutcome:
    """Record a reviewer decision on ``signal_id`` and, if approved and
    ``promote=True``, promote its proposed evidence.

    Raises ``ValueError`` for an unknown ``signal_id``, an already-used
    ``review_id``, ``recompute`` without ``promote``, or ``promote`` on a
    ``rejected`` decision. Returns a blocked ``ReviewOutcome`` (nothing
    written) if the signal already has an approved review and
    ``allow_duplicate`` is False.
    """
    decision = OutcomeLearningReviewDecision(decision)

    signal = repo.get_outcome_learning_signal(signal_id)
    if signal is None:
        raise ValueError(f"no outcome-learning signal {signal_id!r} is stored")
    if repo.get_outcome_learning_signal_review(review_id) is not None:
        raise ValueError(f"review_id {review_id!r} is already used; choose a new one")
    if recompute and not promote:
        raise ValueError("recompute requires promote")
    if promote and decision is OutcomeLearningReviewDecision.REJECTED:
        raise ValueError("a rejected review cannot promote anything")

    if decision is OutcomeLearningReviewDecision.APPROVED and not allow_duplicate:
        prior_approved = [
            r
            for r in repo.list_outcome_learning_signal_reviews(signal_id=signal_id)
            if r.decision is OutcomeLearningReviewDecision.APPROVED
        ]
        if prior_approved:
            return ReviewOutcome(
                review=None,
                stored=False,
                blocked_reason=(
                    f"signal {signal_id!r} already has an approved review "
                    f"({prior_approved[0].review_id!r}); pass allow_duplicate to record another"
                ),
                promotion=None,
            )

    notes = notes if notes is not None else (proposal.suggested_notes if proposal is not None else None)

    # Promotion (which writes belief_evidence and may recompute beliefs) and
    # the review's own audit row are written inside one atomic transaction:
    # if the audit insert fails, the promotion it would have recorded is
    # rolled back too, so an approved review can never leave promoted
    # evidence or recomputed beliefs behind without a stored review to
    # explain them. ``repo.transaction()`` is reentrant, so
    # ``promote_outcome_learning_signal``'s own writes join this transaction
    # rather than committing early.
    promotion: PromotionResult | None = None
    with repo.transaction():
        promoted_evidence_ids: list[str] = []
        recomputed_belief_ids: list[str] = []
        review_notes = notes
        if decision is OutcomeLearningReviewDecision.APPROVED and promote:
            promotion = promote_outcome_learning_signal(
                repo, signal_id=signal_id, as_of=as_of, persist=True, recompute=recompute
            )
            promoted_evidence_ids = list(promotion.inserted_evidence_ids)
            recomputed_belief_ids = [belief["belief_id"] for belief in promotion.recomputed_beliefs]
            if not promotion.authorized and promotion.rejected_reason:
                gate_note = f"promotion gate rejected: {promotion.rejected_reason}"
                review_notes = f"{review_notes}; {gate_note}" if review_notes else gate_note

        review = OutcomeLearningSignalReview(
            review_id=review_id,
            signal_id=signal_id,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=review_notes,
            promotion_requested=promote and decision is OutcomeLearningReviewDecision.APPROVED,
            recompute_requested=recompute and decision is OutcomeLearningReviewDecision.APPROVED,
            promoted_evidence_ids=promoted_evidence_ids,
            recomputed_belief_ids=recomputed_belief_ids,
            created_at=as_of,
            **_VERSION_FIELDS,
        )
        repo.insert_outcome_learning_signal_review(review)

    return ReviewOutcome(review=review, stored=True, blocked_reason=None, promotion=promotion)


def review_status_for(reviews: list[OutcomeLearningSignalReview]) -> str:
    """The current review status of a signal from its reviews (most recent
    wins). ``"pending"`` when there are none."""
    if not reviews:
        return "pending"
    return reviews[-1].decision.value

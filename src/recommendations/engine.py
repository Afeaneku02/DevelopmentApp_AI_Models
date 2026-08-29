"""Deterministic recommendation MVP (blueprint sections 3.2, 6.4, 12.5).

``generate_recommendation()`` takes a user's beliefs and a proposed
``context_key`` and produces one persisted-shaped ``UserRecommendation``:

1. **Authorize** -- ``authorize_beliefs_for_context()`` (the existing
   context/risk policy foundation) resolves the risk tier and filters the
   beliefs. Only beliefs it authorizes can influence the recommendation;
   everything it blocks is recorded in ``blocked_beliefs`` with the reason.
2. **Candidates** -- each authorized belief whose ``belief_value`` is truthy
   maps, via a fixed template table, to a candidate action. Beliefs that map
   to the same action are grouped into one candidate backed by all of them.
3. **Rank** -- ``ranking_score = confidence * status_weight *
   diversity_factor`` per belief (``src/common/registry.RECOMMENDATION_RANKING``),
   averaged over a candidate's beliefs. Deterministic ordering: score
   descending, then action text ascending.
4. **Gate** -- if the resolved context is HIGH risk or its domain
   ``requires_manual_review``, the top candidate is recorded as a *proposed*
   recommendation with ``review_required=True`` / ``review_status=pending``
   / ``required_resolution_mode=reviewer`` and is NOT auto-issued (section
   6.4). Otherwise it is issued directly.
5. **Freeze** -- ``frozen_belief_state`` captures every candidate belief
   (authorized or blocked) exactly as it was, per the section 5 audit rule.

No LLM, no exploration (``exploration_applied`` is always ``False`` in this
MVP), no learned ranking -- those are later phases.
"""
from __future__ import annotations

from datetime import datetime
from statistics import fmean
from typing import Any

from src.beliefs.models import UserBelief
from src.common.enums import (
    BeliefStatus,
    ContextResolutionPath,
    RecommendationReviewStatus,
    RecommendationRiskTier,
    ResolutionMode,
    RiskResolutionPath,
)
from src.common.registry import (
    BELIEF_TYPE_REGISTRY,
    RECOMMENDATION_RANKING,
    RECOMMENDATION_RANKING_VERSION,
)
from src.recommendations.context_policy import (
    ContextAuthorizationResult,
    RecommendationContextProposal,
    authorize_beliefs_for_context,
)
from src.recommendations.models import BlockedBelief, RankedCandidate, UserRecommendation

DEFAULT_MODEL_VERSION = "rec-mvp-0.6"

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)

# Fixed, deterministic candidate-action templates keyed by canonical
# belief_key. A belief_key with no entry falls back to the generic template.
# This table is the entire "candidate generation" surface of the MVP -- there
# is no LLM step. Add an entry when a new belief_key should phrase a specific
# action; the generic fallback keeps unknown keys safe and auditable.
CANDIDATE_TEMPLATES: dict[str, str] = {
    "higher_adherence_after_work": (
        "Schedule this user's workout right after work, when their adherence is highest."
    ),
    "prefers_evening_exercise_sessions": (
        "Offer evening workout slots first for this user."
    ),
    "prefers_short_messages": (
        "Keep nudges to one or two short sentences for this user."
    ),
    "responds_to_streak_framing": (
        "Frame progress as a streak the user is continuing."
    ),
}
_GENERIC_TEMPLATE = "Act on the user's established tendency: {belief_key}."

_ROUND = 6

_RESOLUTION_PATH_MAP: dict[ContextResolutionPath, RiskResolutionPath] = {
    ContextResolutionPath.EXACT_CONTEXT_POLICY: RiskResolutionPath.EXACT_CONTEXT,
    ContextResolutionPath.RISK_DOMAIN_POLICY: RiskResolutionPath.DOMAIN_POLICY,
    ContextResolutionPath.GLOBAL_CONSERVATIVE_FALLBACK: RiskResolutionPath.GLOBAL_FALLBACK,
}


def _is_truthy_value(belief_value: Any) -> bool:
    """A belief only suggests *doing* something when its value is affirmatively
    true. ``False`` / ``None`` / ``0`` / ``""`` beliefs are real beliefs but
    are not a basis for an action in this MVP."""
    return bool(belief_value)


def _candidate_action(belief: UserBelief) -> str:
    return CANDIDATE_TEMPLATES.get(belief.belief_key) or _GENERIC_TEMPLATE.format(
        belief_key=belief.belief_key
    )


def _score_belief(belief: UserBelief) -> tuple[float, dict[str, float]]:
    weights = RECOMMENDATION_RANKING
    status_weight = weights.status_weight.get(belief.status, 0.0)

    expected_sources = BELIEF_TYPE_REGISTRY[belief.belief_type].expected_source_types
    diversity = (
        min(1.0, belief.effective_evidence_count / expected_sources) if expected_sources else 1.0
    )
    diversity_factor = weights.diversity_floor + (1.0 - weights.diversity_floor) * diversity
    score = round(belief.confidence * status_weight * diversity_factor, _ROUND)
    return score, {
        "confidence": round(belief.confidence, _ROUND),
        "status_weight": status_weight,
        "diversity": round(diversity, _ROUND),
        "diversity_factor": round(diversity_factor, _ROUND),
    }


def _frozen_state(
    beliefs: list[UserBelief], authorization: ContextAuthorizationResult
) -> dict[str, dict[str, Any]]:
    reason_by_id = {d.belief_id: d for d in authorization.decisions}
    authorized_ids = set(authorization.authorized_beliefs)
    frozen: dict[str, dict[str, Any]] = {}
    for belief in beliefs:
        decision = reason_by_id.get(belief.belief_id)
        frozen[belief.belief_id] = {
            "belief_key": belief.belief_key,
            "belief_type": belief.belief_type.value,
            "belief_value": belief.belief_value,
            "confidence": belief.confidence,
            "status": belief.status.value,
            "locked_until_recompute": belief.locked_until_recompute,
            "total_evidence_count": belief.total_evidence_count,
            "effective_evidence_count": belief.effective_evidence_count,
            "sensitivity_class": belief.sensitivity_class.value,
            "allowed_contexts": list(belief.allowed_contexts),
            "disallowed_contexts": list(belief.disallowed_contexts),
            "authorized": belief.belief_id in authorized_ids,
            "block_reason": None if decision is None or decision.allowed else decision.reason.value,
        }
    return frozen


def _build_candidates(authorized: list[UserBelief]) -> list[dict[str, Any]]:
    """Group authorized truthy beliefs by their template action into ranked
    candidates. Deterministic: score descending, then action ascending."""
    by_action: dict[str, list[UserBelief]] = {}
    for belief in authorized:
        if not _is_truthy_value(belief.belief_value):
            continue
        by_action.setdefault(_candidate_action(belief), []).append(belief)

    candidates: list[dict[str, Any]] = []
    for action, members in by_action.items():
        scored = [(_score_belief(b)) for b in members]
        member_scores = [s for s, _ in scored]
        candidates.append(
            {
                "action": action,
                "belief_ids": sorted(b.belief_id for b in members),
                "ranking_score": round(fmean(member_scores), _ROUND),
                "confidence": round(fmean(b.confidence for b in members), _ROUND),
                "components": {
                    "member_count": float(len(members)),
                    "mean_member_score": round(fmean(member_scores), _ROUND),
                    "max_member_score": round(max(member_scores), _ROUND),
                },
            }
        )
    candidates.sort(key=lambda c: (-c["ranking_score"], c["action"]))
    return candidates


def generate_recommendation(
    *,
    recommendation_id: str,
    user_id: str,
    context_key: str | RecommendationContextProposal,
    beliefs: list[UserBelief],
    created_at: datetime,
    goal: str | None = None,
    expected_outcome: str | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    context_registry: Any | None = None,
    domain_registry: Any | None = None,
) -> UserRecommendation:
    """Produce one deterministic ``UserRecommendation`` for ``user_id`` in
    ``context_key`` from ``beliefs`` (typically ``repo.list_latest_beliefs(
    user_id=...)``). Pure: no I/O, no persistence -- the caller persists the
    result via ``Repository.insert_recommendation``.

    Only beliefs authorized by ``authorize_beliefs_for_context`` can drive
    the recommendation. In a HIGH-risk or ``requires_manual_review`` context
    the top candidate is returned as a *proposed* recommendation with
    ``review_required=True`` and is not auto-issued.
    """
    beliefs = list(beliefs)
    authorization = authorize_beliefs_for_context(
        beliefs, context_key, context_registry=context_registry, domain_registry=domain_registry
    )

    authorized_ids = set(authorization.authorized_beliefs)
    authorized = [b for b in beliefs if b.belief_id in authorized_ids]

    candidates = _build_candidates(authorized)
    blocked = sorted(
        (
            BlockedBelief(
                belief_id=d.belief_id, belief_key=d.belief_key, reason=d.reason, detail=d.detail
            )
            for d in authorization.decisions
            if not d.allowed
        ),
        key=lambda b: b.belief_id,
    )

    review_required = (
        authorization.requires_manual_review
        or authorization.risk_tier is RecommendationRiskTier.HIGH
    )
    review_status = (
        RecommendationReviewStatus.PENDING if review_required else RecommendationReviewStatus.NOT_REQUIRED
    )
    # MVP: any review-gated recommendation goes to a human reviewer. Per-tier
    # nuance (medium-risk user confirmation under tight constraints; explicit
    # domain-policy exceptions) is deferred with the manual-review workflow.
    required_resolution_mode = ResolutionMode.REVIEWER if review_required else None

    frozen = _frozen_state(beliefs, authorization)

    top = candidates[0] if candidates else None
    candidate_trace = [
        RankedCandidate(
            action=c["action"],
            belief_ids=c["belief_ids"],
            ranking_score=c["ranking_score"],
            components=c["components"],
            selected=(top is not None and c is top),
        )
        for c in candidates
    ]

    if top is not None:
        recommendation_text = top["action"]
        belief_ids_used = list(top["belief_ids"])
        ranking_score = top["ranking_score"]
        confidence = top["confidence"]
    else:
        recommendation_text = (
            "(no recommendation issued: no beliefs are eligible for this context)"
        )
        belief_ids_used = []
        ranking_score = 0.0
        confidence = 0.0

    rationale = _rationale(
        authorization=authorization,
        total_beliefs=len(beliefs),
        authorized_count=len(authorized),
        top=top,
        review_required=review_required,
    )

    return UserRecommendation(
        recommendation_id=recommendation_id,
        user_id=user_id,
        goal=goal,
        recommendation=recommendation_text,
        proposed_context_key=authorization.proposed_context_key,
        recommendation_context=authorization.context_key,
        risk_tier=authorization.risk_tier,
        risk_policy_context_key=authorization.context_key,
        risk_policy_version=authorization.risk_policy_version,
        risk_domain_policy_version=authorization.risk_domain_policy_version,
        risk_resolution_path=_RESOLUTION_PATH_MAP[authorization.resolution_path],
        review_required=review_required,
        review_status=review_status,
        required_resolution_mode=required_resolution_mode,
        actual_resolution_mode=None,
        resolved_by=None,
        resolved_at=None,
        review_notes_or_policy_reference=(
            "requires_manual_review / high-risk context: not auto-issued; awaiting reviewer"
            if review_required
            else None
        ),
        requires_user_confirmation=authorization.requires_user_confirmation,
        exploration_applied=False,
        profile_snapshot_id=None,
        frozen_belief_state=frozen,
        belief_ids_used=belief_ids_used,
        expected_outcome=expected_outcome,
        ranking_score=ranking_score,
        confidence=confidence,
        rationale=rationale,
        candidate_trace=candidate_trace,
        blocked_beliefs=blocked,
        ranking_policy_version=RECOMMENDATION_RANKING_VERSION,
        model_version=model_version,
        prompt_version=None,
        created_at=created_at,
        **_VERSION_FIELDS,
    )


def _rationale(
    *,
    authorization: ContextAuthorizationResult,
    total_beliefs: int,
    authorized_count: int,
    top: dict[str, Any] | None,
    review_required: bool,
) -> str:
    head = (
        f"context {authorization.proposed_context_key!r} resolved to {authorization.context_key!r} "
        f"via {authorization.resolution_path.value} at {authorization.risk_tier.value}-risk "
        f"(policy {authorization.risk_policy_version} / {authorization.risk_domain_policy_version}); "
        f"{authorized_count} of {total_beliefs} candidate belief(s) authorized"
    )
    if top is None:
        return head + "; no eligible belief maps to an action, so nothing was recommended."
    chosen = (
        f"; selected action {top['action']!r} backed by {top['belief_ids']} "
        f"with ranking_score {top['ranking_score']}"
    )
    tail = " -- held for manual review (not auto-issued)." if review_required else " -- issued."
    return head + chosen + tail

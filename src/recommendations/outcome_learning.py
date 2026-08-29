"""Outcome-learning MVP (blueprint section 12.6).

``analyze_recommendation_outcomes()`` reads persisted recommendations and
their outcomes and, only where there have been enough repeated trials,
emits a conservative ``OutcomeLearningSignal`` per recommendation pattern --
a (recommendation_context, belief_ids_used) pair.

It is deliberately timid:

- **Repeated trials required.** A pattern with fewer than
  ``OUTCOME_LEARNING_POLICY.min_trials`` outcomes produces nothing at all.
- **Direction is conservative.** Repeated *followed-and-(successful|mixed)*
  outcomes may propose weak ``support`` for the beliefs the recommendation
  used. Repeated *not_followed / ignored / unsuccessful* outcomes may
  propose weak ``contradiction`` -- but only when the negatives clearly
  outnumber everything else, so one or two bad outcomes among many never
  penalise a belief. Anything ambiguous is ``no_signal``.
- **No causal claim.** The evidence is about the belief that informed the
  recommendation, framed as correlation ("outcomes co-occurred with"), never
  "the recommendation worked". ``OutcomeLearningSignal.causal_claim`` is
  pinned ``False``.
- **Proposals only.** Nothing here mutates a belief, writes to the
  belief_evidence ledger, or recomputes. The output is a list of
  ``BeliefEvidenceProposal`` wrapped in a signal; promoting one into real
  evidence is a separate, later, backend-authorized step.

Pure: no I/O. The caller passes the recommendation and outcome rows (and,
optionally, each used belief's real backing ``source_event_ids`` so the
proposals trace to leaf events); it does the persistence.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime

from src.beliefs.models import BeliefEvidenceProposal
from src.common.enums import (
    BeliefType,
    Direction,
    OutcomeFollowed,
    OutcomeLearningSignalKind,
    OutcomeResult,
    SourceType,
)
from src.common.registry import (
    BELIEF_TYPE_REGISTRY,
    OUTCOME_LEARNING_POLICY,
    OUTCOME_LEARNING_VERSION,
    SOURCE_TYPE_RELIABILITY,
)
from src.recommendations.models import (
    OutcomeLearningSignal,
    RecommendationOutcome,
    UserRecommendation,
)

DEFAULT_MODEL_VERSION = OUTCOME_LEARNING_VERSION

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)

_SUPPORTIVE_FOLLOWED = {OutcomeFollowed.FOLLOWED, OutcomeFollowed.PARTIALLY_FOLLOWED}
_SUPPORTIVE_RESULT = {OutcomeResult.SUCCESSFUL, OutcomeResult.MIXED}
_ADVERSE_FOLLOWED = {OutcomeFollowed.NOT_FOLLOWED, OutcomeFollowed.IGNORED}


def _classify(outcome: RecommendationOutcome) -> str:
    """One outcome -> "supportive" | "adverse" | "neutral". Conservative: a
    followed-and-positive outcome is supportive; a not-followed or
    unsuccessful one is adverse; everything else (unknown result, unknown
    followed state) is neutral and contributes to neither direction."""
    if outcome.followed in _SUPPORTIVE_FOLLOWED and outcome.result in _SUPPORTIVE_RESULT:
        return "supportive"
    if outcome.followed in _ADVERSE_FOLLOWED or outcome.result is OutcomeResult.UNSUCCESSFUL:
        return "adverse"
    return "neutral"


def _signal_id(context: str, belief_ids: list[str], outcome_ids: list[str]) -> str:
    digest = hashlib.sha1(
        "|".join([context, ",".join(sorted(belief_ids)), ",".join(sorted(outcome_ids))]).encode("utf-8")
    ).hexdigest()
    return f"ols-{digest[:16]}"


def _proposal_source_event_ids(
    belief_id: str,
    context: str,
    outcome_ids: list[str],
    belief_source_events: dict[str, list[str]] | None,
) -> list[str]:
    real = (belief_source_events or {}).get(belief_id)
    if real:
        return sorted({event_id for event_id in real if event_id.strip()})
    # No leaf-event provenance was supplied for this belief. Fall back to a
    # stable token that still points back at the outcome rows analysed, so
    # the proposal is never provenance-free. A future promotion step must
    # re-derive real source_event_ids before this becomes ledger evidence.
    return [f"recommendation_outcome_learning:{context}:{belief_id}"]


def analyze_recommendation_outcomes(
    recommendations: Iterable[UserRecommendation],
    outcomes: Iterable[RecommendationOutcome],
    *,
    as_of: datetime,
    min_trials: int | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    belief_source_events: dict[str, list[str]] | None = None,
) -> list[OutcomeLearningSignal]:
    """Group outcomes by recommendation pattern and, for every pattern with
    at least ``min_trials`` outcomes, emit one conservative
    ``OutcomeLearningSignal``.

    Only auto-issued recommendations (``review_required`` False) that
    actually used beliefs are considered. Below the trial threshold a
    pattern yields nothing.
    """
    threshold = OUTCOME_LEARNING_POLICY.min_trials if min_trials is None else min_trials
    policy = OUTCOME_LEARNING_POLICY

    outcomes_by_rec: dict[str, list[RecommendationOutcome]] = {}
    for outcome in outcomes:
        outcomes_by_rec.setdefault(outcome.recommendation_id, []).append(outcome)

    groups: dict[tuple[str, str, tuple[str, ...]], dict] = {}
    for rec in recommendations:
        if rec.review_required or not rec.belief_ids_used:
            continue
        rec_outcomes = outcomes_by_rec.get(rec.recommendation_id, [])
        if not rec_outcomes:
            continue
        key = (rec.user_id, rec.recommendation_context, tuple(sorted(rec.belief_ids_used)))
        group = groups.setdefault(
            key,
            {
                "user_id": rec.user_id,
                "context": rec.recommendation_context,
                "belief_ids": list(key[2]),
                "recommendation_ids": set(),
                "outcomes": [],
                "belief_types": {},
            },
        )
        group["recommendation_ids"].add(rec.recommendation_id)
        group["outcomes"].extend(rec_outcomes)
        for belief_id in rec.belief_ids_used:
            group["belief_types"][belief_id] = rec.frozen_belief_state[belief_id]["belief_type"]

    signals: list[OutcomeLearningSignal] = []
    for key in sorted(groups):
        group = groups[key]
        group_outcomes = group["outcomes"]
        trial_count = len(group_outcomes)
        if trial_count < threshold:
            continue

        supportive = adverse = neutral = 0
        for outcome in group_outcomes:
            bucket = _classify(outcome)
            supportive += bucket == "supportive"
            adverse += bucket == "adverse"
            neutral += bucket == "neutral"

        outcome_ids = sorted(outcome.outcome_id for outcome in group_outcomes)
        recommendation_ids = sorted(group["recommendation_ids"])
        belief_ids = group["belief_ids"]
        context = group["context"]
        independence_group = (
            f"outcome-learning:{context}:" + "|".join(sorted(belief_ids))
        )

        if supportive >= threshold and supportive > adverse:
            kind = OutcomeLearningSignalKind.SUPPORT
            direction = Direction.SUPPORT
            strength = round(
                min(policy.support_strength_cap, policy.support_strength_per_trial * supportive), 6
            )
        elif (
            adverse >= threshold
            and adverse > (supportive + neutral) * policy.contradiction_dominance
        ):
            kind = OutcomeLearningSignalKind.WEAK_CONTRADICTION
            direction = Direction.CONTRADICT
            strength = round(
                min(
                    policy.contradiction_strength_cap,
                    policy.contradiction_strength_per_trial * adverse,
                ),
                6,
            )
        else:
            kind = OutcomeLearningSignalKind.NO_SIGNAL
            direction = None
            strength = 0.0

        proposals: list[BeliefEvidenceProposal] = []
        observed_at = max(outcome.created_at for outcome in group_outcomes)
        if direction is not None:
            for belief_id in belief_ids:
                belief_type = BeliefType(group["belief_types"][belief_id])
                proposals.append(
                    BeliefEvidenceProposal(
                        belief_id=belief_id,
                        user_id=group["user_id"],
                        direction=direction,
                        event_id=None,
                        observation_id=None,
                        source_event_ids=_proposal_source_event_ids(
                            belief_id, context, outcome_ids, belief_source_events
                        ),
                        source_type=SourceType.REPEATED_PATTERN_SUMMARY,
                        context_key=context,
                        strength=strength,
                        source_reliability=SOURCE_TYPE_RELIABILITY[SourceType.REPEATED_PATTERN_SUMMARY],
                        observed_at=observed_at,
                        decay_lambda=BELIEF_TYPE_REGISTRY[belief_type].default_decay_lambda,
                        model_version=model_version,
                        prompt_version=None,
                        **_VERSION_FIELDS,
                    )
                )

        signals.append(
            OutcomeLearningSignal(
                signal_id=_signal_id(context, belief_ids, outcome_ids),
                user_id=group["user_id"],
                recommendation_context=context,
                belief_ids=belief_ids,
                recommendation_ids=recommendation_ids,
                outcome_ids=outcome_ids,
                trial_count=trial_count,
                supportive_count=supportive,
                adverse_count=adverse,
                neutral_count=neutral,
                kind=kind,
                direction=direction,
                independence_group=independence_group,
                proposed_evidence=proposals,
                rationale=_rationale(
                    context=context,
                    belief_ids=belief_ids,
                    trial_count=trial_count,
                    supportive=supportive,
                    adverse=adverse,
                    neutral=neutral,
                    kind=kind,
                    strength=strength,
                ),
                model_version=model_version,
                prompt_version=None,
                created_at=as_of,
                **_VERSION_FIELDS,
            )
        )

    return signals


def _rationale(
    *,
    context: str,
    belief_ids: list[str],
    trial_count: int,
    supportive: int,
    adverse: int,
    neutral: int,
    kind: OutcomeLearningSignalKind,
    strength: float,
) -> str:
    head = (
        f"{trial_count} recorded outcome(s) for auto-issued recommendations in context "
        f"{context!r} that used belief(s) {belief_ids}: {supportive} followed-and-"
        f"(successful|mixed), {adverse} not-followed/ignored/unsuccessful, {neutral} other."
    )
    tail = {
        OutcomeLearningSignalKind.SUPPORT: (
            f" These positive outcomes repeatedly co-occurred with recommendations informed by "
            f"these beliefs, so a weak SUPPORT proposal (strength {strength}) is offered. This is a "
            "correlational summary only: it does not establish that the recommendation, or the "
            "belief, caused the outcome."
        ),
        OutcomeLearningSignalKind.WEAK_CONTRADICTION: (
            f" Recommendations informed by these beliefs were repeatedly not followed or did not "
            f"succeed, and the negative outcomes clearly outnumber the rest, so a weak CONTRADICTION "
            f"proposal (strength {strength}) is offered. This is a correlational summary only and "
            "asserts no causation."
        ),
        OutcomeLearningSignalKind.NO_SIGNAL: (
            " No direction is clearly dominant across the repeated trials, so no evidence is "
            "proposed. A minority of adverse outcomes never penalises a belief on its own."
        ),
    }[kind]
    return head + tail

"""Versioned registry defaults: belief_type_registry, source_type reliability,
and the canonical belief_key alias registry.

Blueprint section 3.4.4 "Canonical belief_type Registry", section 6.2.1
"Default Source Reliability Values", and section 5.2 "Belief-Key
Canonicalization & Deduplication". These are frozen, versioned lookup
tables, not scattered constants (section 3.4.3) -- code that needs a decay
lambda, expected source-type count, sensitivity default, persistence
default, source reliability, or a known belief_key alias must resolve it
from here by the recorded registry/scoring version, never hard-code it
inline.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.common.enums import (
    BeliefStatus,
    BeliefType,
    PersistencePolicy,
    RecommendationRiskTier,
    SensitivityClass,
    SourceType,
)

BELIEF_TYPE_REGISTRY_VERSION = "belief-types-0.6"


@dataclass(frozen=True)
class BeliefTypeDefaults:
    default_decay_lambda: float
    expected_source_types: int
    default_sensitivity_class: SensitivityClass
    default_persistence_policy: PersistencePolicy


BELIEF_TYPE_REGISTRY: dict[BeliefType, BeliefTypeDefaults] = {
    BeliefType.BEHAVIORAL_TENDENCY: BeliefTypeDefaults(
        0.015, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.ROUTINE_OR_PREFERENCE: BeliefTypeDefaults(
        0.015, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.COMMUNICATION_OR_LEARNING_PREFERENCE: BeliefTypeDefaults(
        0.005, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CURRENT_STATE_RELATED: BeliefTypeDefaults(
        0.080, 1, SensitivityClass.NORMAL, PersistencePolicy.SHORT_TERM
    ),
    BeliefType.GOAL_OR_INTENTION: BeliefTypeDefaults(
        0.020, 1, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CONSTRAINT_OR_AVERSION: BeliefTypeDefaults(
        0.020, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CROSS_CONTEXT_TENDENCY: BeliefTypeDefaults(
        0.010, 3, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.RECOMMENDATION_RESPONSE_PATTERN: BeliefTypeDefaults(
        0.020, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE: BeliefTypeDefaults(
        0.030, 4, SensitivityClass.RESTRICTED, PersistencePolicy.DO_NOT_PERSIST
    ),
}

# Matches the scoring_config example's scoring_version in blueprint section 5.3;
# source reliability defaults are versioned as part of scoring_config (section 3.4.3).
SOURCE_RELIABILITY_SCORING_VERSION = "belief-score-0.6"

SOURCE_TYPE_RELIABILITY: dict[SourceType, float] = {
    SourceType.EXPLICIT_USER_CORRECTION: 1.00,
    SourceType.RECORDED_EVENT: 0.95,
    SourceType.EXPLICIT_USER_STATEMENT: 0.85,
    SourceType.REPEATED_PATTERN_SUMMARY: 0.80,
    SourceType.MODEL_OBSERVATION: 0.70,
    SourceType.LLM_INFERENCE: 0.55,
    SourceType.UNVERIFIED_HYPOTHESIS: 0.35,
}

# create-belief-evidence's completion check (skills/shared/create-belief-evidence/SKILL.md):
# "source_reliability matching (or explicitly justifying a deviation from) the
# versioned default for that type." This tolerance matches the one already
# enforced independently by that skill's scripts/validate_evidence.py, so a
# proposal that passes src/beliefs/models.py's validation also passes that
# skill's deterministic check, and vice versa.
SOURCE_RELIABILITY_TOLERANCE = 0.05

CANONICAL_BELIEF_KEY_REGISTRY_VERSION = "belief-key-canon-0.6"

# Known, pre-vetted (belief_type, proposed_key) -> canonical_key aliases
# (blueprint section 5.2). This is deliberately a small, hand-curated
# exact-match table, not a similarity/embedding index: the blueprint allows
# optional lightweight embeddings for this narrow deduplication task, but
# does not require them, and a deterministic exact-match registry is the
# conservative default this phase implements -- see
# src/beliefs/canonicalization.py's own module docstring for why an
# unmatched key is never guessed at. Only genuine aliases belong here (never
# a key mapped to itself); adding an entry is itself the "backend policy"
# decision that a pair of belief_key spellings mean the same thing.
CANONICAL_BELIEF_KEY_ALIASES: dict[tuple[BeliefType, str], str] = {
    (BeliefType.BEHAVIORAL_TENDENCY, "more_consistent_after_work_workouts"): "higher_adherence_after_work",
    (BeliefType.BEHAVIORAL_TENDENCY, "prefers_evening_exercise_sessions"): "higher_adherence_after_work",
}

# ------------------------------- recommendation context / risk policy (6.4) --

# Blueprint section 6.4 "Recommendation Risk Tiers & Exploration Policy":
# risk assignment is a backend policy lookup performed before candidate
# ranking. "The LLM may label the semantic recommendation context, but it may
# not freely invent, lower, or override risk_tier." These two frozen tables
# are that backend policy. They are versioned exactly like every other
# registry here; changing an entry is itself the policy decision.
RECOMMENDATION_CONTEXT_POLICY_VERSION = "rec-context-policy-0.6"
RISK_DOMAIN_POLICY_VERSION = "risk-domain-policy-0.6"

# The belief_type that must never influence a recommendation regardless of
# context (blueprint section 6.5: "numerically strong but blocked because
# sensitivity_class is sensitive/restricted"; sensitive_or_high_impact_inference
# beliefs default to RESTRICTED sensitivity in BELIEF_TYPE_REGISTRY above).
_SENSITIVE_BELIEF_TYPE = BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE

# Everyday, low-sensitivity belief types a conservative domain-level fallback
# is willing to consider when no exact context policy exists.
_CONSERVATIVE_DOMAIN_BELIEF_TYPES: frozenset[BeliefType] = frozenset(
    {
        BeliefType.ROUTINE_OR_PREFERENCE,
        BeliefType.COMMUNICATION_OR_LEARNING_PREFERENCE,
        BeliefType.CONSTRAINT_OR_AVERSION,
    }
)


@dataclass(frozen=True)
class RecommendationContextPolicy:
    """One row of ``recommendation_context_policy`` (blueprint section 6.4).

    ``allowed_belief_types`` empty means "no positive allow-list, so any
    belief_type not in ``disallowed_belief_types`` passes the type gate";
    a non-empty set means the belief_type must be listed. ``disallowed``
    always wins.

    Deliberate modeling choice vs. the blueprint's reference contract: the
    blueprint's ``recommendation_context_policy`` code block lists a single
    ``max_allowed_belief_status`` field, which presumes a total ordering over
    the six-value ``status`` enum that the blueprint never actually defines.
    This foundation models the same intent (section 6.5: medium/high-risk
    policy "can require validated status") as ``permitted_belief_statuses``,
    an explicit set of accepted statuses -- deterministic without inventing
    an ordering. Flagged for the blueprint's own maintainers rather than
    silently resolved, the same way ``enums.UnknownTypeBehavior`` documents a
    source-document inconsistency.

    Not yet modeled (deferred to the recommendation-loop phase, per section
    6.4 and roadmap section 11/12.5): ``created_at`` and, on
    ``RiskDomainPolicy``, ``high_risk_categories`` / ``medium_risk_categories``
    (category escalation) and ``manual_review_resolution_modes``.
    """

    context_key: str
    domain_key: str
    default_risk_tier: RecommendationRiskTier
    allowed_belief_types: frozenset[BeliefType]
    disallowed_belief_types: frozenset[BeliefType]
    permitted_belief_statuses: frozenset[BeliefStatus]
    allow_exploration: bool
    requires_user_confirmation: bool


@dataclass(frozen=True)
class RiskDomainPolicy:
    """One row of ``risk_domain_policy`` (blueprint section 6.4): the
    versioned fallback used only when no exact context policy exists. It
    "owns domain-level unknown-context defaults" -- never a way to lower
    risk below what an exact policy would have set."""

    domain_key: str
    default_unknown_context_risk_tier: RecommendationRiskTier
    requires_manual_review: bool


_EVERYDAY_STATUSES: frozenset[BeliefStatus] = frozenset(
    {BeliefStatus.CANDIDATE, BeliefStatus.PROVISIONAL, BeliefStatus.VALIDATED}
)
_MATURE_STATUSES: frozenset[BeliefStatus] = frozenset({BeliefStatus.PROVISIONAL, BeliefStatus.VALIDATED})
_VALIDATED_ONLY: frozenset[BeliefStatus] = frozenset({BeliefStatus.VALIDATED})

_ALL_NON_SENSITIVE_BELIEF_TYPES: frozenset[BeliefType] = frozenset(
    bt for bt in BeliefType if bt is not _SENSITIVE_BELIEF_TYPE
)

RECOMMENDATION_CONTEXT_POLICY: dict[str, RecommendationContextPolicy] = {
    "fitness_scheduling": RecommendationContextPolicy(
        context_key="fitness_scheduling",
        domain_key="fitness",
        default_risk_tier=RecommendationRiskTier.LOW,
        allowed_belief_types=_ALL_NON_SENSITIVE_BELIEF_TYPES,
        disallowed_belief_types=frozenset({_SENSITIVE_BELIEF_TYPE}),
        permitted_belief_statuses=_EVERYDAY_STATUSES,
        allow_exploration=True,
        requires_user_confirmation=False,
    ),
    "habit_nudge": RecommendationContextPolicy(
        context_key="habit_nudge",
        domain_key="fitness",
        default_risk_tier=RecommendationRiskTier.LOW,
        allowed_belief_types=_ALL_NON_SENSITIVE_BELIEF_TYPES,
        disallowed_belief_types=frozenset({_SENSITIVE_BELIEF_TYPE}),
        permitted_belief_statuses=_EVERYDAY_STATUSES,
        allow_exploration=True,
        requires_user_confirmation=False,
    ),
    "nutrition_guidance": RecommendationContextPolicy(
        context_key="nutrition_guidance",
        domain_key="health",
        default_risk_tier=RecommendationRiskTier.MEDIUM,
        allowed_belief_types=frozenset(
            {
                BeliefType.ROUTINE_OR_PREFERENCE,
                BeliefType.BEHAVIORAL_TENDENCY,
                BeliefType.GOAL_OR_INTENTION,
                BeliefType.CONSTRAINT_OR_AVERSION,
                BeliefType.COMMUNICATION_OR_LEARNING_PREFERENCE,
            }
        ),
        disallowed_belief_types=frozenset({_SENSITIVE_BELIEF_TYPE}),
        permitted_belief_statuses=_MATURE_STATUSES,
        allow_exploration=False,
        requires_user_confirmation=True,
    ),
    "mental_health_support": RecommendationContextPolicy(
        context_key="mental_health_support",
        domain_key="health",
        default_risk_tier=RecommendationRiskTier.HIGH,
        allowed_belief_types=frozenset(
            {
                BeliefType.COMMUNICATION_OR_LEARNING_PREFERENCE,
                BeliefType.CONSTRAINT_OR_AVERSION,
                BeliefType.ROUTINE_OR_PREFERENCE,
            }
        ),
        disallowed_belief_types=frozenset(
            {_SENSITIVE_BELIEF_TYPE, BeliefType.CURRENT_STATE_RELATED}
        ),
        permitted_belief_statuses=_VALIDATED_ONLY,
        allow_exploration=False,
        requires_user_confirmation=True,
    ),
    "financial_planning": RecommendationContextPolicy(
        context_key="financial_planning",
        domain_key="finance",
        default_risk_tier=RecommendationRiskTier.HIGH,
        allowed_belief_types=frozenset(
            {BeliefType.CONSTRAINT_OR_AVERSION, BeliefType.GOAL_OR_INTENTION}
        ),
        disallowed_belief_types=frozenset({_SENSITIVE_BELIEF_TYPE}),
        permitted_belief_statuses=_VALIDATED_ONLY,
        allow_exploration=False,
        requires_user_confirmation=True,
    ),
}

RISK_DOMAIN_POLICY: dict[str, RiskDomainPolicy] = {
    "fitness": RiskDomainPolicy("fitness", RecommendationRiskTier.MEDIUM, requires_manual_review=False),
    "health": RiskDomainPolicy("health", RecommendationRiskTier.HIGH, requires_manual_review=True),
    "finance": RiskDomainPolicy("finance", RecommendationRiskTier.HIGH, requires_manual_review=True),
}

# The domain-fallback policy is not a hand-written row per domain: it is
# derived deterministically from the domain's RiskDomainPolicy so a new
# domain cannot accidentally ship a lenient belief-type gate. Conservative
# by construction: only everyday, non-sensitive belief types, validated
# status only at HIGH risk, no exploration, confirmation always required.
def domain_fallback_context_policy(domain: RiskDomainPolicy) -> RecommendationContextPolicy:
    tier = domain.default_unknown_context_risk_tier
    return RecommendationContextPolicy(
        context_key=f"{domain.domain_key}:__unknown_context__",
        domain_key=domain.domain_key,
        default_risk_tier=tier,
        allowed_belief_types=_CONSERVATIVE_DOMAIN_BELIEF_TYPES,
        disallowed_belief_types=frozenset({_SENSITIVE_BELIEF_TYPE}),
        permitted_belief_statuses=(
            _VALIDATED_ONLY if tier is RecommendationRiskTier.HIGH else _MATURE_STATUSES
        ),
        allow_exploration=False,
        requires_user_confirmation=True,
    )


# ------------------------------ recommendation ranking heuristic (MVP, 6.4) --

# The deterministic candidate-ranking heuristic behind ``ranking_score`` on a
# recommendation. Blueprint section 3.2: "rank candidates using auditable
# heuristic scores" until learned ranking has data; section 12.5: "policy
# constraints so weak inferences cannot drive high-impact recommendations
# alone". Versioned like every other scoring/policy table so a historical
# recommendation's score is replayable.
RECOMMENDATION_RANKING_VERSION = "rec-ranking-0.6"


@dataclass(frozen=True)
class RecommendationRankingWeights:
    """``ranking_score = confidence * status_weight * diversity_factor``,
    where ``diversity_factor = diversity_floor + (1 - diversity_floor) *
    min(1, effective_evidence_count / expected_source_types)``.

    ``status_weight`` deliberately has no entry for ``outdated`` / ``rejected``
    / ``contested`` -- those never reach ranking (``authorize_beliefs_for_context``
    excludes them), and a missing key scores 0.0 as a fail-safe.
    """

    status_weight: dict[BeliefStatus, float]
    diversity_floor: float


RECOMMENDATION_RANKING = RecommendationRankingWeights(
    status_weight={
        BeliefStatus.CANDIDATE: 0.50,
        BeliefStatus.PROVISIONAL: 0.75,
        BeliefStatus.VALIDATED: 1.00,
    },
    diversity_floor=0.5,
)

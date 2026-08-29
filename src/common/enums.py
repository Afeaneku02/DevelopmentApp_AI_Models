"""Canonical closed enums for the Better You adaptive user model.

Every value here is frozen by Blueprint v0.6.2. Unknown values are rejected by
construction (Pydantic validates enum fields automatically); do not add a
value here without a corresponding registry-version bump in `registry.py`
and a blueprint decision-log entry, per section 3.4.3 "Versioned Scoring
Configuration" and section 10.4's canonical-registry contract.
"""
from __future__ import annotations

from enum import Enum


class BeliefType(str, Enum):
    """The nine-value canonical belief_type vocabulary (blueprint section 3.4.4)."""

    BEHAVIORAL_TENDENCY = "behavioral_tendency"
    ROUTINE_OR_PREFERENCE = "routine_or_preference"
    COMMUNICATION_OR_LEARNING_PREFERENCE = "communication_or_learning_preference"
    CURRENT_STATE_RELATED = "current_state_related"
    GOAL_OR_INTENTION = "goal_or_intention"
    CONSTRAINT_OR_AVERSION = "constraint_or_aversion"
    CROSS_CONTEXT_TENDENCY = "cross_context_tendency"
    RECOMMENDATION_RESPONSE_PATTERN = "recommendation_response_pattern"
    SENSITIVE_OR_HIGH_IMPACT_INFERENCE = "sensitive_or_high_impact_inference"


class SourceType(str, Enum):
    """The seven-value canonical evidence source_type enum (blueprint section 6.2.1)."""

    EXPLICIT_USER_CORRECTION = "explicit_user_correction"
    RECORDED_EVENT = "recorded_event"
    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    REPEATED_PATTERN_SUMMARY = "repeated_pattern_summary"
    MODEL_OBSERVATION = "model_observation"
    LLM_INFERENCE = "llm_inference"
    UNVERIFIED_HYPOTHESIS = "unverified_hypothesis"


class BeliefStatus(str, Enum):
    """The six-value belief lifecycle status enum (blueprint section 3.4.2)."""

    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    CONTESTED = "contested"
    OUTDATED = "outdated"
    REJECTED = "rejected"


class Direction(str, Enum):
    """Evidence direction relative to the belief it is attached to."""

    SUPPORT = "support"
    CONTRADICT = "contradict"


class AggregationMode(str, Enum):
    """belief_evidence aggregation mode (blueprint section 6.1.2). ``leaf_default``
    is always the safe fallback; ``aggregate_replacement`` requires explicit
    backend authorization."""

    LEAF_DEFAULT = "leaf_default"
    AGGREGATE_REPLACEMENT = "aggregate_replacement"


class AggregationReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CanonicalizationDecision(str, Enum):
    """belief_key canonicalization outcome (blueprint section 5.2): a
    proposal (an extractor/LLM's suggestion) or an authorized record (the
    backend's final decision) both use this same four-value vocabulary.
    ``merge`` is a value a proposal may request, but backend authorization
    logic never grants it automatically -- see
    ``src.beliefs.canonicalization.authorize_belief_key_canonicalization()``."""

    KEEP_SEPARATE = "keep_separate"
    ALIAS = "alias"
    MERGE = "merge"
    MANUAL_REVIEW = "manual_review"


class SensitivityClass(str, Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class RecommendationRiskTier(str, Enum):
    """recommendation_context_policy.default_risk_tier / the final resolved
    ``risk_tier`` on a recommendation (blueprint section 6.4). Backend policy
    owns this value end to end: an LLM may label the semantic recommendation
    context, but it may never invent, lower, or override the tier."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextResolutionPath(str, Enum):
    """How ``risk_tier`` for a recommendation context was resolved, persisted
    with the recommendation for auditability (blueprint section 6.4:
    "normalize context_key -> exact context policy -> parent/domain
    risk_domain_policy -> global conservative fallback")."""

    EXACT_CONTEXT_POLICY = "exact_context_policy"
    RISK_DOMAIN_POLICY = "risk_domain_policy"
    GLOBAL_CONSERVATIVE_FALLBACK = "global_conservative_fallback"


class RiskResolutionPath(str, Enum):
    """recommendations.risk_resolution_path (blueprint section 5, section 6.4):
    the path by which a recommendation's ``risk_tier`` was assigned. The
    first three mirror ``ContextResolutionPath``; the last three are only
    reached once a manual-review gate has been resolved (not in this MVP)."""

    EXACT_CONTEXT = "exact_context"
    DOMAIN_POLICY = "domain_policy"
    GLOBAL_FALLBACK = "global_fallback"
    MANUAL_REVIEW = "manual_review"
    USER_CONFIRMATION = "user_confirmation"
    DOMAIN_APPROVAL = "domain_approval"


class RecommendationReviewStatus(str, Enum):
    """recommendations.review_status (blueprint section 5). ``not_required``
    is the only value an automatically issued recommendation may carry;
    every other value implies a manual-review gate that a non-LLM backend
    actor must resolve."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    USER_CONFIRMED = "user_confirmed"
    DOMAIN_POLICY_APPROVED = "domain_policy_approved"


class ResolutionMode(str, Enum):
    """recommendations.required_resolution_mode / actual_resolution_mode
    (blueprint section 5, section 6.4). Backend policy selects the required
    mode by risk tier; an LLM may never choose or downgrade it."""

    REVIEWER = "reviewer"
    EXPLICIT_USER_CONFIRMATION = "explicit_user_confirmation"
    DOMAIN_POLICY_APPROVAL = "domain_policy_approval"


class OutcomeFollowed(str, Enum):
    """recommendation_outcomes.followed (blueprint section 5, section 12.6:
    "Define what counts as followed, ignored, partially followed ... and
    unknown"). This is the user's *behavior* relative to the recommendation,
    kept separate from ``OutcomeResult`` (what happened) and from
    ``user_feedback`` (what the user said)."""

    FOLLOWED = "followed"
    PARTIALLY_FOLLOWED = "partially_followed"
    NOT_FOLLOWED = "not_followed"
    IGNORED = "ignored"
    UNKNOWN = "unknown"


class OutcomeResult(str, Enum):
    """recommendation_outcomes.result (blueprint section 5, section 12.6).
    A classification of what was observed after the recommendation -- never
    a causal claim that the recommendation produced it (section F: "Do not
    assume correlation proves why a recommendation worked")."""

    SUCCESSFUL = "successful"
    MIXED = "mixed"
    UNSUCCESSFUL = "unsuccessful"
    NOT_YET_KNOWN = "not_yet_known"
    UNKNOWN = "unknown"


class ContextEligibilityReason(str, Enum):
    """Why one belief was or was not authorized for use in a given
    recommendation context (blueprint sections 6.4 and 6.5). Every value
    except ``allowed`` is a deterministic block reason."""

    ALLOWED = "allowed"
    BLOCKED_INCOMPLETE_POLICY_RESOLUTION = "blocked_incomplete_policy_resolution"
    BLOCKED_LOCKED = "blocked_locked_until_recompute"
    BLOCKED_STATUS = "blocked_status"
    BLOCKED_STATUS_NOT_PERMITTED_BY_CONTEXT = "blocked_status_not_permitted_by_context"
    BLOCKED_BELIEF_TYPE = "blocked_belief_type"
    BLOCKED_SENSITIVITY = "blocked_sensitivity_class"
    BLOCKED_DISALLOWED_CONTEXT = "blocked_disallowed_context"
    BLOCKED_NOT_IN_ALLOWED_CONTEXTS = "blocked_not_in_allowed_contexts"


class PersistencePolicy(str, Enum):
    SESSION = "session"
    SHORT_TERM = "short_term"
    RETAINED = "retained"
    DO_NOT_PERSIST = "do_not_persist"


class LinkRole(str, Enum):
    """observation_events.link_role (blueprint section 5, observation_events)."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"


class DeletionRequestedScope(str, Enum):
    ALL_MODEL_DATA = "all_model_data"
    RAW_CONTENT = "raw_content"
    BELIEFS = "beliefs"
    CONTEXT = "context"
    DATE_RANGE = "date_range"


class DeletionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResetScope(str, Enum):
    ALL_BELIEFS = "all_beliefs"
    CONTEXT = "context"
    SELECTED_KEYS = "selected_keys"


class RecomputeStatus(str, Enum):
    """scoring_recompute_status on deletion/reset requests (blueprint section 6.0.1-6.0.2)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UnknownTypeBehavior(str, Enum):
    """belief_type_registry.unknown_type_behavior.

    NOTE: the blueprint's own field-list comment for this column (section
    5.1.11) lists exactly ``reject | mapped_reviewed | manual_review``, but
    the worked belief_type_registry JSON example (section 5.3) instead shows
    the single string ``"reject_or_review"``, which matches neither of those
    three values. This is an inconsistency in the source document, not a
    modeling choice made here. We follow the more detailed field-list
    contract (three distinct values) since section 10.4 treats the registry
    as authoritative and reviewable; flagged for the blueprint's own
    maintainers rather than silently resolved.
    """

    REJECT = "reject"
    MAPPED_REVIEWED = "mapped_reviewed"
    MANUAL_REVIEW = "manual_review"

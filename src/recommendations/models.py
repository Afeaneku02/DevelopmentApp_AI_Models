"""Persisted ``recommendations`` record (blueprint section 5 "recommendations",
section 6.4).

This MVP is deterministic: candidate actions are derived from beliefs by a
fixed template table and ranked by an auditable heuristic
(``src/recommendations/engine.py``); no LLM is called, so ``prompt_version``
is always ``None`` and ``model_version`` names the deterministic engine.

The record keeps every field needed to replay the decision later:

- the resolved risk context: ``risk_tier``, ``risk_policy_context_key``,
  ``risk_resolution_path``, and the two exact policy versions used
  (``risk_policy_version`` / ``risk_domain_policy_version``) -- the section
  14.1 contract test "persisted recommendations record review_required,
  review_status, risk_resolution_path, and the exact risk policy/domain
  policy versions used";
- the manual-review gate: ``review_required`` / ``review_status`` /
  ``required_resolution_mode`` (section 6.4: ``requires_manual_review=true``
  blocks automatic issuance until an allowed resolution path completes; an
  LLM can never self-approve or downgrade the mode);
- the frozen decision state: ``frozen_belief_state`` captures every candidate
  belief (authorized or blocked) exactly as it was, because "live belief
  rows may change later" (section 5 audit rule);
- the ranking trace: ``candidate_trace`` (every candidate action considered,
  its score components, and which was selected) and ``blocked_beliefs``
  (each excluded belief and the deterministic policy reason).

Fields beyond the blueprint's section 5 ``recommendations`` list, added for
this deterministic MVP and kept additive (they never relax a blueprint
contract): ``proposed_context_key`` (the raw string the caller passed,
alongside the resolved ``recommendation_context``), ``rationale``,
``candidate_trace``, ``blocked_beliefs``, and ``ranking_policy_version`` (the
heuristic's version; ``scoring_version`` stays ``belief-score-0.6`` because
the numbers it multiplies -- belief ``confidence`` -- come from there).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.enums import (
    ContextEligibilityReason,
    OutcomeFollowed,
    OutcomeResult,
    RecommendationReviewStatus,
    RecommendationRiskTier,
    ResolutionMode,
    RiskResolutionPath,
)
from src.common.versioning import VersionedModel


class RankedCandidate(BaseModel):
    """One deterministic candidate action considered during ranking."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)
    belief_ids: list[str] = Field(min_length=1)
    ranking_score: float = Field(ge=0.0)
    components: dict[str, float]
    selected: bool


class BlockedBelief(BaseModel):
    """One candidate belief that policy excluded from the recommendation."""

    model_config = ConfigDict(extra="forbid")

    belief_id: str = Field(min_length=1)
    belief_key: str = Field(min_length=1)
    reason: ContextEligibilityReason
    detail: str


class UserRecommendation(VersionedModel):
    """A single persisted recommendation decision. Construct only via
    ``src.recommendations.engine.generate_recommendation`` -- direct
    construction is for deserializing an already-decided row."""

    recommendation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    goal: str | None = None

    recommendation: str = Field(min_length=1)
    proposed_context_key: str = Field(min_length=1)
    recommendation_context: str = Field(min_length=1)  # the resolved / normalized context_key

    risk_tier: RecommendationRiskTier
    risk_policy_context_key: str = Field(min_length=1)
    risk_policy_version: str = Field(min_length=1)
    risk_domain_policy_version: str = Field(min_length=1)
    risk_resolution_path: RiskResolutionPath

    review_required: bool
    review_status: RecommendationReviewStatus
    required_resolution_mode: ResolutionMode | None = None
    actual_resolution_mode: ResolutionMode | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    review_notes_or_policy_reference: str | None = None
    requires_user_confirmation: bool

    exploration_applied: bool = False

    profile_snapshot_id: str | None = None
    frozen_belief_state: dict[str, Any]
    belief_ids_used: list[str] = Field(default_factory=list)

    expected_outcome: str | None = None
    ranking_score: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)

    rationale: str = Field(min_length=1)
    candidate_trace: list[RankedCandidate] = Field(default_factory=list)
    blocked_beliefs: list[BlockedBelief] = Field(default_factory=list)

    ranking_policy_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _decision_state_is_frozen(self) -> "UserRecommendation":
        # Section 5 audit rule: a recommendation must preserve the exact
        # user-model state that informed it. If beliefs drove the decision,
        # their frozen state (or a profile snapshot) must be present.
        if self.belief_ids_used and self.profile_snapshot_id is None and not self.frozen_belief_state:
            raise ValueError(
                "belief_ids_used is non-empty but neither profile_snapshot_id nor frozen_belief_state "
                "was recorded; the decision state is not auditable"
            )
        missing = [bid for bid in self.belief_ids_used if bid not in self.frozen_belief_state]
        if missing and self.profile_snapshot_id is None:
            raise ValueError(f"belief_ids_used not present in frozen_belief_state: {missing}")
        return self

    @model_validator(mode="after")
    def _exploration_off_while_review_required(self) -> "UserRecommendation":
        # Section 6.4 / the recommendations schema: exploration_applied is
        # false whenever review_required is true.
        if self.review_required and self.exploration_applied:
            raise ValueError("exploration_applied must be false when review_required is true")
        return self

    @model_validator(mode="after")
    def _review_fields_consistent(self) -> "UserRecommendation":
        if self.review_required:
            if self.review_status == RecommendationReviewStatus.NOT_REQUIRED:
                raise ValueError("review_required is true but review_status is 'not_required'")
            if self.required_resolution_mode is None:
                raise ValueError("review_required is true but required_resolution_mode is unset")
        else:
            if self.review_status != RecommendationReviewStatus.NOT_REQUIRED:
                raise ValueError(
                    f"review_required is false but review_status is {self.review_status.value!r}"
                )
            if self.required_resolution_mode is not None or self.actual_resolution_mode is not None:
                raise ValueError("review_required is false but a resolution mode is set")
        return self

    @model_validator(mode="after")
    def _resolution_actor_recorded(self) -> "UserRecommendation":
        if self.actual_resolution_mode is not None and (self.resolved_by is None or self.resolved_at is None):
            raise ValueError("actual_resolution_mode is set but resolved_by/resolved_at is missing")
        return self

    @model_validator(mode="after")
    def _selected_trace_matches_output(self) -> "UserRecommendation":
        selected = [c for c in self.candidate_trace if c.selected]
        if len(selected) > 1:
            raise ValueError("more than one candidate marked selected")
        if selected:
            if selected[0].action != self.recommendation:
                raise ValueError("selected candidate action does not match recommendation text")
            if sorted(selected[0].belief_ids) != sorted(self.belief_ids_used):
                raise ValueError("selected candidate belief_ids do not match belief_ids_used")
            if selected[0].ranking_score != self.ranking_score:
                raise ValueError("selected candidate ranking_score does not match recommendation ranking_score")
        elif self.belief_ids_used:
            raise ValueError("belief_ids_used is non-empty but no candidate is marked selected")
        return self


class RecommendationOutcome(VersionedModel):
    """One observation of what happened after a persisted recommendation
    (blueprint section 5 "recommendation_outcomes", section 12.6).

    Purely descriptive -- it records *what was observed*, never *why*. There
    is deliberately no field linking an outcome to a belief or asserting the
    recommendation caused the result (section F: "Do not assume correlation
    proves why a recommendation worked"). The only link is
    ``recommendation_id``. Mapping outcomes back to evidence and updating
    belief confidence is a separate, later step and does not happen here.

    Append-only: many outcomes may reference the same ``recommendation_id``
    (a first report, a later follow-up, a measured signal arriving after the
    user's feedback); only ``outcome_id`` is unique. Recording an outcome
    never mutates the recommendation it points at -- the recommendation's
    ``frozen_belief_state`` and every other field stay exactly as decided.

    Three independent signals, kept separate on purpose (section 12.6:
    "Separate user feedback from measured behavior"):

    - ``followed`` -- did the user act on the recommendation (behavior)?
    - ``result`` -- how did things turn out (classification)?
    - ``user_feedback`` -- what did the user say about it (free text)?
      with ``measured_result`` for any measured/behavioral signal.
    """

    outcome_id: str = Field(min_length=1)
    recommendation_id: str = Field(min_length=1)

    followed: OutcomeFollowed
    result: OutcomeResult
    user_feedback: str | None = None
    measured_result: str | None = None

    # Where this outcome record came from (e.g. "app_event", "user_survey",
    # "support_ticket", "manual"). Required so a later learning step can weigh
    # self-report against measured behavior.
    source: str = Field(min_length=1)

    # When the outcome was observed (may be earlier than, or absent when
    # different from, ``created_at``, which is when this row was written).
    observed_at: datetime | None = None
    created_at: datetime

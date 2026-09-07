"""Request models for the local API -- the *entire* client-fillable surface
of each write endpoint.

Every model sets ``extra="forbid"``: a payload that includes a backend-owned
field (any version tag, an evidence authorization field, a belief
``confidence``/``status``/``locked_until_recompute``, ...) is rejected with
422 rather than quietly accepted. The backend injects those fields itself in
``src.api.service``.

Field names deliberately mirror the corresponding ``tools/add_*`` CLI flags
and the domain models, so a caller who knows the CLI knows the API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.common.enums import (
    BeliefType,
    Direction,
    OutcomeFollowed,
    OutcomeResult,
    SourceType,
)


class ApiModel(BaseModel):
    """Base for every request model: unknown keys are a hard error."""

    model_config = ConfigDict(extra="forbid")


class EventIn(ApiModel):
    user_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    source: str = Field(min_length=1)
    timestamp: datetime | None = None
    raw_content: str | None = None
    structured_data: dict[str, Any] | None = None


class ObservationIn(ApiModel):
    observation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    event_ids: list[str] = Field(min_length=1)
    primary_event_id: str | None = None
    category: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    importance: float
    confidence: float
    created_at: datetime | None = None


class BeliefEvidenceIn(ApiModel):
    evidence_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    belief_id: str = Field(min_length=1)
    belief_type: BeliefType
    direction: Direction
    source_type: SourceType
    context_key: str = Field(min_length=1)
    strength: float
    model_version: str = Field(min_length=1)
    created_at: datetime | None = None


class RecomputeIn(ApiModel):
    # belief_id comes from the path
    user_id: str = Field(min_length=1)
    belief_type: BeliefType
    belief_key: str = Field(min_length=1)
    belief_value: Any
    as_of: datetime | None = None
    first_observed: datetime | None = None
    recompute_reason: str | None = None
    allow_no_evidence: bool = False


class RecommendationIn(ApiModel):
    recommendation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    context_key: str = Field(min_length=1)
    goal: str | None = None
    expected_outcome: str | None = None
    created_at: datetime | None = None


class RecommendationOutcomeIn(ApiModel):
    outcome_id: str = Field(min_length=1)
    recommendation_id: str = Field(min_length=1)
    followed: OutcomeFollowed
    result: OutcomeResult
    source: str = Field(min_length=1)
    user_feedback: str | None = None
    measured_result: str | None = None
    observed_at: datetime | None = None
    created_at: datetime | None = None

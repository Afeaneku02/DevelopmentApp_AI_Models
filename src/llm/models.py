from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# Bounded, per-request clarification context: the mentor's earlier clarifying
# questions in this interaction and the user's answers, oldest first. The
# client (Better You) holds and resends them - this service keeps no
# conversation state, never persists them, and never turns them into
# evidence. Both fields are untrusted data: `question` is model output echoed
# back by a client, `answer` is user text.
MAX_CLARIFICATION_TURNS = 2


class ClarificationTurn(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=500)

    @field_validator("question", "answer")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clarification text must not be blank")
        return value.strip()


class MentorRequest(StrictModel):
    context_key: str = Field(min_length=1, max_length=120)
    goal: str = Field(default="", max_length=500)
    question: str = Field(default="", max_length=500)
    clarifications: list[ClarificationTurn] = Field(default_factory=list, max_length=MAX_CLARIFICATION_TURNS)

    @field_validator("context_key")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context_key must not be blank")
        return value.strip()


class ApprovedBelief(StrictModel):
    belief_id: str
    key: str
    value: str
    confidence: float
    status: str
    evidence_ids: list[str]
    source_event_ids: list[str]
    versions: dict[str, str]


class MentorContext(StrictModel):
    snapshot_id: str
    context_key: str
    goal: str
    question: str
    risk_tier: Literal["low"] = "low"
    risk_policy_version: str
    risk_domain_policy_version: str
    beliefs: list[ApprovedBelief]
    constraints: list[str]
    clarifications: list[ClarificationTurn] = Field(default_factory=list, max_length=MAX_CLARIFICATION_TURNS)


class GuidanceRecommendation(StrictModel):
    action: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=700)
    grounded_in_belief_ids: list[str] = Field(min_length=1, max_length=5)


class MentorGuidance(StrictModel):
    summary: str = Field(min_length=1, max_length=1000)
    recommendations: list[GuidanceRecommendation] = Field(max_length=3)
    clarifying_question: str | None = Field(max_length=500)
    needs_more_information: bool
    needs_web: bool

    @model_validator(mode="after")
    def consistent(self):
        if self.needs_more_information != bool(self.clarifying_question and self.clarifying_question.strip()):
            raise ValueError("clarification flags must agree")
        if (self.needs_more_information or self.needs_web) and self.recommendations:
            raise ValueError("unresolved information needs cannot issue recommendations")
        return self


class UsageTelemetry(StrictModel):
    provider: str
    model: str
    returned_model: str | None = None
    operation_id: str | None = None
    prompt_version: str = "mentor-guidance-2"
    request_id: str | None = None
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    success: bool = False
    failure_code: str | None = None
    estimated_cost_usd: float | None = None
    reserved_cost_usd: float = 0.0
    price_version: str = "openai-standard-2026-09-20"


class ProviderResult(StrictModel):
    guidance: MentorGuidance
    telemetry: UsageTelemetry


class MentorResponse(StrictModel):
    status: Literal["openai", "mock", "deterministic_fallback", "needs_more_information", "unavailable"]
    reason_code: str | None = None
    guidance: MentorGuidance
    snapshot_id: str | None = None
    telemetry: UsageTelemetry | None = None

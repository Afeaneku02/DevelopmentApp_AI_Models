from typing import Protocol
from src.llm.models import MentorContext, MentorGuidance, ProviderResult, UsageTelemetry


class ProviderUnavailable(Exception):
    """Only backend-owned codes; never carry an SDK exception or payload."""

    def __init__(self, code: str, telemetry: UsageTelemetry | None = None):
        super().__init__(code)
        self.code = code
        self.telemetry = telemetry


class MentorProvider(Protocol):
    def generate_guidance(self, context: MentorContext) -> ProviderResult: ...


class MockMentorProvider:
    def __init__(self, guidance: MentorGuidance | None = None):
        self.guidance = guidance
        self.contexts: list[MentorContext] = []

    def generate_guidance(self, context: MentorContext) -> ProviderResult:
        self.contexts.append(context)
        guidance = self.guidance or MentorGuidance(
            summary="A little more context would help choose a small next step.",
            recommendations=[], clarifying_question="What small change would you like to try?",
            needs_more_information=True, needs_web=False,
        )
        return ProviderResult(guidance=guidance, telemetry=UsageTelemetry(
            provider="mock", model="mock", success=True,
        ))

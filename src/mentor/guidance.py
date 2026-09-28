"""Snapshot assembly and policy enforcement, independent of the LLM provider."""
from __future__ import annotations

import hashlib
import json

from src.common.enums import PersistencePolicy
from src.llm.base import MentorProvider, ProviderUnavailable
from src.llm.models import ApprovedBelief, GuidanceRecommendation, MentorContext, MentorGuidance, MentorRequest, MentorResponse
from src.llm.validation import validate_guidance
from src.mentor.feedback import generate_mentor_feedback, _is_groundable, _support_is_solid, _is_high_stakes
from src.recommendations.context_policy import authorize_beliefs_for_context

VERSIONS = ("schema_version", "scoring_version", "canonicalizer_version", "policy_version", "belief_type_registry_version")


def clarification(reason: str) -> MentorResponse:
    return MentorResponse(status="needs_more_information", reason_code=reason, guidance=MentorGuidance(
        summary="There is not enough approved context for a small, low-risk suggestion.",
        recommendations=[], clarifying_question="What small scheduling or learning habit would you like help with?",
        needs_more_information=True, needs_web=False,
    ))


def build_context(repo, user_id: str, request: MentorRequest) -> MentorContext | MentorResponse:
    beliefs = [b for b in repo.list_latest_beliefs(user_id=user_id) if b.user_id == user_id]
    authorization = authorize_beliefs_for_context(beliefs, request.context_key)
    if (authorization.risk_tier.value != "low" or authorization.requires_manual_review
            or authorization.requires_user_confirmation):
        return clarification("policy_requires_resolution")
    # Clarification turns are untrusted client-supplied text like goal and
    # question, so they pass the same high-stakes gate before any of it can
    # reach a provider.
    clarification_text = " ".join(f"{t.question} {t.answer}" for t in request.clarifications)
    if _is_high_stakes(request.goal + " " + request.question + " " + clarification_text):
        return clarification("consequential_request")
    allowed = set(authorization.authorized_beliefs)
    selected = []
    for belief in sorted(beliefs, key=lambda b: (-b.confidence, b.belief_id)):
        if belief.belief_id not in allowed or not _is_groundable(belief):
            continue
        if belief.persistence_policy in {PersistencePolicy.SESSION, PersistencePolicy.DO_NOT_PERSIST}:
            continue
        evidence = [e for e in repo.list_active_evidence(user_id=user_id, belief_id=belief.belief_id)
                    if e.user_id == user_id and e.belief_id == belief.belief_id]
        if not _support_is_solid(belief, evidence):
            continue
        value = json.dumps(belief.belief_value, ensure_ascii=True, allow_nan=False)
        if len(value) > 500 or len(belief.belief_key) > 160 or len(belief.belief_id) > 160:
            continue
        if _is_high_stakes(belief.belief_key + " " + value):
            continue
        # IDs only, not raw events, observations, evidence text or reasoning.
        selected.append(ApprovedBelief(
            belief_id=belief.belief_id, key=belief.belief_key, value=value,
            confidence=belief.confidence, status=belief.status.value,
            evidence_ids=sorted(e.evidence_id for e in evidence),
            source_event_ids=sorted({eid for e in evidence for eid in e.source_event_ids}),
            versions={v: getattr(belief, v) for v in VERSIONS},
        ))
        if len(selected) == 5:
            break
    if not selected:
        return clarification("insufficient_authorized_evidence")
    context = MentorContext(
        snapshot_id="", context_key=authorization.context_key, goal=request.goal, question=request.question,
        risk_policy_version=authorization.risk_policy_version,
        risk_domain_policy_version=authorization.risk_domain_policy_version,
        beliefs=selected, constraints=["Only small reversible habits", "No tools or state writes", "Ask when uncertain"],
        # Per-request only: never stored, never evidence (see ClarificationTurn).
        clarifications=list(request.clarifications),
    )
    snapshot_id = hashlib.sha256(context.model_dump_json().encode()).hexdigest()
    return context.model_copy(update={"snapshot_id": snapshot_id})


def generate_guidance(repo, *, user_id: str, request: MentorRequest, provider: MentorProvider) -> MentorResponse:
    context = build_context(repo, user_id, request)
    if isinstance(context, MentorResponse):
        return context
    telemetry = None
    try:
        result = provider.generate_guidance(context)
        telemetry = result.telemetry
        guidance = validate_guidance(result.guidance, context)
        return MentorResponse(
            status="needs_more_information" if guidance.needs_more_information else
                   "unavailable" if guidance.needs_web else
                   "mock" if result.telemetry.provider == "mock" else "openai",
            reason_code="web_not_enabled" if guidance.needs_web else None,
            guidance=guidance, snapshot_id=context.snapshot_id, telemetry=telemetry,
        )
    except ProviderUnavailable as exc:
        reason, telemetry = exc.code, exc.telemetry
    except Exception:
        reason = "validation_failed"
        if telemetry:
            telemetry = telemetry.model_copy(update={"success": False, "failure_code": reason})
    baseline = generate_mentor_feedback(repo, user_id=user_id, context_key=request.context_key)
    approved = {b.belief_id for b in context.beliefs}
    recommendations = [GuidanceRecommendation(
        action=item.recommended_next_action, reason=item.message,
        grounded_in_belief_ids=item.grounded_in_belief_ids,
    ) for item in baseline.feedback
        if item.recommended_next_action and set(item.grounded_in_belief_ids) <= approved][:3]
    if not recommendations:
        result = clarification(reason)
        return result.model_copy(update={"status": "unavailable", "snapshot_id": context.snapshot_id, "telemetry": telemetry})
    return MentorResponse(
        status="deterministic_fallback", reason_code=reason, snapshot_id=context.snapshot_id, telemetry=telemetry,
        guidance=MentorGuidance(summary="Guidance from the deterministic mentor.", recommendations=recommendations,
                                clarifying_question=None, needs_more_information=False, needs_web=False),
    )

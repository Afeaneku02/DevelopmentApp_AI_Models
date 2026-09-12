"""First mentor feedback engine -- deterministic, read-only, no LLM.

``generate_mentor_feedback(repo, user_id=..., ...)`` turns a user's *already
stored* beliefs and evidence into structured mentor feedback. Every message
traces to one or more belief ids; nothing is invented about the user.

Safety rules, enforced here in code:

- **Only settled, positive beliefs.** A belief is used only if it is
  ``validated`` or ``provisional`` (never ``candidate``/``contested``/
  ``outdated``/``rejected``), not ``locked_until_recompute``, of a
  non-sensitive belief_type and sensitivity class, and above a small
  confidence floor.
- **Existing confidence/risk rules reused, not re-implemented.** When a
  ``context_key`` is supplied, the surviving beliefs go through
  ``authorize_beliefs_for_context`` (``src.recommendations.context_policy``)
  -- the same context/risk gate the recommendation engine uses -- and the
  resolved risk tier drives how directive the feedback is allowed to be.
- **Evidence must be clear.** Active supporting evidence must exist and must
  clearly outweigh contradicting evidence (both by count and by effective
  mass). Weak, contradictory, or missing evidence yields
  ``needs_more_data``.
- **No high-stakes life advice.** Feedback reflects small behavioural
  patterns back and, at most, suggests one small next step. Any item whose
  wording trips a high-stakes filter (careers, medical, financial, legal,
  relocation, education decisions, ...) is dropped entirely.
- **No overclaiming.** Confidence is capped by risk tier; a ``high``-risk
  context produces reflective observations only, never a recommended
  action.
- **Read-only.** Nothing is written. The result is a plain value the caller
  may serialise or discard.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from src.beliefs.models import BeliefEvidence, UserBelief
from src.common.enums import BeliefStatus, BeliefType, Direction, SensitivityClass
from src.recommendations.context_policy import authorize_beliefs_for_context

_GROUNDABLE_STATUSES = frozenset({BeliefStatus.VALIDATED, BeliefStatus.PROVISIONAL})
_MIN_CONFIDENCE = 0.20
_MIN_ACTIVE_EVIDENCE = 1
_MAX_ITEMS = 5

# Conservative default when no context_key is given: there is nothing to
# resolve a real risk tier from, so treat it as medium -- enough to reflect
# a pattern back, not enough to hand out a concrete next step.
_NO_CONTEXT_RISK_TIER: Literal["medium"] = "medium"

# Confidence ceilings by risk tier, so a message never sounds more certain
# than the model actually is.
_CONFIDENCE_CAP: dict[str, float] = {"low": 0.85, "medium": 0.70, "high": 0.55}

# Wording that would make this read like high-stakes life advice. Any
# feedback item containing one of these (in its message or its next action)
# is discarded -- this engine is not allowed to go there.
_HIGH_STAKES_PATTERNS: tuple[str, ...] = (
    "college", "university", "degree", "grad school", "drop out", "dropout",
    "quit your job", "quit my job", "resign", "new job", "change careers",
    "career change", "get promoted", "promotion",
    "medical", "medication", "meds", "doctor", "physician", "therapist",
    "therapy", "diagnos", "treatment", "prescription",
    "invest", "investment", "loan", "mortgage", "debt", "retirement",
    "stocks", "crypto", "savings account", "financial advisor",
    "divorce", "get married", "marry", "break up", "move out", "move in",
    "relocate", "move to ", "move abroad",
    "lawyer", "attorney", "sue ", "lawsuit", "immigration", "visa",
)

RiskTier = Literal["low", "medium", "high"]
FeedbackStatus = Literal["ready", "needs_more_data"]


class MentorFeedbackItem(BaseModel):
    message: str
    confidence: float
    grounded_in_belief_ids: list[str]
    why: str
    recommended_next_action: str | None = None
    risk_tier: RiskTier


class MentorFeedbackResult(BaseModel):
    status: FeedbackStatus
    feedback: list[MentorFeedbackItem] = Field(default_factory=list)
    needs_more_data_reason: str | None = None


# --------------------------------------------------------------- internals --


def _list_or_empty(list_fn: Callable[..., list[Any]], **kwargs: Any) -> list[Any]:
    """Call a ``Repository.list_*`` helper, treating a missing table (an old
    DB opened read-only) as an empty result rather than an error."""
    try:
        return list_fn(**kwargs)
    except sqlite3.OperationalError:
        return []


def _needs_more_data(reason: str) -> MentorFeedbackResult:
    return MentorFeedbackResult(status="needs_more_data", feedback=[], needs_more_data_reason=reason)


def _is_groundable(belief: UserBelief) -> bool:
    """The lifecycle gate: a belief settled and safe enough to say anything
    about, before any context/risk policy is applied."""
    if belief.locked_until_recompute:
        return False
    if belief.status not in _GROUNDABLE_STATUSES:
        return False
    if belief.belief_type is BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE:
        return False
    if belief.sensitivity_class is not SensitivityClass.NORMAL:
        return False
    if belief.confidence < _MIN_CONFIDENCE:
        return False
    return True


def _support_is_solid(belief: UserBelief, active: list[BeliefEvidence]) -> bool:
    """Active supporting evidence must exist and clearly outweigh
    contradicting evidence -- by row count and by effective mass."""
    if len(active) < _MIN_ACTIVE_EVIDENCE:
        return False
    support = sum(1 for e in active if e.direction is Direction.SUPPORT)
    contra = sum(1 for e in active if e.direction is Direction.CONTRADICT)
    if support == 0 or contra >= support:
        return False
    effective_support = belief.effective_support_count
    effective_contra = max(0.0, belief.effective_evidence_count - belief.effective_support_count)
    if effective_support <= effective_contra:
        return False
    return True


def _readable_subject(belief: UserBelief) -> str:
    key = belief.belief_key.replace("_", " ").strip()
    value = belief.belief_value
    if value is True:
        return key
    if value is False:
        return f"not {key}"
    if isinstance(value, str) and value.strip():
        return f"{key}: {value.strip()}"
    return key


def _evidence_summary(active: list[BeliefEvidence]) -> tuple[int, int, list[str]]:
    support = sum(1 for e in active if e.direction is Direction.SUPPORT)
    contra = sum(1 for e in active if e.direction is Direction.CONTRADICT)
    source_types = sorted({e.source_type.value for e in active})
    return support, contra, source_types


def _next_action(subject: str, risk_tier: str) -> str | None:
    if risk_tier == "low":
        return f'Consider one small, low-commitment action this week that leans into "{subject}".'
    if risk_tier == "medium":
        return (
            f'You might take a moment to notice whether "{subject}" is something you want to '
            "lean into -- no action needed yet."
        )
    return None  # high risk: reflective observation only, never a directive


def _message(subject: str, support: int, contra: int, risk_tier: str) -> str:
    signal = f"{support} supporting" + (f" and {contra} contradicting" if contra else "") + " signal(s)"
    if risk_tier == "high":
        return (
            f'Your recorded activity leans toward "{subject}" ({signal}). This is a higher-stakes '
            "area, so treat it only as an observation to sit with, not a nudge to act on."
        )
    return (
        f'Your recorded activity shows a fairly consistent pattern around "{subject}" ({signal}). '
        "That looks like a real thing you could build on, gently."
    )


def _is_high_stakes(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _HIGH_STAKES_PATTERNS)


def _feedback_item(
    belief: UserBelief,
    active: list[BeliefEvidence],
    *,
    risk_tier: str,
    context_label: str,
) -> MentorFeedbackItem:
    subject = _readable_subject(belief)
    support, contra, source_types = _evidence_summary(active)
    confidence = round(min(belief.confidence, _CONFIDENCE_CAP.get(risk_tier, 0.55)), 3)

    why = (
        f"Grounded in belief {belief.belief_id} (key {belief.belief_key!r}, value "
        f"{belief.belief_value!r}, status {belief.status.value}, model confidence "
        f"{belief.confidence:.2f}). Active evidence: {support} supporting / {contra} contradicting "
        f"across source type(s) {source_types}. The belief is not locked and passes the "
        f"{context_label} risk gate at the {risk_tier} tier. No claim beyond this observed pattern "
        "is made."
    )

    return MentorFeedbackItem(
        message=_message(subject, support, contra, risk_tier),
        confidence=confidence,
        grounded_in_belief_ids=[belief.belief_id],
        why=why,
        recommended_next_action=_next_action(subject, risk_tier),
        risk_tier=risk_tier,  # type: ignore[arg-type]
    )


def _diagnose_lifecycle(beliefs: list[UserBelief]) -> str:
    total = len(beliefs)
    locked = sum(1 for b in beliefs if b.locked_until_recompute)
    bad_status = sum(1 for b in beliefs if b.status not in _GROUNDABLE_STATUSES)
    sensitive = sum(
        1
        for b in beliefs
        if b.belief_type is BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE
        or b.sensitivity_class is not SensitivityClass.NORMAL
    )
    low_conf = sum(1 for b in beliefs if b.confidence < _MIN_CONFIDENCE)
    parts: list[str] = []
    if locked:
        parts.append(f"{locked} locked pending recompute")
    if bad_status:
        parts.append(f"{bad_status} not validated/provisional (candidate, contested, outdated, or rejected)")
    if sensitive:
        parts.append(f"{sensitive} of a sensitive belief type or sensitivity class")
    if low_conf:
        parts.append(f"{low_conf} below the {_MIN_CONFIDENCE} confidence floor")
    detail = "; ".join(parts) or "none are settled enough to ground feedback"
    return f"none of the {total} stored belief(s) are usable yet: {detail}"


def _diagnose_evidence(beliefs: list[UserBelief]) -> str:
    return (
        f"the {len(beliefs)} otherwise-usable belief(s) do not yet have clearly supportive "
        "evidence -- active support is missing, or contradicting signals are not outweighed"
    )


# ---------------------------------------------------------------- public --


def generate_mentor_feedback(
    repo: Any,
    *,
    user_id: str,
    context_key: str | None = None,
    goal_category: str | None = None,
    goal_title: str | None = None,
    as_of: datetime | None = None,
) -> MentorFeedbackResult:
    """Return structured mentor feedback for ``user_id`` from stored state.

    ``repo`` is only ever read from. ``context_key`` (when given) selects the
    context/risk policy that gates which beliefs may be used and how directive
    the feedback may be. ``goal_category`` / ``goal_title`` are accepted for
    forward compatibility and are not yet used to change the output beyond
    being unable to introduce any claim of their own.

    Returns ``status="needs_more_data"`` with a specific reason whenever the
    stored state is too thin, weak, contradictory, locked, or rejected to say
    anything responsibly.
    """
    _ = as_of or datetime.now(timezone.utc)  # reserved; keeps the signature stable
    _ = (goal_category, goal_title)  # accepted, intentionally inert for now

    beliefs: list[UserBelief] = _list_or_empty(repo.list_latest_beliefs, user_id=user_id)
    if not beliefs:
        return _needs_more_data("no beliefs are stored for this user yet")

    groundable = [b for b in beliefs if _is_groundable(b)]
    if not groundable:
        return _needs_more_data(_diagnose_lifecycle(beliefs))

    if context_key is not None and context_key.strip():
        authorization = authorize_beliefs_for_context(groundable, context_key)
        risk_tier = authorization.risk_tier.value
        context_label = repr(authorization.context_key)
        allowed_ids = set(authorization.authorized_beliefs)
        usable = [b for b in groundable if b.belief_id in allowed_ids]
        if not usable:
            reasons = sorted({d.reason.value for d in authorization.decisions if not d.allowed})
            return _needs_more_data(
                f"the {authorization.context_key!r} context/risk policy blocked all "
                f"{len(groundable)} otherwise-usable belief(s) "
                f"(reason(s): {', '.join(reasons) or 'unspecified'})"
            )
    else:
        risk_tier = _NO_CONTEXT_RISK_TIER
        context_label = "default (no context)"
        usable = groundable

    grounded: list[tuple[UserBelief, list[BeliefEvidence]]] = []
    for belief in usable:
        active = _list_or_empty(
            repo.list_active_evidence, user_id=user_id, belief_id=belief.belief_id
        )
        if _support_is_solid(belief, active):
            grounded.append((belief, active))

    if not grounded:
        return _needs_more_data(_diagnose_evidence(usable))

    grounded.sort(
        key=lambda pair: (pair[0].confidence, pair[0].effective_support_count), reverse=True
    )

    items: list[MentorFeedbackItem] = []
    for belief, active in grounded:
        item = _feedback_item(belief, active, risk_tier=risk_tier, context_label=context_label)
        if any(
            _is_high_stakes(text)
            for text in (item.message, item.why, item.recommended_next_action or "")
        ):
            continue
        items.append(item)
        if len(items) >= _MAX_ITEMS:
            break

    if not items:
        return _needs_more_data(
            "the usable belief(s) could not be phrased as feedback without straying into "
            "high-stakes advice"
        )

    return MentorFeedbackResult(status="ready", feedback=items)

"""Recommendation context / risk policy foundation (blueprint section 6.4-6.5).

The one job of this module: given a recommendation ``context_key`` and a set
of ``UserBelief`` rows, decide -- deterministically, by backend policy alone
-- which of those beliefs may be used as inputs to a recommendation in that
context, and at what risk tier.

The model-proposes / backend-authorizes boundary (the same structural
pattern as ``src/beliefs/canonicalization.py``):

- ``RecommendationContextProposal`` is the entire surface an LLM / extractor
  may fill in: a single ``proposed_context_key`` string, nothing else. It
  has no ``risk_tier``, ``default_risk_tier``, ``resolution_path``, or any
  policy-owned field, and ``extra="forbid"`` (inherited from
  ``VersionedModel``) means a payload that tries to smuggle one in fails
  construction rather than being quietly accepted. Blueprint section 6.4:
  "The LLM may label the semantic recommendation context, but it may not
  freely invent, lower, or override risk_tier."
- ``resolve_context_policy()`` turns a proposal (or a bare string) into a
  ``ResolvedContextPolicy`` using only the frozen, versioned
  ``RECOMMENDATION_CONTEXT_POLICY`` / ``RISK_DOMAIN_POLICY`` tables in
  ``src/common/registry.py``. Resolution order is exactly the blueprint's:
  normalize -> exact context policy -> parent/domain risk_domain_policy ->
  global conservative fallback. An unknown context can only ever resolve
  *up* in risk (to MEDIUM or HIGH), never down, and never non-deterministically.
- ``authorize_beliefs_for_context()`` applies that resolved policy plus the
  per-belief purpose-limitation checks (section 6.5) and returns a
  ``ContextAuthorizationResult`` recording every decision and both policy
  versions, so the authorization path can be replayed later.

Nothing here ranks, scores, explores, or issues a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from src.beliefs.models import UserBelief
from src.common.enums import (
    BeliefStatus,
    ContextEligibilityReason,
    ContextResolutionPath,
    RecommendationRiskTier,
)
from src.common.registry import (
    RECOMMENDATION_CONTEXT_POLICY,
    RECOMMENDATION_CONTEXT_POLICY_VERSION,
    RISK_DOMAIN_POLICY,
    RISK_DOMAIN_POLICY_VERSION,
    RecommendationContextPolicy,
    RiskDomainPolicy,
    domain_fallback_context_policy,
)
from src.common.versioning import VersionedModel

# Belief lifecycle statuses that are never a usable recommendation input,
# regardless of context (requirement: exclude locked / outdated / rejected).
# ``locked_until_recompute`` is checked separately since it is a flag, not a
# status. ``contested`` is excluded here too: a belief with live contradicting
# evidence is not a safe recommendation input while the contest is unresolved.
GLOBALLY_EXCLUDED_STATUSES: frozenset[BeliefStatus] = frozenset(
    {BeliefStatus.OUTDATED, BeliefStatus.REJECTED, BeliefStatus.CONTESTED}
)

# Sensitivity classes that block a belief from any recommendation context
# (blueprint section 6.5: "numerically strong but blocked because
# sensitivity_class is sensitive/restricted").
_BLOCKED_SENSITIVITY = {"sensitive", "restricted"}


class RecommendationContextProposal(VersionedModel):
    """The complete set of fields an LLM / extractor may propose for a
    recommendation context: just the semantic label. No risk field exists
    on this model by design -- see the module docstring."""

    proposed_context_key: str = Field(min_length=1)


def normalize_context_key(raw: str) -> str:
    """Deterministic context-key normalization (blueprint section 6.4 step 1:
    "normalize context_key"). Lowercased, trimmed, internal whitespace and
    dashes collapsed to single underscores, surrounding punctuation dropped.
    Pure and idempotent."""
    text = raw.strip().lower()
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "/", ".", ":"}:
            out.append("_")
        # any other character is dropped
    collapsed = "_".join(part for part in "".join(out).split("_") if part)
    return collapsed


def _domain_of(normalized_key: str) -> str:
    """The domain segment of a normalized context key -- everything before
    the first underscore, or the whole key if it has none."""
    return normalized_key.split("_", 1)[0]


@dataclass(frozen=True)
class ResolvedContextPolicy:
    """The backend's final, replayable decision for one recommendation
    context. Every field here is policy-owned; none is model-supplied."""

    proposed_context_key: str
    context_key: str
    risk_tier: RecommendationRiskTier
    resolution_path: ContextResolutionPath
    allow_exploration: bool
    requires_user_confirmation: bool
    requires_manual_review: bool
    # When True no belief is ever authorized: policy resolution was
    # incomplete (unknown context, no domain policy) so the conservative
    # global fallback blocks recommendation use entirely.
    blocks_all_beliefs: bool
    risk_policy_version: str
    risk_domain_policy_version: str
    _context_policy: RecommendationContextPolicy | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class BeliefContextDecision:
    """Per-belief outcome of ``authorize_beliefs_for_context()``."""

    belief_id: str
    belief_key: str
    allowed: bool
    reason: ContextEligibilityReason
    detail: str
    risk_tier: RecommendationRiskTier
    context_key: str
    risk_policy_version: str
    risk_domain_policy_version: str


@dataclass(frozen=True)
class ContextAuthorizationResult:
    """Everything ``authorize_beliefs_for_context()`` decided, including both
    policy versions and one ``BeliefContextDecision`` per input belief."""

    proposed_context_key: str
    context_key: str
    risk_tier: RecommendationRiskTier
    resolution_path: ContextResolutionPath
    allow_exploration: bool
    requires_user_confirmation: bool
    requires_manual_review: bool
    risk_policy_version: str
    risk_domain_policy_version: str
    decisions: list[BeliefContextDecision]

    @property
    def authorized_beliefs(self) -> list[str]:
        """belief_id of every belief that passed every gate."""
        return [d.belief_id for d in self.decisions if d.allowed]


def _resolve_from_exact_policy(policy: RecommendationContextPolicy, proposed: str) -> ResolvedContextPolicy:
    domain = RISK_DOMAIN_POLICY.get(policy.domain_key)
    return ResolvedContextPolicy(
        proposed_context_key=proposed,
        context_key=policy.context_key,
        risk_tier=policy.default_risk_tier,
        resolution_path=ContextResolutionPath.EXACT_CONTEXT_POLICY,
        allow_exploration=policy.allow_exploration,
        requires_user_confirmation=policy.requires_user_confirmation,
        requires_manual_review=bool(domain and domain.requires_manual_review),
        blocks_all_beliefs=False,
        risk_policy_version=RECOMMENDATION_CONTEXT_POLICY_VERSION,
        risk_domain_policy_version=RISK_DOMAIN_POLICY_VERSION,
        _context_policy=policy,
    )


def _resolve_from_domain_policy(
    domain: RiskDomainPolicy, normalized: str, proposed: str
) -> ResolvedContextPolicy:
    fallback = domain_fallback_context_policy(domain)
    return ResolvedContextPolicy(
        proposed_context_key=proposed,
        context_key=normalized,
        risk_tier=domain.default_unknown_context_risk_tier,
        resolution_path=ContextResolutionPath.RISK_DOMAIN_POLICY,
        allow_exploration=False,
        requires_user_confirmation=True,
        requires_manual_review=domain.requires_manual_review,
        blocks_all_beliefs=False,
        risk_policy_version=RECOMMENDATION_CONTEXT_POLICY_VERSION,
        risk_domain_policy_version=RISK_DOMAIN_POLICY_VERSION,
        _context_policy=fallback,
    )


def _resolve_global_fallback(normalized: str, proposed: str) -> ResolvedContextPolicy:
    return ResolvedContextPolicy(
        proposed_context_key=proposed,
        context_key=normalized or proposed,
        risk_tier=RecommendationRiskTier.HIGH,
        resolution_path=ContextResolutionPath.GLOBAL_CONSERVATIVE_FALLBACK,
        allow_exploration=False,
        requires_user_confirmation=True,
        requires_manual_review=True,
        blocks_all_beliefs=True,
        risk_policy_version=RECOMMENDATION_CONTEXT_POLICY_VERSION,
        risk_domain_policy_version=RISK_DOMAIN_POLICY_VERSION,
        _context_policy=None,
    )


def resolve_context_policy(
    context: str | RecommendationContextProposal,
    *,
    context_registry: dict[str, RecommendationContextPolicy] | None = None,
    domain_registry: dict[str, RiskDomainPolicy] | None = None,
) -> ResolvedContextPolicy:
    """Resolve a proposed recommendation context to a backend-owned
    ``ResolvedContextPolicy``.

    Resolution order (blueprint section 6.4, deterministic, backend-owned):

    1. **Exact context policy** -- ``normalize(proposed)`` is a key in
       ``RECOMMENDATION_CONTEXT_POLICY``: use it verbatim, risk tier and all.
    2. **Parent / domain risk_domain_policy** -- the domain segment of the
       normalized key is a known ``RISK_DOMAIN_POLICY`` domain: risk tier is
       that domain's ``default_unknown_context_risk_tier`` (only ever MEDIUM
       or HIGH), belief eligibility is the conservative domain fallback.
    3. **Global conservative fallback** -- otherwise: HIGH risk,
       ``blocks_all_beliefs=True``, manual review required. An unknown
       context with no known domain never becomes a usable recommendation
       input source.

    The LLM's proposed string only ever selects *which* row is looked up; it
    never supplies a tier and can never move an unknown context to a lower
    tier than this function would otherwise assign.
    """
    proposed = (
        context.proposed_context_key
        if isinstance(context, RecommendationContextProposal)
        else context
    )
    contexts = RECOMMENDATION_CONTEXT_POLICY if context_registry is None else context_registry
    domains = RISK_DOMAIN_POLICY if domain_registry is None else domain_registry

    normalized = normalize_context_key(proposed)

    exact = contexts.get(normalized)
    if exact is not None:
        return _resolve_from_exact_policy(exact, proposed)

    domain = domains.get(_domain_of(normalized)) if normalized else None
    if domain is not None:
        return _resolve_from_domain_policy(domain, normalized, proposed)

    return _resolve_global_fallback(normalized, proposed)


def _decide_one_belief(belief: UserBelief, resolved: ResolvedContextPolicy) -> BeliefContextDecision:
    def decision(allowed: bool, reason: ContextEligibilityReason, detail: str) -> BeliefContextDecision:
        return BeliefContextDecision(
            belief_id=belief.belief_id,
            belief_key=belief.belief_key,
            allowed=allowed,
            reason=reason,
            detail=detail,
            risk_tier=resolved.risk_tier,
            context_key=resolved.context_key,
            risk_policy_version=resolved.risk_policy_version,
            risk_domain_policy_version=resolved.risk_domain_policy_version,
        )

    if resolved.blocks_all_beliefs:
        return decision(
            False,
            ContextEligibilityReason.BLOCKED_INCOMPLETE_POLICY_RESOLUTION,
            "context resolved only to the global conservative fallback; no belief may be used",
        )

    if belief.locked_until_recompute:
        return decision(
            False, ContextEligibilityReason.BLOCKED_LOCKED,
            "belief is locked_until_recompute; its cached confidence/status are non-authoritative",
        )

    if belief.status in GLOBALLY_EXCLUDED_STATUSES:
        return decision(
            False, ContextEligibilityReason.BLOCKED_STATUS,
            f"belief status {belief.status.value!r} is never a usable recommendation input",
        )

    context_policy = resolved._context_policy
    assert context_policy is not None  # only None when blocks_all_beliefs is True, handled above

    if belief.status not in context_policy.permitted_belief_statuses:
        return decision(
            False, ContextEligibilityReason.BLOCKED_STATUS_NOT_PERMITTED_BY_CONTEXT,
            f"belief status {belief.status.value!r} is not permitted by the {resolved.risk_tier.value}"
            f"-risk context {resolved.context_key!r} (permitted: "
            f"{sorted(s.value for s in context_policy.permitted_belief_statuses)})",
        )

    if belief.belief_type in context_policy.disallowed_belief_types:
        return decision(
            False, ContextEligibilityReason.BLOCKED_BELIEF_TYPE,
            f"belief_type {belief.belief_type.value!r} is on the disallowed list for context "
            f"{resolved.context_key!r}",
        )
    if (
        context_policy.allowed_belief_types
        and belief.belief_type not in context_policy.allowed_belief_types
    ):
        return decision(
            False, ContextEligibilityReason.BLOCKED_BELIEF_TYPE,
            f"belief_type {belief.belief_type.value!r} is not on the allow-list for context "
            f"{resolved.context_key!r}",
        )

    if belief.sensitivity_class.value in _BLOCKED_SENSITIVITY:
        return decision(
            False, ContextEligibilityReason.BLOCKED_SENSITIVITY,
            f"sensitivity_class {belief.sensitivity_class.value!r} is blocked from recommendation use",
        )

    if resolved.context_key in belief.disallowed_contexts:
        return decision(
            False, ContextEligibilityReason.BLOCKED_DISALLOWED_CONTEXT,
            f"context {resolved.context_key!r} is in this belief's disallowed_contexts",
        )
    if belief.allowed_contexts and resolved.context_key not in belief.allowed_contexts:
        return decision(
            False, ContextEligibilityReason.BLOCKED_NOT_IN_ALLOWED_CONTEXTS,
            f"this belief restricts use to {sorted(belief.allowed_contexts)}, which excludes "
            f"{resolved.context_key!r}",
        )

    return decision(True, ContextEligibilityReason.ALLOWED, "passed every context/risk/purpose gate")


def authorize_beliefs_for_context(
    beliefs: list[UserBelief],
    context_key: str | RecommendationContextProposal,
    *,
    context_registry: dict[str, RecommendationContextPolicy] | None = None,
    domain_registry: dict[str, RiskDomainPolicy] | None = None,
) -> ContextAuthorizationResult:
    """Pure function. Decide which ``beliefs`` may be used as inputs to a
    recommendation in ``context_key``.

    ``context_key`` is whatever the caller (possibly an LLM) proposed -- a
    bare string or a ``RecommendationContextProposal``. It is resolved to a
    backend-owned risk tier and policy via ``resolve_context_policy()``; the
    caller cannot pass a risk tier at all.

    Each belief is checked, in order, against: incomplete policy resolution
    -> ``locked_until_recompute`` -> globally-excluded status (outdated /
    rejected / contested) -> status not permitted by this context -> belief
    type disallowed / not allowed -> sensitivity class -> the belief's own
    ``disallowed_contexts`` -> the belief's own ``allowed_contexts``
    allow-list. The first failing gate is the recorded reason.

    Returns a ``ContextAuthorizationResult`` with one ``BeliefContextDecision``
    per input belief (order preserved) and both policy versions, so the
    decision is fully replayable.
    """
    resolved = resolve_context_policy(
        context_key, context_registry=context_registry, domain_registry=domain_registry
    )
    decisions = [_decide_one_belief(belief, resolved) for belief in beliefs]
    return ContextAuthorizationResult(
        proposed_context_key=resolved.proposed_context_key,
        context_key=resolved.context_key,
        risk_tier=resolved.risk_tier,
        resolution_path=resolved.resolution_path,
        allow_exploration=resolved.allow_exploration,
        requires_user_confirmation=resolved.requires_user_confirmation,
        requires_manual_review=resolved.requires_manual_review,
        risk_policy_version=resolved.risk_policy_version,
        risk_domain_policy_version=resolved.risk_domain_policy_version,
        decisions=decisions,
    )

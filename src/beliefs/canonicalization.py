"""Belief-key canonicalization (blueprint section 5.2): prevents semantically
duplicate belief keys (e.g. ``"prefers_evening_exercise_sessions"`` vs.
``"higher_adherence_after_work"``) from fragmenting a user's evidence and
confidence across multiple ``UserBelief`` rows that should really be one.

The model-proposes/backend-authorizes boundary (section 5.2's own
"Authorization rule", the same structural pattern already enforced for
belief_evidence in ``src/beliefs/models.py``):

- ``BeliefKeyCanonicalizationProposal`` is the complete set of fields a
  skill, extractor, or LLM output may set: a candidate ``proposed_key`` and,
  optionally, its own guess at ``proposed_canonical_key``/
  ``proposed_decision``/``reason``. It has no ``canonical_key``, ``decision``,
  or any other backend-owned field at all.
- ``BeliefKeyCanonicalization`` extends it with exactly the backend-owned
  fields (``canonicalization_id``, ``canonical_key``, ``decision``,
  ``decision_reason``, ``authorized_by``, ``authorized_at``,
  ``canonical_key_registry_version``). It can only be constructed by hand
  (e.g. when deserializing an existing persisted record) or via
  ``authorize_belief_key_canonicalization()`` below.
- ``authorize_belief_key_canonicalization()`` is the single sanctioned path
  from a proposal to a persisted record. A proposal's own
  ``proposed_decision``/``proposed_canonical_key`` are never trusted at face
  value: the canonical belief_key registry (``src/common/registry.py``,
  itself the actual backend policy data, not a runtime guess) is checked
  first and always wins when it has an answer; only when the registry has no
  opinion does a proposal's own suggestion get any influence at all, and
  never enough to grant ``merge`` or an unverified ``alias`` -- both
  downgrade to ``manual_review`` (section 5.2: "risky, cross-context,
  sensitive, or semantically uncertain merges must remain separate or enter
  a human-review queue").

Deliberately narrow, matching this phase's own scope: no semantic-similarity
or embedding-based matching (the blueprint explicitly allows, but does not
require, "lightweight embeddings" for this -- a small, exact-match,
hand-curated registry is the conservative default this phase implements
instead); no execution of an actual belief/evidence merge (old rows are
never rewritten or deleted -- a ``merge`` decision is never even granted
automatically, only ever downgraded); no recommendation ranking (a separate,
much later phase).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.common.enums import BeliefType, CanonicalizationDecision
from src.common.registry import CANONICAL_BELIEF_KEY_ALIASES, CANONICAL_BELIEF_KEY_REGISTRY_VERSION
from src.common.versioning import VersionedModel


class BeliefKeyCanonicalizationProposal(VersionedModel):
    """Fields a skill/extractor/LLM may propose when a new or ambiguous
    belief_key needs to be resolved. ``proposed_decision`` defaults to the
    safest possible suggestion (``keep_separate``); a proposal requesting
    ``merge`` or ``alias`` is only ever a *request* -- see this module's own
    docstring for why authorization never grants either on a proposal's say-so
    alone.
    """

    user_id: str = Field(min_length=1)
    belief_type: BeliefType
    proposed_key: str = Field(min_length=1)
    proposed_canonical_key: str | None = None
    proposed_decision: CanonicalizationDecision = CanonicalizationDecision.KEEP_SEPARATE
    reason: str | None = None


class BeliefKeyCanonicalization(BeliefKeyCanonicalizationProposal):
    """The full persisted canonicalization record: the proposal fields plus
    every backend-owned field. Do not construct this directly from untrusted
    input -- use ``authorize_belief_key_canonicalization()``."""

    canonicalization_id: str = Field(min_length=1)
    canonical_key: str = Field(min_length=1)
    decision: CanonicalizationDecision
    decision_reason: str = Field(min_length=1)
    authorized_by: str = Field(min_length=1)
    authorized_at: datetime
    canonical_key_registry_version: str = Field(min_length=1)


def authorize_belief_key_canonicalization(
    proposal: BeliefKeyCanonicalizationProposal,
    *,
    canonicalization_id: str,
    authorized_at: datetime,
    authorized_by: str = "canonicalization_policy",
    registry: dict[tuple[BeliefType, str], str] | None = None,
    registry_version: str = CANONICAL_BELIEF_KEY_REGISTRY_VERSION,
) -> BeliefKeyCanonicalization:
    """The only sanctioned constructor from a proposal to a persisted
    ``BeliefKeyCanonicalization`` record (blueprint section 5.2's
    authorization rule).

    Decision order, each one checked only when the previous found nothing:

    1. **Registry hit** (``(belief_type, proposed_key)`` is a known alias in
       ``registry``, default ``CANONICAL_BELIEF_KEY_ALIASES``): authorized as
       ``alias`` to the registry's canonical key, regardless of whatever the
       proposal itself suggested -- this is the one case backend policy has
       enough certainty to act on unattended, because the registry entry
       itself *is* the pre-vetted backend decision, not a runtime guess.
    2. **Proposal requests ``merge``**: always downgraded to
       ``manual_review``. Automatic merge requires a deterministic guard this
       phase does not implement (no similarity scoring exists to clear the
       blueprint's "narrow, same-context, same-polarity, high-similarity"
       bar), and old belief/evidence rows are never rewritten or deleted, so
       there is nothing here that could execute a merge safely.
    3. **Proposal requests ``alias`` to some other key it names, unconfirmed
       by the registry**: also downgraded to ``manual_review`` -- exactly the
       "semantically uncertain" case section 5.2 says must not be silently
       merged. A proposal cannot self-authorize an alias any more than
       ``BeliefEvidenceProposal`` can self-authorize ``aggregate_replacement``.
    4. **Otherwise** (no registry knowledge, no risky request): ``keep_separate``,
       using ``proposed_key`` itself as the canonical key. An unknown key is
       never guessed into an existing one.

    ``canonical_key`` is always set (never null) -- even a ``manual_review``
    or ``keep_separate`` decision needs *some* key a caller can act on today
    (typically ``proposed_key`` itself) while treating it as provisional
    until a human resolves the ambiguity.
    """
    active_registry = CANONICAL_BELIEF_KEY_ALIASES if registry is None else registry
    registry_hit = active_registry.get((proposal.belief_type, proposal.proposed_key))

    if registry_hit is not None and registry_hit != proposal.proposed_key:
        canonical_key = registry_hit
        decision = CanonicalizationDecision.ALIAS
        decision_reason = (
            f"proposed_key {proposal.proposed_key!r} matches a known alias in the canonical "
            f"belief_key registry ({registry_version}) for belief_type "
            f"{proposal.belief_type.value!r}; authorized as an alias to {canonical_key!r}."
        )
    elif proposal.proposed_decision == CanonicalizationDecision.MERGE:
        canonical_key = proposal.proposed_key
        decision = CanonicalizationDecision.MANUAL_REVIEW
        decision_reason = (
            "proposed_decision='merge' was requested but automatic merge requires a deterministic "
            "registry match, which was not found; downgraded to manual_review rather than executed "
            "or silently discarded (blueprint section 5.2: risky or uncertain merges require review)."
        )
    elif (
        proposal.proposed_decision == CanonicalizationDecision.ALIAS
        and proposal.proposed_canonical_key is not None
        and proposal.proposed_canonical_key != proposal.proposed_key
    ):
        canonical_key = proposal.proposed_key
        decision = CanonicalizationDecision.MANUAL_REVIEW
        decision_reason = (
            f"proposed_decision='alias' to {proposal.proposed_canonical_key!r} was requested but is "
            f"not backed by the canonical belief_key registry ({registry_version}); downgraded to "
            "manual_review rather than trusting an unverified proposal."
        )
    else:
        canonical_key = proposal.proposed_key
        decision = CanonicalizationDecision.KEEP_SEPARATE
        decision_reason = (
            f"no known alias for proposed_key {proposal.proposed_key!r} in the canonical belief_key "
            f"registry ({registry_version}) for belief_type {proposal.belief_type.value!r}; kept as "
            "its own distinct key rather than guessed at (blueprint section 5.2: unknown or "
            "uncertain keys default to keep-separate)."
        )

    return BeliefKeyCanonicalization(
        **proposal.model_dump(),
        canonicalization_id=canonicalization_id,
        canonical_key=canonical_key,
        decision=decision,
        decision_reason=decision_reason,
        authorized_by=authorized_by,
        authorized_at=authorized_at,
        canonical_key_registry_version=registry_version,
    )

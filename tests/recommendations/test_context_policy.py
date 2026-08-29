"""Tests for src/recommendations/context_policy.py -- the recommendation
context / risk policy foundation (blueprint section 6.4-6.5).

Covers exactly the guarantees this phase must prove:
1. a low-risk context allows normal behavioral beliefs;
2. a high-risk context blocks sensitive / high-impact inference;
3. an unknown context follows the deterministic conservative fallback;
4. locked (and outdated / rejected) beliefs are excluded;
5. an LLM-supplied risk_tier is rejected at construction, never honoured;
6. both policy versions are recorded on the result and on every decision.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.beliefs.models import UserBelief
from src.common.enums import ContextEligibilityReason, ContextResolutionPath, RecommendationRiskTier
from src.common.registry import (
    RECOMMENDATION_CONTEXT_POLICY_VERSION,
    RISK_DOMAIN_POLICY_VERSION,
)
from src.recommendations.context_policy import (
    RecommendationContextProposal,
    authorize_beliefs_for_context,
    normalize_context_key,
    resolve_context_policy,
)

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _belief(**overrides) -> UserBelief:
    base = dict(
        belief_id="bel_1", user_id="usr_17", belief_type="behavioral_tendency",
        belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
        belief_value=True, confidence=0.68, supporting_evidence_count=4,
        contradicting_evidence_count=1, total_evidence_count=5,
        effective_support_count=3.62, effective_evidence_count=4.31,
        evidence_for=4, evidence_against=1, allowed_contexts=[], disallowed_contexts=[],
        sensitivity_class="normal", persistence_policy="retained",
        first_observed=datetime(2026, 8, 10, 18, tzinfo=timezone.utc),
        last_validated=datetime(2026, 8, 22, 17, tzinfo=timezone.utc),
        status="validated",
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return UserBelief(**base)


class LowRiskContextTests(unittest.TestCase):
    def test_low_risk_context_allows_a_normal_behavioral_belief(self) -> None:
        result = authorize_beliefs_for_context([_belief()], "fitness_scheduling")

        self.assertEqual(result.risk_tier, RecommendationRiskTier.LOW)
        self.assertEqual(result.resolution_path, ContextResolutionPath.EXACT_CONTEXT_POLICY)
        self.assertEqual(result.authorized_beliefs, ["bel_1"])
        self.assertEqual(result.decisions[0].reason, ContextEligibilityReason.ALLOWED)
        self.assertTrue(result.allow_exploration)

    def test_low_risk_context_still_admits_a_candidate_status_belief(self) -> None:
        result = authorize_beliefs_for_context([_belief(status="candidate", confidence=0.1)], "habit_nudge")
        self.assertEqual(result.authorized_beliefs, ["bel_1"])


class HighRiskContextTests(unittest.TestCase):
    def test_high_risk_context_blocks_sensitive_high_impact_inference(self) -> None:
        sensitive = _belief(
            belief_id="bel_sensitive", belief_type="sensitive_or_high_impact_inference",
            belief_key="infers_medical_condition", sensitivity_class="restricted",
        )
        result = authorize_beliefs_for_context([sensitive], "mental_health_support")

        self.assertEqual(result.risk_tier, RecommendationRiskTier.HIGH)
        self.assertEqual(result.authorized_beliefs, [])
        # belief_type gate is hit before the sensitivity gate; both would block.
        self.assertEqual(result.decisions[0].reason, ContextEligibilityReason.BLOCKED_BELIEF_TYPE)
        self.assertFalse(result.decisions[0].allowed)

    def test_high_risk_context_blocks_a_merely_provisional_belief(self) -> None:
        allowed_type_provisional = _belief(
            belief_type="communication_or_learning_preference", status="provisional",
            belief_key="prefers_short_messages",
        )
        result = authorize_beliefs_for_context([allowed_type_provisional], "mental_health_support")
        self.assertEqual(
            result.decisions[0].reason,
            ContextEligibilityReason.BLOCKED_STATUS_NOT_PERMITTED_BY_CONTEXT,
        )

    def test_high_risk_context_allows_a_validated_permitted_type(self) -> None:
        ok = _belief(
            belief_type="communication_or_learning_preference", status="validated",
            belief_key="prefers_short_messages",
        )
        result = authorize_beliefs_for_context([ok], "mental_health_support")
        self.assertEqual(result.authorized_beliefs, ["bel_1"])


class UnknownContextFallbackTests(unittest.TestCase):
    def test_unknown_context_with_known_domain_escalates_deterministically(self) -> None:
        first = authorize_beliefs_for_context([_belief()], "fitness_brand_new_surface")
        second = authorize_beliefs_for_context([_belief()], "fitness_brand_new_surface")

        self.assertEqual(first.resolution_path, ContextResolutionPath.RISK_DOMAIN_POLICY)
        self.assertEqual(first.risk_tier, RecommendationRiskTier.MEDIUM)  # fitness domain default
        self.assertEqual(first.risk_tier, second.risk_tier)
        self.assertEqual(first.decisions[0].reason, second.decisions[0].reason)

    def test_unknown_context_with_unknown_domain_blocks_everything_at_high_risk(self) -> None:
        result = authorize_beliefs_for_context([_belief()], "some_unmapped_surface")

        self.assertEqual(result.resolution_path, ContextResolutionPath.GLOBAL_CONSERVATIVE_FALLBACK)
        self.assertEqual(result.risk_tier, RecommendationRiskTier.HIGH)
        self.assertTrue(result.requires_manual_review)
        self.assertFalse(result.allow_exploration)
        self.assertEqual(result.authorized_beliefs, [])
        self.assertEqual(
            result.decisions[0].reason,
            ContextEligibilityReason.BLOCKED_INCOMPLETE_POLICY_RESOLUTION,
        )

    def test_unknown_context_never_resolves_below_medium(self) -> None:
        for surface in ("fitness_x", "health_x", "finance_x", "mystery_x", "  "):
            tier = resolve_context_policy(surface).risk_tier
            self.assertIn(tier, (RecommendationRiskTier.MEDIUM, RecommendationRiskTier.HIGH))

    def test_normalization_is_deterministic_and_idempotent(self) -> None:
        for raw in ("Fitness Scheduling", "fitness-scheduling", " fitness/scheduling "):
            self.assertEqual(normalize_context_key(raw), "fitness_scheduling")
        once = normalize_context_key("Health // Random Thing")
        self.assertEqual(normalize_context_key(once), once)


class LockedAndStaleBeliefsExcludedTests(unittest.TestCase):
    def test_locked_belief_is_excluded_even_in_a_low_risk_context(self) -> None:
        locked = _belief(locked_until_recompute=True)
        result = authorize_beliefs_for_context([locked], "fitness_scheduling")
        self.assertEqual(result.authorized_beliefs, [])
        self.assertEqual(result.decisions[0].reason, ContextEligibilityReason.BLOCKED_LOCKED)

    def test_outdated_and_rejected_and_contested_are_excluded(self) -> None:
        for status in ("outdated", "rejected", "contested"):
            belief = _belief(status=status, confidence=0.0 if status == "outdated" else 0.5)
            result = authorize_beliefs_for_context([belief], "fitness_scheduling")
            self.assertEqual(result.authorized_beliefs, [], status)
            self.assertEqual(result.decisions[0].reason, ContextEligibilityReason.BLOCKED_STATUS, status)

    def test_a_mixed_batch_keeps_only_the_usable_beliefs(self) -> None:
        beliefs = [
            _belief(belief_id="ok"),
            _belief(belief_id="locked", locked_until_recompute=True),
            _belief(belief_id="outdated", status="outdated", confidence=0.0),
        ]
        result = authorize_beliefs_for_context(beliefs, "fitness_scheduling")
        self.assertEqual(result.authorized_beliefs, ["ok"])
        self.assertEqual(len(result.decisions), 3)


class PurposeLimitationTests(unittest.TestCase):
    def test_belief_disallowed_contexts_block_use(self) -> None:
        belief = _belief(disallowed_contexts=["fitness_scheduling"])
        result = authorize_beliefs_for_context([belief], "fitness_scheduling")
        self.assertEqual(
            result.decisions[0].reason, ContextEligibilityReason.BLOCKED_DISALLOWED_CONTEXT
        )

    def test_belief_allowed_contexts_allow_list_blocks_other_contexts(self) -> None:
        belief = _belief(allowed_contexts=["some_other_context"])
        result = authorize_beliefs_for_context([belief], "fitness_scheduling")
        self.assertEqual(
            result.decisions[0].reason, ContextEligibilityReason.BLOCKED_NOT_IN_ALLOWED_CONTEXTS
        )

    def test_belief_allowed_contexts_allow_list_permits_the_named_context(self) -> None:
        belief = _belief(allowed_contexts=["fitness_scheduling"])
        result = authorize_beliefs_for_context([belief], "fitness_scheduling")
        self.assertEqual(result.authorized_beliefs, ["bel_1"])

    def test_context_policy_disallowed_belief_type_blocks_use(self) -> None:
        # current_state_related is on mental_health_support's disallowed list.
        belief = _belief(
            belief_type="current_state_related", status="validated", belief_key="currently_stressed",
            persistence_policy="short_term",
        )
        result = authorize_beliefs_for_context([belief], "mental_health_support")
        self.assertEqual(result.decisions[0].reason, ContextEligibilityReason.BLOCKED_BELIEF_TYPE)


class LlmCannotSetRiskTierTests(unittest.TestCase):
    def test_proposal_model_rejects_a_risk_tier_field(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendationContextProposal(
                proposed_context_key="mental_health_support", risk_tier="low", **VERSION_FIELDS
            )

    def test_proposal_model_rejects_other_policy_owned_fields(self) -> None:
        for smuggled in ("default_risk_tier", "resolution_path", "risk_policy_version"):
            with self.assertRaises(ValidationError):
                RecommendationContextProposal(
                    proposed_context_key="x", **{smuggled: "low"}, **VERSION_FIELDS
                )

    def test_a_proposal_only_selects_which_policy_row_is_used(self) -> None:
        proposal = RecommendationContextProposal(
            proposed_context_key="Mental Health Support", **VERSION_FIELDS
        )
        resolved = resolve_context_policy(proposal)
        # The backend still assigns HIGH from the registry; the proposal only
        # chose the label.
        self.assertEqual(resolved.risk_tier, RecommendationRiskTier.HIGH)
        self.assertEqual(resolved.context_key, "mental_health_support")
        self.assertEqual(resolved.proposed_context_key, "Mental Health Support")


class PolicyVersionRecordedTests(unittest.TestCase):
    def test_result_and_every_decision_record_both_policy_versions(self) -> None:
        result = authorize_beliefs_for_context(
            [_belief(belief_id="a"), _belief(belief_id="b", locked_until_recompute=True)],
            "fitness_scheduling",
        )
        self.assertEqual(result.risk_policy_version, RECOMMENDATION_CONTEXT_POLICY_VERSION)
        self.assertEqual(result.risk_domain_policy_version, RISK_DOMAIN_POLICY_VERSION)
        for decision in result.decisions:
            self.assertEqual(decision.risk_policy_version, RECOMMENDATION_CONTEXT_POLICY_VERSION)
            self.assertEqual(decision.risk_domain_policy_version, RISK_DOMAIN_POLICY_VERSION)

    def test_versions_are_recorded_even_on_the_global_fallback_path(self) -> None:
        result = authorize_beliefs_for_context([_belief()], "totally_unmapped")
        self.assertEqual(result.risk_policy_version, RECOMMENDATION_CONTEXT_POLICY_VERSION)
        self.assertEqual(result.risk_domain_policy_version, RISK_DOMAIN_POLICY_VERSION)
        self.assertEqual(result.decisions[0].risk_policy_version, RECOMMENDATION_CONTEXT_POLICY_VERSION)


if __name__ == "__main__":
    unittest.main()

"""Tests for src/recommendations/engine.py -- the deterministic recommendation MVP.

Proves the two things the task requires:
- high-risk / disallowed / locked / stale beliefs are excluded and cannot
  drive a recommendation;
- allowed beliefs do drive a recommendation, deterministically, with the
  full policy/scoring/version trace persisted on the record.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.beliefs.models import UserBelief
from src.common.enums import (
    ContextEligibilityReason,
    RecommendationReviewStatus,
    RecommendationRiskTier,
    ResolutionMode,
    RiskResolutionPath,
)
from src.common.registry import (
    RECOMMENDATION_CONTEXT_POLICY_VERSION,
    RECOMMENDATION_RANKING_VERSION,
    RISK_DOMAIN_POLICY_VERSION,
)
from src.recommendations.engine import generate_recommendation

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _belief(
    belief_id: str,
    *,
    belief_key: str = "higher_adherence_after_work",
    belief_type: str = "behavioral_tendency",
    status: str = "validated",
    confidence: float = 0.7,
    belief_value: object = True,
    effective_evidence_count: float = 2.0,
    locked: bool = False,
    sensitivity_class: str = "normal",
    allowed_contexts: list[str] | None = None,
    disallowed_contexts: list[str] | None = None,
    persistence_policy: str = "retained",
) -> UserBelief:
    return UserBelief(
        belief_id=belief_id, user_id="usr_1", belief_type=belief_type,
        belief_type_registry_version="belief-types-0.6", belief_key=belief_key,
        belief_value=belief_value, confidence=confidence,
        supporting_evidence_count=3, contradicting_evidence_count=0, total_evidence_count=3,
        effective_support_count=effective_evidence_count,
        effective_evidence_count=effective_evidence_count, evidence_for=3, evidence_against=0,
        allowed_contexts=allowed_contexts or [], disallowed_contexts=disallowed_contexts or [],
        sensitivity_class=sensitivity_class, persistence_policy=persistence_policy,
        first_observed=AS_OF - timedelta(days=5), last_validated=AS_OF, status=status,
        locked_until_recompute=locked, **VERSION_FIELDS,
    )


def _generate(beliefs, context_key="fitness_scheduling", recommendation_id="rec_1"):
    return generate_recommendation(
        recommendation_id=recommendation_id, user_id="usr_1", context_key=context_key,
        beliefs=beliefs, created_at=AS_OF, goal="be more consistent",
    )


class AllowedBeliefDrivesARecommendationTests(unittest.TestCase):
    def test_a_validated_behavioral_belief_produces_an_issued_recommendation(self) -> None:
        rec = _generate([_belief("b1")])

        self.assertEqual(rec.risk_tier, RecommendationRiskTier.LOW)
        self.assertFalse(rec.review_required)
        self.assertEqual(rec.review_status, RecommendationReviewStatus.NOT_REQUIRED)
        self.assertEqual(rec.belief_ids_used, ["b1"])
        self.assertGreater(rec.ranking_score, 0.0)
        self.assertNotIn("no recommendation issued", rec.recommendation)
        self.assertIn("b1", rec.frozen_belief_state)
        self.assertTrue(rec.frozen_belief_state["b1"]["authorized"])

    def test_ranking_is_deterministic_and_picks_the_higher_scored_belief(self) -> None:
        strong = _belief("strong", confidence=0.9, status="validated")
        weak = _belief(
            "weak", belief_key="responds_to_streak_framing", confidence=0.3, status="candidate"
        )
        first = _generate([strong, weak], recommendation_id="rec_a")
        second = _generate([weak, strong], recommendation_id="rec_b")

        self.assertEqual(first.belief_ids_used, ["strong"])
        self.assertEqual(first.recommendation, second.recommendation)
        self.assertEqual(first.ranking_score, second.ranking_score)
        self.assertEqual(
            [c.action for c in first.candidate_trace], [c.action for c in second.candidate_trace]
        )
        selected = [c for c in first.candidate_trace if c.selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].belief_ids, ["strong"])

    def test_beliefs_sharing_a_template_action_form_one_candidate(self) -> None:
        rec = _generate(
            [_belief("b1", effective_evidence_count=2.0), _belief("b2", effective_evidence_count=1.0)]
        )
        self.assertEqual(sorted(rec.belief_ids_used), ["b1", "b2"])
        self.assertEqual(len(rec.candidate_trace), 1)

    def test_a_false_valued_belief_does_not_drive_an_action(self) -> None:
        rec = _generate([_belief("b1", belief_value=False)])
        self.assertEqual(rec.belief_ids_used, [])
        self.assertIn("no recommendation issued", rec.recommendation)
        # It is still authorized (present, not blocked) -- just not actionable.
        self.assertTrue(rec.frozen_belief_state["b1"]["authorized"])
        self.assertEqual(rec.blocked_beliefs, [])


class HighRiskAndDisallowedBeliefsAreExcludedTests(unittest.TestCase):
    def test_sensitive_high_impact_belief_is_blocked_in_a_high_risk_context(self) -> None:
        sensitive = _belief(
            "b_sensitive", belief_key="infers_condition",
            belief_type="sensitive_or_high_impact_inference", sensitivity_class="restricted",
        )
        rec = _generate([sensitive], context_key="mental_health_support")

        self.assertEqual(rec.risk_tier, RecommendationRiskTier.HIGH)
        self.assertEqual(rec.belief_ids_used, [])
        blocked_ids = {b.belief_id for b in rec.blocked_beliefs}
        self.assertIn("b_sensitive", blocked_ids)
        self.assertEqual(
            rec.blocked_beliefs[0].reason, ContextEligibilityReason.BLOCKED_BELIEF_TYPE
        )
        self.assertFalse(rec.frozen_belief_state["b_sensitive"]["authorized"])

    def test_a_high_risk_context_holds_even_an_eligible_belief_for_review(self) -> None:
        eligible = _belief(
            "b1", belief_key="prefers_short_messages",
            belief_type="communication_or_learning_preference", status="validated",
        )
        rec = _generate([eligible], context_key="mental_health_support")

        self.assertTrue(rec.review_required)
        self.assertEqual(rec.review_status, RecommendationReviewStatus.PENDING)
        self.assertEqual(rec.required_resolution_mode, ResolutionMode.REVIEWER)
        self.assertIsNone(rec.actual_resolution_mode)
        self.assertFalse(rec.exploration_applied)
        # The candidate is recorded as a proposal, not auto-issued.
        self.assertEqual(rec.belief_ids_used, ["b1"])
        self.assertIn("manual review", rec.rationale)

    def test_disallowed_belief_type_for_a_context_is_excluded(self) -> None:
        # current_state_related is on mental_health_support's disallowed list.
        belief = _belief(
            "b_state", belief_key="currently_stressed", belief_type="current_state_related",
            status="validated", persistence_policy="short_term",
        )
        rec = _generate([belief], context_key="mental_health_support")
        self.assertEqual(rec.belief_ids_used, [])
        self.assertEqual(
            rec.blocked_beliefs[0].reason, ContextEligibilityReason.BLOCKED_BELIEF_TYPE
        )

    def test_locked_belief_is_excluded_from_a_low_risk_recommendation(self) -> None:
        rec = _generate([_belief("b_locked", locked=True)])
        self.assertEqual(rec.belief_ids_used, [])
        self.assertEqual(
            rec.blocked_beliefs[0].reason, ContextEligibilityReason.BLOCKED_LOCKED
        )

    def test_outdated_belief_is_excluded(self) -> None:
        rec = _generate([_belief("b_old", status="outdated", confidence=0.0)])
        self.assertEqual(rec.belief_ids_used, [])
        self.assertEqual(rec.blocked_beliefs[0].reason, ContextEligibilityReason.BLOCKED_STATUS)

    def test_belief_disallowed_context_is_respected(self) -> None:
        rec = _generate([_belief("b1", disallowed_contexts=["fitness_scheduling"])])
        self.assertEqual(rec.belief_ids_used, [])
        self.assertEqual(
            rec.blocked_beliefs[0].reason, ContextEligibilityReason.BLOCKED_DISALLOWED_CONTEXT
        )


class UnknownContextTests(unittest.TestCase):
    def test_unknown_context_blocks_all_beliefs_and_still_persists_a_record(self) -> None:
        rec = _generate([_belief("b1")], context_key="totally_unmapped_surface")

        self.assertEqual(rec.risk_tier, RecommendationRiskTier.HIGH)
        self.assertEqual(rec.risk_resolution_path, RiskResolutionPath.GLOBAL_FALLBACK)
        self.assertTrue(rec.review_required)
        self.assertEqual(rec.belief_ids_used, [])
        self.assertIn("no recommendation issued", rec.recommendation)
        self.assertEqual(
            rec.blocked_beliefs[0].reason,
            ContextEligibilityReason.BLOCKED_INCOMPLETE_POLICY_RESOLUTION,
        )

    def test_unknown_context_with_known_domain_uses_domain_policy_path(self) -> None:
        rec = _generate([_belief("b1")], context_key="fitness_some_new_surface")
        self.assertEqual(rec.risk_resolution_path, RiskResolutionPath.DOMAIN_POLICY)
        self.assertEqual(rec.risk_tier, RecommendationRiskTier.MEDIUM)


class PolicyAndVersionTraceTests(unittest.TestCase):
    def test_record_carries_every_policy_and_scoring_version(self) -> None:
        rec = _generate([_belief("b1")])

        self.assertEqual(rec.risk_policy_version, RECOMMENDATION_CONTEXT_POLICY_VERSION)
        self.assertEqual(rec.risk_domain_policy_version, RISK_DOMAIN_POLICY_VERSION)
        self.assertEqual(rec.ranking_policy_version, RECOMMENDATION_RANKING_VERSION)
        self.assertEqual(rec.scoring_version, "belief-score-0.6")
        self.assertEqual(rec.policy_version, "policy-0.6")
        self.assertEqual(rec.recommendation_context, "fitness_scheduling")
        self.assertEqual(rec.risk_resolution_path, RiskResolutionPath.EXACT_CONTEXT)
        self.assertIsNone(rec.prompt_version)  # no LLM

    def test_frozen_belief_state_captures_all_candidate_beliefs(self) -> None:
        beliefs = [
            _belief("used"),
            _belief("blocked_locked", locked=True),
            _belief("blocked_status", status="rejected", confidence=0.0),
        ]
        rec = _generate(beliefs)
        self.assertEqual(set(rec.frozen_belief_state), {"used", "blocked_locked", "blocked_status"})
        self.assertTrue(rec.frozen_belief_state["used"]["authorized"])
        self.assertFalse(rec.frozen_belief_state["blocked_locked"]["authorized"])
        self.assertEqual(
            rec.frozen_belief_state["blocked_status"]["block_reason"], "blocked_status"
        )

    def test_no_beliefs_at_all_produces_an_auditable_empty_record(self) -> None:
        rec = _generate([])
        self.assertEqual(rec.belief_ids_used, [])
        self.assertEqual(rec.frozen_belief_state, {})
        self.assertEqual(rec.candidate_trace, [])
        self.assertEqual(rec.ranking_score, 0.0)
        self.assertEqual(rec.confidence, 0.0)
        self.assertIn("no recommendation issued", rec.recommendation)


if __name__ == "__main__":
    unittest.main()

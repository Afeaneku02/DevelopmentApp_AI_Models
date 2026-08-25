from __future__ import annotations

import unittest

from pydantic import BaseModel, ValidationError

from src.common.enums import BeliefStatus, BeliefType, SourceType


class ClosedEnumTests(unittest.TestCase):
    def test_belief_type_has_exactly_nine_canonical_values(self) -> None:
        self.assertEqual(
            {member.value for member in BeliefType},
            {
                "behavioral_tendency",
                "routine_or_preference",
                "communication_or_learning_preference",
                "current_state_related",
                "goal_or_intention",
                "constraint_or_aversion",
                "cross_context_tendency",
                "recommendation_response_pattern",
                "sensitive_or_high_impact_inference",
            },
        )

    def test_source_type_has_exactly_seven_canonical_values(self) -> None:
        self.assertEqual(
            {member.value for member in SourceType},
            {
                "explicit_user_correction",
                "recorded_event",
                "explicit_user_statement",
                "repeated_pattern_summary",
                "model_observation",
                "llm_inference",
                "unverified_hypothesis",
            },
        )

    def test_belief_status_has_exactly_six_canonical_values(self) -> None:
        self.assertEqual(
            {member.value for member in BeliefStatus},
            {"candidate", "provisional", "validated", "contested", "outdated", "rejected"},
        )

    def test_unknown_belief_type_is_rejected_not_silently_mapped(self) -> None:
        class _Holder(BaseModel):
            belief_type: BeliefType

        with self.assertRaises(ValidationError):
            _Holder(belief_type="made_up_type")

    def test_unknown_source_type_is_rejected(self) -> None:
        class _Holder(BaseModel):
            source_type: SourceType

        with self.assertRaises(ValidationError):
            _Holder(source_type="legacy_source_string")


if __name__ == "__main__":
    unittest.main()

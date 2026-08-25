from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.beliefs.models import UserBelief

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _belief(**overrides):
    base = dict(
        belief_id="bel_88", user_id="usr_17", belief_type="behavioral_tendency",
        belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
        belief_value=True, confidence=0.68, supporting_evidence_count=4,
        contradicting_evidence_count=1, total_evidence_count=5,
        effective_support_count=3.62, effective_evidence_count=4.31,
        evidence_for=4, evidence_against=1, allowed_contexts=["fitness", "planning"],
        disallowed_contexts=["employment_decision", "medical_diagnosis"],
        sensitivity_class="normal", persistence_policy="retained",
        first_observed=datetime(2026, 8, 10, 18, tzinfo=timezone.utc),
        last_validated=datetime(2026, 8, 22, 17, 21, tzinfo=timezone.utc),
        status="validated",
        reasoning_summary="Repeated after-work completions exceed morning completions.",
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return base


class UserBeliefTests(unittest.TestCase):
    def test_matches_blueprint_worked_example(self) -> None:
        belief = UserBelief(**_belief())
        self.assertEqual(belief.belief_type.value, "behavioral_tendency")
        self.assertEqual(belief.status.value, "validated")

    def test_confidence_of_exactly_1_0_is_rejected(self) -> None:
        # The 0.98 ceiling exists specifically to prevent inferred confidence
        # from reaching mathematical certainty (blueprint section 3.4.1).
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(confidence=1.0))

    def test_confidence_strictly_between_0_and_0_02_is_rejected(self) -> None:
        # No branch in the scoring algorithm produces a value in this band:
        # it is either exactly 0.0 (no evidence) or clamped to >= 0.02.
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(confidence=0.01))

    def test_confidence_of_exactly_0_0_is_allowed(self) -> None:
        belief = UserBelief(**_belief(confidence=0.0, status="candidate"))
        self.assertEqual(belief.confidence, 0.0)

    def test_total_evidence_count_must_equal_sum_of_parts(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(total_evidence_count=999))

    def test_unregistered_belief_type_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(belief_type="not_a_real_belief_type"))


if __name__ == "__main__":
    unittest.main()

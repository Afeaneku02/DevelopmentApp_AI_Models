"""Regression tests: contract-governed numeric fields must accept only real
JSON numbers -- not bools (bool is a subclass of int in Python, so Pydantic's
default lax mode would otherwise coerce True/False to 1.0/0.0) and not
numeric strings like "0.95". A plain JSON int for a float field (e.g. 1 for
strength) is still a genuine number and must still be accepted.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.beliefs.models import BeliefEvidenceProposal, UserBelief

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _proposal(**overrides) -> dict:
    base = dict(
        belief_id="bel_88", user_id="usr_17", direction="support",
        event_id="evt_1042", observation_id="obs_310", source_event_ids=["evt_1042"],
        source_type="recorded_event", context_key="fitness", strength=0.85,
        source_reliability=0.95, observed_at=datetime(2026, 8, 22, 17, 20, tzinfo=timezone.utc),
        decay_lambda=0.015, model_version="observation-model-0.6", prompt_version="evidence_prompt_05",
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return base


def _belief(**overrides) -> dict:
    base = dict(
        belief_id="bel_88", user_id="usr_17", belief_type="behavioral_tendency",
        belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
        belief_value=True, confidence=0.68, supporting_evidence_count=4,
        contradicting_evidence_count=1, total_evidence_count=5,
        effective_support_count=3.62, effective_evidence_count=4.31,
        evidence_for=4, evidence_against=1, sensitivity_class="normal",
        persistence_policy="retained", first_observed=datetime(2026, 8, 10, 18, tzinfo=timezone.utc),
        last_validated=datetime(2026, 8, 22, 17, 21, tzinfo=timezone.utc), status="validated",
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return base


class BeliefEvidenceProposalStrictNumericTests(unittest.TestCase):
    def test_source_reliability_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(source_reliability=True))

    def test_source_reliability_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(source_reliability="0.95"))

    def test_strength_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(strength=True))

    def test_strength_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(strength="0.85"))

    def test_decay_lambda_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(decay_lambda=False))

    def test_decay_lambda_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**_proposal(decay_lambda="0.015"))

    def test_plain_int_is_still_accepted_for_strength(self) -> None:
        # strict=True still permits int -> float (a genuine JSON number),
        # it only rejects bool and str.
        proposal = BeliefEvidenceProposal(**_proposal(strength=1))
        self.assertEqual(proposal.strength, 1.0)
        self.assertIsInstance(proposal.strength, float)

    def test_plain_int_is_still_accepted_for_source_reliability(self) -> None:
        # explicit_user_correction's canonical default is exactly 1.00, so
        # source_reliability=1 (int) lands exactly on it -- no tolerance
        # float-precision edge case, isolating the int->float coercion check.
        proposal = BeliefEvidenceProposal(**_proposal(
            source_type="explicit_user_correction", source_reliability=1,
        ))
        self.assertEqual(proposal.source_reliability, 1.0)
        self.assertIsInstance(proposal.source_reliability, float)


class UserBeliefStrictNumericTests(unittest.TestCase):
    def test_confidence_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(confidence=True))

    def test_confidence_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(confidence="0.68"))

    def test_supporting_evidence_count_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(supporting_evidence_count=True))

    def test_supporting_evidence_count_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(supporting_evidence_count="4"))

    def test_effective_support_count_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(effective_support_count=True))

    def test_effective_support_count_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(effective_support_count="3.62"))

    def test_evidence_for_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserBelief(**_belief(evidence_for=True))

    def test_plain_int_is_still_accepted_for_confidence(self) -> None:
        # 0 is a valid confidence value (the no-evidence branch); ensure a
        # plain JSON int 0 is accepted, not just float 0.0.
        belief = UserBelief(**_belief(confidence=0, status="candidate"))
        self.assertEqual(belief.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

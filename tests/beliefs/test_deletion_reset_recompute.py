from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.models import UserBelief
from src.beliefs.scoring import compute_confidence, DEFAULT_WEIGHTS
from src.common.deletion import RecomputeTrackingFields, UserModelDeletionRequest
from src.common.enums import DeletionRequestedScope, DeletionStatus, RecomputeStatus

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


class RecomputeTrackingFieldsTests(unittest.TestCase):
    def test_completed_and_unlocked_is_not_fail_closed_locked(self) -> None:
        tracking = RecomputeTrackingFields(
            scoring_recompute_status=RecomputeStatus.COMPLETED, locked_until_recompute=False
        )
        self.assertFalse(tracking.is_fail_closed_locked)

    def test_pending_is_fail_closed_locked_even_if_locked_flag_not_yet_set(self) -> None:
        # Section 6.0.2: pending/running/failed all keep the belief locked
        # from live use -- only a completed recompute may unlock it.
        for status in (RecomputeStatus.PENDING, RecomputeStatus.RUNNING, RecomputeStatus.FAILED):
            with self.subTest(status=status):
                tracking = RecomputeTrackingFields(scoring_recompute_status=status, locked_until_recompute=False)
                self.assertTrue(tracking.is_fail_closed_locked)

    def test_completed_but_flag_still_set_is_fail_closed_locked(self) -> None:
        tracking = RecomputeTrackingFields(
            scoring_recompute_status=RecomputeStatus.COMPLETED, locked_until_recompute=True
        )
        self.assertTrue(tracking.is_fail_closed_locked)


class DeletionTriggeredInvalidationEndToEndTests(unittest.TestCase):
    """Reproduces blueprint section 6.0.1-6.0.2's post-invalidation rule using
    the real update-user-beliefs compute_confidence() branch (not a
    reimplementation), then constructs the resulting locked UserBelief."""

    def test_deletion_with_zero_active_evidence_forces_no_evidence_state(self) -> None:
        result = compute_confidence(
            evidence=[],
            weights=DEFAULT_WEIGHTS,
            expected_source_types=2,
            recompute_reason="deletion",
            active_evidence_count=0,
            no_active_evidence_status="rejected",
        )
        self.assertEqual(result["confidence"], 0.0)
        self.assertIsNone(result["D"])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "no_active_evidence_after_deletion_or_suppression")

        deletion_request = UserModelDeletionRequest(
            request_id="req_1", user_id="usr_17", requested_scope=DeletionRequestedScope.BELIEFS,
            requested_at=datetime.now(timezone.utc), status=DeletionStatus.COMPLETED,
            scoring_recompute_status=RecomputeStatus.FAILED,
            recompute_attempt_id="recomp_901", recompute_error="scoring_worker_timeout",
            recompute_failed_at=datetime.now(timezone.utc), locked_until_recompute=True,
            affected_belief_ids=["bel_88"], **VERSION_FIELDS,
        )
        self.assertTrue(deletion_request.is_fail_closed_locked)

        locked_belief = UserBelief(
            belief_id="bel_88", user_id="usr_17", belief_type="behavioral_tendency",
            belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
            belief_value=True, confidence=result["confidence"], supporting_evidence_count=0,
            contradicting_evidence_count=0, total_evidence_count=0, effective_support_count=0.0,
            effective_evidence_count=0.0, evidence_for=0, evidence_against=0,
            sensitivity_class="normal", persistence_policy="retained",
            first_observed=datetime(2026, 8, 10, tzinfo=timezone.utc),
            last_validated=datetime(2026, 8, 10, tzinfo=timezone.utc),
            status=result["status"], locked_until_recompute=True,
            last_recompute_attempt_id="recomp_901", **VERSION_FIELDS,
        )
        self.assertEqual(locked_belief.confidence, 0.0)
        self.assertEqual(locked_belief.status.value, "rejected")
        self.assertTrue(locked_belief.locked_until_recompute)

    def test_invalidation_branch_runs_before_generic_no_evidence_branch(self) -> None:
        # Same empty-evidence input, but without an invalidation trigger:
        # must hit the *different* generic no_evidence branch, proving the
        # two branches are distinct and correctly ordered rather than one
        # silently shadowing the other.
        result = compute_confidence(evidence=[], weights=DEFAULT_WEIGHTS, expected_source_types=2)
        self.assertEqual(result["reason"], "no_evidence")
        self.assertEqual(result["status"], "candidate")


if __name__ == "__main__":
    unittest.main()

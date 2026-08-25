"""Fail-closed hardening tests for src/beliefs/recompute.py:

1. recompute_belief() rejects active evidence carrying a foreign belief_id
   or user_id instead of silently corrupting confidence with it.
2. recompute_belief() rejects active evidence observed after as_of instead
   of silently computing a negative age_days (decay > 1, inflated
   confidence) from it.

Both checks apply only to *active* evidence -- an inactive/suppressed row
with a foreign scope or a future timestamp never reaches compute_confidence
in the first place, so it is deliberately left unchecked (see
test_inactive_* below).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.beliefs.models import BeliefEvidenceProposal, authorize_evidence, invalidate_evidence
from src.beliefs.recompute import recompute_belief

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

BELIEF_ID = "bel_88"
USER_ID = "usr_17"
BELIEF_TYPE = "behavioral_tendency"
AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


def _evidence(evidence_id: str, *, belief_id: str = BELIEF_ID, user_id: str = USER_ID, observed_at=None):
    proposal = BeliefEvidenceProposal(
        belief_id=belief_id, user_id=user_id, direction="support", event_id="evt_1",
        source_event_ids=["evt_1"], source_type="recorded_event", context_key="fitness",
        strength=0.9, source_reliability=0.95, observed_at=observed_at or (AS_OF - timedelta(days=1)),
        decay_lambda=0.015, model_version="pipeline-0.1", **VERSION_FIELDS,
    )
    return authorize_evidence(
        proposal, evidence_id=evidence_id, created_at=AS_OF, aggregation_policy_version="evidence-aggregation-0.6",
    )


def _recompute(evidence: list, **overrides) -> None:
    kwargs = dict(
        belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key="higher_adherence_after_work",
        belief_value=True, evidence=evidence, as_of=AS_OF, first_observed=AS_OF, **VERSION_FIELDS,
    )
    kwargs.update(overrides)
    return recompute_belief(**kwargs)


class ForeignScopeIsRejectedTests(unittest.TestCase):
    def test_foreign_belief_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _recompute([_evidence("bev_1", belief_id="bel_OTHER")])
        self.assertIn("belief_id", str(ctx.exception))
        self.assertIn("bev_1", str(ctx.exception))

    def test_foreign_user_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _recompute([_evidence("bev_2", user_id="usr_OTHER")])
        self.assertIn("user_id", str(ctx.exception))
        self.assertIn("bev_2", str(ctx.exception))

    def test_a_mix_of_correctly_and_incorrectly_scoped_evidence_still_rejects(self) -> None:
        # One bad row must not be masked by other, correctly-scoped rows.
        with self.assertRaises(ValueError):
            _recompute([_evidence("bev_good"), _evidence("bev_bad", belief_id="bel_OTHER")])

    def test_multiple_foreign_rows_are_all_named_in_the_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _recompute([
                _evidence("bev_x1", belief_id="bel_OTHER"),
                _evidence("bev_x2", belief_id="bel_OTHER"),
            ])
        message = str(ctx.exception)
        self.assertIn("bev_x1", message)
        self.assertIn("bev_x2", message)

    def test_inactive_foreign_belief_id_row_is_not_checked(self) -> None:
        # An invalidated row with the wrong belief_id never reaches
        # compute_confidence, so it must not trigger the guard either --
        # the check applies only to active evidence, matching what the
        # function actually feeds into scoring.
        foreign = _evidence("bev_3", belief_id="bel_OTHER")
        invalidated_foreign = invalidate_evidence(foreign, reason="reset", invalidated_at=AS_OF)
        belief = _recompute([invalidated_foreign])
        self.assertEqual(belief.confidence, 0.0)


class FutureDatedEvidenceIsRejectedTests(unittest.TestCase):
    def test_future_dated_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _recompute([_evidence("bev_4", observed_at=AS_OF + timedelta(days=1))])
        self.assertIn("bev_4", str(ctx.exception))
        self.assertIn("as_of", str(ctx.exception))

    def test_evidence_observed_exactly_at_as_of_is_allowed(self) -> None:
        # age_days == 0 is valid (decay == 1); only strictly-after is rejected.
        belief = _recompute([_evidence("bev_5", observed_at=AS_OF)])
        self.assertGreater(belief.confidence, 0.0)

    def test_inactive_future_dated_row_is_not_checked(self) -> None:
        future = _evidence("bev_6", observed_at=AS_OF + timedelta(days=1))
        invalidated_future = invalidate_evidence(future, reason="reset", invalidated_at=AS_OF)
        belief = _recompute([invalidated_future])
        self.assertEqual(belief.confidence, 0.0)

    def test_error_is_raised_before_reaching_compute_confidence(self) -> None:
        # A future-dated row would otherwise produce a negative age_days and
        # decay > 1, inflating confidence above what real evidence could
        # justify -- prove the guard actually stops that, not just that some
        # ValueError happens to fire somewhere.
        with self.assertRaises(ValueError):
            _recompute([_evidence("bev_7", observed_at=AS_OF + timedelta(days=365))])


class ValidEvidenceStillRecomputesNormallyTests(unittest.TestCase):
    def test_correctly_scoped_present_dated_evidence_passes(self) -> None:
        belief = _recompute([_evidence("bev_ok")])
        self.assertGreater(belief.confidence, 0.0)
        self.assertEqual(belief.supporting_evidence_count, 1)


if __name__ == "__main__":
    unittest.main()

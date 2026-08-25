"""End-to-end test of the first narrow adaptive-user-model pipeline:

UserEvent -> UserObservation -> BeliefEvidenceProposal -> authorized
BeliefEvidence -> UserBelief recompute.

Uses the blueprint's own recurring worked example -- after-work workout
completion, belief_key "higher_adherence_after_work" -- so the numbers here
are traceable back to the document (section 5.3's JSON contracts) rather
than an arbitrary scenario.

What this proves, matching the five things that must be preserved:
- source_event_ids stays the authoritative provenance at every stage.
- canonical source_type/source_reliability defaults are used, not invented.
- strict numeric validation is never bypassed anywhere in the chain.
- no stage can write a model-owned/backend-owned field.
- zero active evidence never leaves a stale confidence behind.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from src.beliefs.models import BeliefEvidence, active_evidence, authorize_evidence, invalidate_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation
from src.beliefs.recompute import recompute_belief
from src.beliefs.scoring import DEFAULT_WEIGHTS, EvidenceItem, compute_confidence
from src.common.registry import SOURCE_TYPE_RELIABILITY
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

BELIEF_ID = "bel_88"
USER_ID = "usr_17"
BELIEF_TYPE = "behavioral_tendency"
BELIEF_KEY = "higher_adherence_after_work"

# Three after-work workout completions, 5/3/1 days before the recompute.
AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)
_EVENT_TIMESTAMPS = [AS_OF - timedelta(days=5), AS_OF - timedelta(days=3), AS_OF - timedelta(days=1)]


def _run_event_through_the_pipeline(index: int, timestamp: datetime) -> BeliefEvidence:
    """Runs one UserEvent through steps 1-3 of the pipeline and returns the
    resulting authorized BeliefEvidence."""
    event = UserEvent(
        event_id=f"evt_104{index}", user_id=USER_ID, event_type="goal_completed",
        timestamp=timestamp,
        structured_data={"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:20"},
        source="app", goal_id="goal_8", **VERSION_FIELDS,
    )

    # Step 1: UserEvent -> UserObservation.
    observation, links = create_observation_from_event(
        event, observation_id=f"obs_31{index}", category="routine",
        observation_text="User completed an after-work workout.",
        importance=0.6, confidence=0.6, created_at=timestamp, **VERSION_FIELDS,
    )

    # Step 2: UserObservation -> BeliefEvidenceProposal.
    proposal = propose_evidence_from_observation(
        observation, links, belief_id=BELIEF_ID, direction="support", source_type="recorded_event",
        context_key="fitness", strength=0.9, model_version="pipeline-0.1", belief_type=BELIEF_TYPE,
        **VERSION_FIELDS,
    )

    # Step 3: BeliefEvidenceProposal -> authorized BeliefEvidence.
    return authorize_evidence(
        proposal, evidence_id=f"bev_50{index}", created_at=timestamp,
        aggregation_policy_version="evidence-aggregation-0.6",
    )


class WorkoutAfterWorkPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            _run_event_through_the_pipeline(index, ts) for index, ts in enumerate(_EVENT_TIMESTAMPS, start=1)
        ]

    def test_source_event_ids_traces_back_to_the_originating_event_at_every_stage(self) -> None:
        for evidence_row, expected_event_id in zip(self.evidence, ("evt_1041", "evt_1042", "evt_1043")):
            self.assertEqual(evidence_row.source_event_ids, [expected_event_id])
            self.assertEqual(evidence_row.event_id, expected_event_id)

    def test_canonical_source_type_and_reliability_default_used_throughout(self) -> None:
        for evidence_row in self.evidence:
            self.assertEqual(evidence_row.source_type.value, "recorded_event")
            self.assertEqual(evidence_row.source_reliability, SOURCE_TYPE_RELIABILITY["recorded_event"])

    def test_no_stage_produced_a_backend_authorized_mode_other_than_leaf_default(self) -> None:
        # Nothing in this pipeline ever calls authorize_evidence with
        # backend_validation_passed=True, so every row must still be leaf_default.
        for evidence_row in self.evidence:
            self.assertEqual(evidence_row.authorized_aggregation_mode.value, "leaf_default")
            self.assertEqual(evidence_row.aggregation_review_status.value, "not_required")

    def test_recompute_produces_a_belief_matching_a_direct_compute_confidence_call(self) -> None:
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.evidence, as_of=AS_OF, first_observed=_EVENT_TIMESTAMPS[0],
            **VERSION_FIELDS,
        )

        # Independently reconstruct what compute_confidence should have seen,
        # to prove recompute_belief() is not silently massaging the numbers.
        expected_items = [
            EvidenceItem(
                direction="support", strength=0.9, source_reliability=SOURCE_TYPE_RELIABILITY["recorded_event"],
                decay_lambda=0.015, age_days=(AS_OF - ts).total_seconds() / 86400.0, source_type="recorded_event",
            )
            for ts in _EVENT_TIMESTAMPS
        ]
        expected = compute_confidence(expected_items, DEFAULT_WEIGHTS, expected_source_types=2)

        self.assertAlmostEqual(belief.confidence, expected["confidence"], places=9)
        self.assertEqual(belief.supporting_evidence_count, 3)
        self.assertEqual(belief.contradicting_evidence_count, 0)
        self.assertEqual(belief.total_evidence_count, 3)
        self.assertAlmostEqual(belief.effective_support_count, expected["effective_support_count"], places=9)
        self.assertEqual(belief.evidence_for, 3)
        self.assertEqual(belief.evidence_against, 0)
        self.assertGreater(belief.confidence, 0.0)

    def test_recompute_defaults_sensitivity_and_persistence_from_the_belief_type_registry(self) -> None:
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.evidence, as_of=AS_OF, first_observed=_EVENT_TIMESTAMPS[0],
            **VERSION_FIELDS,
        )
        self.assertEqual(belief.sensitivity_class.value, "normal")
        self.assertEqual(belief.persistence_policy.value, "retained")

    def test_recompute_after_full_invalidation_drops_confidence_to_zero_not_stale(self) -> None:
        belief_before = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.evidence, as_of=AS_OF, first_observed=_EVENT_TIMESTAMPS[0],
            **VERSION_FIELDS,
        )
        self.assertGreater(belief_before.confidence, 0.0)

        invalidated = [
            invalidate_evidence(row, reason="deletion", invalidated_at=AS_OF) for row in self.evidence
        ]
        self.assertEqual(active_evidence(invalidated), [])

        belief_after = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=invalidated, as_of=AS_OF, first_observed=_EVENT_TIMESTAMPS[0],
            recompute_reason="deletion", **VERSION_FIELDS,
        )
        # Confidence must be exactly 0.0, not the prior ~0.4-ish value carried
        # forward -- there is no cache here for a stale number to hide in.
        self.assertEqual(belief_after.confidence, 0.0)
        self.assertEqual(belief_after.status.value, "outdated")
        self.assertEqual(belief_after.supporting_evidence_count, 0)
        self.assertEqual(belief_after.total_evidence_count, 0)
        # A *successful* recompute -- even one landing on zero active evidence
        # -- is not the fail-closed-locked case; that is reserved for a
        # recompute attempt that itself fails.
        self.assertFalse(belief_after.locked_until_recompute)

    def test_recompute_with_no_evidence_and_no_invalidation_reason_hits_the_generic_branch(self) -> None:
        # Same zero-confidence outcome, but via the *other* early-return
        # branch (support_mass + contradiction_mass == 0, not an
        # invalidation trigger) -- proving both branches are reachable and
        # distinct through this pipeline, not just the invalidation one.
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=[], as_of=AS_OF, first_observed=_EVENT_TIMESTAMPS[0], **VERSION_FIELDS,
        )
        self.assertEqual(belief.confidence, 0.0)
        self.assertEqual(belief.status.value, "candidate")

    def test_strict_numeric_validation_is_preserved_through_the_pipeline(self) -> None:
        # A bool strength must still be rejected when it flows in through
        # the propose step of the real pipeline, not just in isolation.
        event = UserEvent(
            event_id="evt_9001", user_id=USER_ID, event_type="goal_completed",
            timestamp=AS_OF, source="app", **VERSION_FIELDS,
        )
        observation, links = create_observation_from_event(
            event, observation_id="obs_9001", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=AS_OF, **VERSION_FIELDS,
        )
        with self.assertRaises(ValidationError):
            propose_evidence_from_observation(
                observation, links, belief_id=BELIEF_ID, direction="support", source_type="recorded_event",
                context_key="fitness", strength=True, model_version="pipeline-0.1", belief_type=BELIEF_TYPE,
                **VERSION_FIELDS,
            )

    def test_no_pipeline_stage_accepts_a_backend_owned_field_as_input(self) -> None:
        # None of the proposal-producing stages have a parameter for
        # authorized_aggregation_mode; the only place it can be set is
        # authorize_evidence(), which is the backend boundary itself.
        event = UserEvent(
            event_id="evt_9002", user_id=USER_ID, event_type="goal_completed",
            timestamp=AS_OF, source="app", **VERSION_FIELDS,
        )
        observation, links = create_observation_from_event(
            event, observation_id="obs_9002", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=AS_OF, **VERSION_FIELDS,
        )
        with self.assertRaises(TypeError):
            propose_evidence_from_observation(
                observation, links, belief_id=BELIEF_ID, direction="support", source_type="recorded_event",
                context_key="fitness", strength=0.9, model_version="pipeline-0.1", belief_type=BELIEF_TYPE,
                authorized_aggregation_mode="aggregate_replacement", **VERSION_FIELDS,
            )


if __name__ == "__main__":
    unittest.main()

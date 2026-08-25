"""Tests for src/beliefs/propose_evidence.py -- the first runtime slice over
the Phase 1 contracts, converting a UserEvent/UserObservation into a
BeliefEvidenceProposal (create-belief-evidence's runtime implementation).

Organized around the five properties this pipeline must prove:
1. event/observation provenance flows into source_event_ids.
2. source_type reliability defaults are used.
3. model output cannot set backend-owned fields.
4. aggregate_replacement remains proposal-only unless backend authorization
   explicitly approves it.
5. the proposal passes the shared create-belief-evidence validator.
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.beliefs.models import BeliefEvidence, BeliefEvidenceProposal, authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_event, propose_evidence_from_observation
from src.common.registry import SOURCE_TYPE_RELIABILITY
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation

_SKILL_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "shared" / "create-belief-evidence" / "scripts"
)
if str(_SKILL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS_DIR))

from validate_evidence import validate_row  # type: ignore  # noqa: E402

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _event(**overrides) -> UserEvent:
    base = dict(
        event_id="evt_1042", user_id="usr_17", event_type="goal_completed",
        timestamp=datetime(2026, 8, 22, 17, 20, tzinfo=timezone.utc), source="app", **VERSION_FIELDS,
    )
    base.update(overrides)
    return UserEvent(**base)


def _observation(**overrides) -> UserObservation:
    base = dict(
        observation_id="obs_310", user_id="usr_17", category="routine",
        observation="User has repeatedly completed after-work workouts.",
        importance=0.72, confidence=0.86, created_at=datetime(2026, 8, 22, 17, 21, tzinfo=timezone.utc),
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return UserObservation(**base)


def _link(event_id: str, role: str, observation_id: str = "obs_310") -> ObservationEvent:
    return ObservationEvent(
        observation_id=observation_id, event_id=event_id, link_role=role,
        created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
    )


_EVENT_KWARGS = dict(
    belief_id="bel_88", direction="support", source_type="recorded_event",
    context_key="fitness", strength=0.85, model_version="observation-model-0.6",
    belief_type="behavioral_tendency", **VERSION_FIELDS,
)
_OBSERVATION_KWARGS = dict(
    belief_id="bel_88", direction="support", source_type="model_observation",
    context_key="fitness", strength=0.8, model_version="observation-model-0.6",
    belief_type="behavioral_tendency", **VERSION_FIELDS,
)


class EventProvenanceTests(unittest.TestCase):
    """1. event provenance flows into source_event_ids."""

    def test_source_event_ids_is_exactly_the_one_event(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertEqual(proposal.source_event_ids, ["evt_1042"])

    def test_event_id_matches_the_event(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertEqual(proposal.event_id, "evt_1042")

    def test_user_id_and_observed_at_come_from_the_event(self) -> None:
        event = _event(user_id="usr_99", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        proposal = propose_evidence_from_event(event, **_EVENT_KWARGS)
        self.assertEqual(proposal.user_id, "usr_99")
        self.assertEqual(proposal.observed_at, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_a_different_event_id_flows_through_correctly(self) -> None:
        proposal = propose_evidence_from_event(_event(event_id="evt_9999"), **_EVENT_KWARGS)
        self.assertEqual(proposal.source_event_ids, ["evt_9999"])
        self.assertEqual(proposal.event_id, "evt_9999")


class ObservationProvenanceTests(unittest.TestCase):
    """1. observation provenance flows into source_event_ids via
    observation_events, never through an ad hoc field on the observation."""

    def test_source_event_ids_includes_every_linked_event_primary_and_supporting(self) -> None:
        links = [_link("evt_1040", "supporting"), _link("evt_1041", "supporting"), _link("evt_1042", "primary")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        self.assertEqual(proposal.source_event_ids, ["evt_1040", "evt_1041", "evt_1042"])

    def test_event_id_is_the_single_primary_link(self) -> None:
        links = [_link("evt_1040", "supporting"), _link("evt_1042", "primary")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        self.assertEqual(proposal.event_id, "evt_1042")

    def test_event_id_is_null_when_there_are_multiple_primary_links(self) -> None:
        links = [_link("evt_1040", "primary"), _link("evt_1042", "primary")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        self.assertIsNone(proposal.event_id)
        # source_event_ids remains complete and correct regardless.
        self.assertEqual(proposal.source_event_ids, ["evt_1040", "evt_1042"])

    def test_links_belonging_to_a_different_observation_are_ignored(self) -> None:
        links = [_link("evt_1042", "primary"), _link("evt_9999", "primary", observation_id="obs_other")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        self.assertEqual(proposal.source_event_ids, ["evt_1042"])

    def test_no_links_at_all_raises(self) -> None:
        with self.assertRaises(ValueError):
            propose_evidence_from_observation(_observation(), [], **_OBSERVATION_KWARGS)

    def test_no_primary_link_raises(self) -> None:
        links = [_link("evt_1040", "supporting")]
        with self.assertRaises(ValueError):
            propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)

    def test_observation_id_and_observed_at_come_from_the_observation(self) -> None:
        observation = _observation(observation_id="obs_555", created_at=datetime(2026, 2, 2, tzinfo=timezone.utc))
        links = [_link("evt_1042", "primary", observation_id="obs_555")]
        proposal = propose_evidence_from_observation(observation, links, **_OBSERVATION_KWARGS)
        self.assertEqual(proposal.observation_id, "obs_555")
        self.assertEqual(proposal.observed_at, datetime(2026, 2, 2, tzinfo=timezone.utc))


class SourceReliabilityDefaultTests(unittest.TestCase):
    """2. source_type reliability defaults are used."""

    def test_recorded_event_defaults_to_its_registry_reliability(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertEqual(proposal.source_reliability, SOURCE_TYPE_RELIABILITY["recorded_event"])

    def test_model_observation_defaults_to_its_own_registry_reliability(self) -> None:
        links = [_link("evt_1042", "primary")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        self.assertEqual(proposal.source_reliability, SOURCE_TYPE_RELIABILITY["model_observation"])
        self.assertNotEqual(SOURCE_TYPE_RELIABILITY["model_observation"], SOURCE_TYPE_RELIABILITY["recorded_event"])

    def test_explicit_reliability_within_tolerance_overrides_the_default(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{**_EVENT_KWARGS, "source_reliability": 0.90})
        self.assertEqual(proposal.source_reliability, 0.90)

    def test_unjustified_out_of_tolerance_override_is_rejected(self) -> None:
        # The runtime function does not bypass BeliefEvidenceProposal's own
        # deviation-justification validator.
        with self.assertRaises(ValidationError):
            propose_evidence_from_event(_event(), **{**_EVENT_KWARGS, "source_reliability": 0.20})

    def test_justified_out_of_tolerance_override_is_accepted(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "source_reliability": 0.20,
            "reliability_deviation_reason": "Deliberately lowered trust pending source audit.",
        })
        self.assertEqual(proposal.source_reliability, 0.20)

    def test_decay_lambda_defaults_from_belief_type_registry(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertEqual(proposal.decay_lambda, 0.015)  # behavioral_tendency's registry default

    def test_missing_decay_lambda_and_belief_type_raises(self) -> None:
        kwargs = {k: v for k, v in _EVENT_KWARGS.items() if k != "belief_type"}
        with self.assertRaises(ValueError):
            propose_evidence_from_event(_event(), **kwargs)


class BackendOwnedFieldsCannotBeSetTests(unittest.TestCase):
    """3. model output cannot set backend-owned fields."""

    def test_function_return_type_is_proposal_not_full_record(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertIsInstance(proposal, BeliefEvidenceProposal)
        self.assertNotIsInstance(proposal, BeliefEvidence)

    def test_function_has_no_parameter_for_backend_owned_fields(self) -> None:
        with self.assertRaises(TypeError):
            propose_evidence_from_event(
                _event(), **_EVENT_KWARGS, authorized_aggregation_mode="aggregate_replacement",
            )

    def test_strict_numeric_typing_is_not_bypassed_by_this_function(self) -> None:
        # Passing a bool for a numeric field must still be rejected, proving
        # this runtime layer does not smuggle values past BeliefEvidenceProposal's
        # own strict=True fields.
        with self.assertRaises(ValidationError):
            propose_evidence_from_event(_event(), **{**_EVENT_KWARGS, "strength": True})


class AggregateReplacementStaysProposalOnlyTests(unittest.TestCase):
    """4. aggregate_replacement remains proposal-only unless backend
    authorization explicitly approves it."""

    def test_proposing_aggregate_replacement_is_allowed_at_the_proposal_stage(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "proposed_aggregation_mode": "aggregate_replacement",
            "replaces_evidence_ids": ["bev_100"],
        })
        self.assertEqual(proposal.proposed_aggregation_mode.value, "aggregate_replacement")
        # It is still only a BeliefEvidenceProposal -- no authorized_aggregation_mode exists yet.
        self.assertNotIn("authorized_aggregation_mode", type(proposal).model_fields)

    def test_authorize_evidence_without_backend_approval_forces_leaf_default(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "proposed_aggregation_mode": "aggregate_replacement",
            "replaces_evidence_ids": ["bev_100"],
        })
        record = authorize_evidence(
            proposal, evidence_id="bev_501", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            # backend_validation_passed intentionally omitted (defaults False)
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "pending")

    def test_authorize_evidence_with_explicit_backend_approval_grants_replacement(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "proposed_aggregation_mode": "aggregate_replacement",
            "replaces_evidence_ids": ["bev_100"],
        })
        record = authorize_evidence(
            proposal, evidence_id="bev_502", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "aggregate_replacement")
        self.assertEqual(record.aggregation_review_status.value, "approved")

    def test_default_proposed_mode_is_leaf_default(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        self.assertEqual(proposal.proposed_aggregation_mode.value, "leaf_default")


class PassesSharedCreateBeliefEvidenceValidatorTests(unittest.TestCase):
    """5. the proposal passes the shared create-belief-evidence validator."""

    def test_event_derived_proposal_passes(self) -> None:
        proposal = propose_evidence_from_event(_event(), **_EVENT_KWARGS)
        row = proposal.model_dump(mode="json")
        self.assertEqual(validate_row(row), [])

    def test_observation_derived_proposal_passes(self) -> None:
        links = [_link("evt_1040", "supporting"), _link("evt_1042", "primary")]
        proposal = propose_evidence_from_observation(_observation(), links, **_OBSERVATION_KWARGS)
        row = proposal.model_dump(mode="json")
        self.assertEqual(validate_row(row), [])

    def test_justified_reliability_deviation_still_passes_the_shared_validator(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "source_reliability": 0.20,
            "reliability_deviation_reason": "Deliberately lowered trust pending source audit.",
        })
        row = proposal.model_dump(mode="json")
        self.assertEqual(validate_row(row), [])

    def test_aggregate_replacement_proposal_still_passes_the_shared_validator(self) -> None:
        proposal = propose_evidence_from_event(_event(), **{
            **_EVENT_KWARGS,
            "proposed_aggregation_mode": "aggregate_replacement",
            "replaces_evidence_ids": ["bev_100"],
        })
        row = proposal.model_dump(mode="json")
        self.assertEqual(validate_row(row), [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.models import BeliefEvidenceProposal, authorize_evidence
from src.common.provenance import validate_belief_evidence_provenance, validate_observation_provenance
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _event(event_id: str, user_id: str = "usr_17") -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=datetime.now(timezone.utc), source="app", **VERSION_FIELDS,
    )


def _observation(observation_id: str, user_id: str = "usr_17") -> UserObservation:
    return UserObservation(
        observation_id=observation_id, user_id=user_id, category="routine",
        observation="x", importance=0.5, confidence=0.5,
        created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
    )


def _link(observation_id: str, event_id: str, role: str) -> ObservationEvent:
    return ObservationEvent(
        observation_id=observation_id, event_id=event_id, link_role=role,
        created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
    )


class ObservationProvenanceTests(unittest.TestCase):
    def test_observation_with_primary_link_passes(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")],
            [_link("obs_1", "evt_1", "supporting"), _link("obs_1", "evt_2", "primary")],
            [_event("evt_1"), _event("evt_2")],
        )
        self.assertEqual(errors, [])

    def test_observation_with_no_primary_link_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_1", "evt_1", "supporting")], [_event("evt_1")],
        )
        self.assertTrue(any("no primary" in e for e in errors))

    def test_link_to_unknown_observation_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_999", "evt_1", "primary")], [_event("evt_1")],
        )
        self.assertTrue(any("unknown observation_id" in e for e in errors))

    def test_link_to_unknown_event_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_1", "evt_ghost", "primary")], [_event("evt_1")],
        )
        self.assertTrue(any("unknown event_id" in e for e in errors))

    def test_link_to_a_different_users_event_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1", user_id="usr_17")],
            [_link("obs_1", "evt_1", "primary")],
            [_event("evt_1", user_id="usr_OTHER")],
        )
        self.assertTrue(any("crosses users" in e for e in errors))

    def test_wrong_user_event_still_counts_toward_the_no_primary_link_check(self) -> None:
        # A wrong-user primary link is a genuine provenance violation, not a
        # "missing primary link" -- both errors should be reported, not one
        # masking the other.
        errors = validate_observation_provenance(
            [_observation("obs_1", user_id="usr_17")],
            [_link("obs_1", "evt_1", "primary")],
            [_event("evt_1", user_id="usr_OTHER")],
        )
        self.assertTrue(any("crosses users" in e for e in errors))
        self.assertFalse(any("no primary" in e for e in errors))


class BeliefEvidenceProvenanceTests(unittest.TestCase):
    def _evidence(self, event_ids: list[str], *, user_id: str = "usr_17", event_id: str | None = None):
        proposal = BeliefEvidenceProposal(
            belief_id="bel_1", user_id=user_id, direction="support", event_id=event_id or event_ids[0],
            source_event_ids=event_ids, source_type="recorded_event", context_key="fitness",
            strength=0.8, source_reliability=0.95, observed_at=datetime.now(timezone.utc),
            decay_lambda=0.015, model_version="m-0.6", **VERSION_FIELDS,
        )
        return authorize_evidence(
            proposal, evidence_id="bev_1", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )

    def test_evidence_with_known_events_passes(self) -> None:
        errors = validate_belief_evidence_provenance([self._evidence(["evt_1"])], [_event("evt_1")])
        self.assertEqual(errors, [])

    def test_evidence_referencing_unknown_event_fails(self) -> None:
        errors = validate_belief_evidence_provenance([self._evidence(["evt_ghost"])], [_event("evt_1")])
        self.assertTrue(any("unknown event_id" in e for e in errors))

    def test_evidence_referencing_a_different_users_event_fails(self) -> None:
        errors = validate_belief_evidence_provenance(
            [self._evidence(["evt_1"], user_id="usr_17")], [_event("evt_1", user_id="usr_OTHER")],
        )
        self.assertTrue(any("different user" in e for e in errors))

    def test_multiple_source_event_ids_all_checked(self) -> None:
        # One good, one bad -- the bad one must still be reported.
        errors = validate_belief_evidence_provenance(
            [self._evidence(["evt_1", "evt_ghost"], event_id="evt_1")], [_event("evt_1")],
        )
        self.assertTrue(any("evt_ghost" in e for e in errors))

    def test_event_id_outside_source_event_ids_is_caught_as_defense_in_depth(self) -> None:
        # BeliefEvidenceProposal's own model validator already makes this
        # unreachable through normal construction; model_construct() skips
        # that validation, simulating a row that reached this validator
        # through some other path (deserialization this process did not
        # itself validate). The provenance validator must not assume the
        # object in front of it was necessarily built safely.
        valid = self._evidence(["evt_1"])
        unsafe = valid.model_construct(**{**valid.model_dump(), "event_id": "evt_not_in_source_event_ids"})
        errors = validate_belief_evidence_provenance([unsafe], [_event("evt_1")])
        self.assertTrue(any("not included in its own source_event_ids" in e for e in errors))

    def test_duplicate_event_id_in_events_batch_is_rejected(self) -> None:
        # event_by_id = {event.event_id: event for event in events} would
        # otherwise silently collapse these two rows to whichever appears
        # last, making the result order-dependent instead of a hard failure.
        duplicated_events = [_event("evt_1"), _event("evt_1")]
        errors = validate_belief_evidence_provenance([self._evidence(["evt_1"])], duplicated_events)
        self.assertTrue(any("Duplicate event_id" in e for e in errors))

    def test_cross_user_duplicate_event_id_is_rejected_regardless_of_order(self) -> None:
        # Two different UserEvent rows sharing one event_id but belonging to
        # two different users -- the most dangerous form of this bug, since
        # a naive dict comprehension would make the user_id check pass or
        # fail purely based on which of the two happened to be listed last.
        evidence = self._evidence(["evt_1"], user_id="usr_17")
        events_a_last = [_event("evt_1", user_id="usr_OTHER"), _event("evt_1", user_id="usr_17")]
        events_b_last = [_event("evt_1", user_id="usr_17"), _event("evt_1", user_id="usr_OTHER")]

        errors_a = validate_belief_evidence_provenance([evidence], events_a_last)
        errors_b = validate_belief_evidence_provenance([evidence], events_b_last)

        # Both orderings must fail identically -- the result must not depend
        # on which same-id event happened to be listed last.
        self.assertTrue(any("Duplicate event_id" in e for e in errors_a))
        self.assertTrue(any("Duplicate event_id" in e for e in errors_b))


if __name__ == "__main__":
    unittest.main()

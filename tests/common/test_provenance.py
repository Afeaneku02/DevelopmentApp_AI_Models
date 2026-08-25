from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.models import authorize_evidence
from src.beliefs.models import BeliefEvidenceProposal
from src.common.provenance import validate_belief_evidence_provenance, validate_observation_provenance
from src.observations.models import ObservationEvent, UserObservation

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _observation(observation_id: str) -> UserObservation:
    return UserObservation(
        observation_id=observation_id, user_id="usr_17", category="routine",
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
            known_event_ids={"evt_1", "evt_2"},
        )
        self.assertEqual(errors, [])

    def test_observation_with_no_primary_link_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_1", "evt_1", "supporting")], known_event_ids={"evt_1"}
        )
        self.assertTrue(any("no primary" in e for e in errors))

    def test_link_to_unknown_observation_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_999", "evt_1", "primary")], known_event_ids={"evt_1"}
        )
        self.assertTrue(any("unknown observation_id" in e for e in errors))

    def test_link_to_unknown_event_fails(self) -> None:
        errors = validate_observation_provenance(
            [_observation("obs_1")], [_link("obs_1", "evt_ghost", "primary")], known_event_ids={"evt_1"}
        )
        self.assertTrue(any("unknown event_id" in e for e in errors))


class BeliefEvidenceProvenanceTests(unittest.TestCase):
    def _evidence(self, event_ids: list[str]):
        proposal = BeliefEvidenceProposal(
            belief_id="bel_1", user_id="usr_17", direction="support", source_event_ids=event_ids,
            source_type="recorded_event", context_key="fitness", strength=0.8, source_reliability=0.95,
            observed_at=datetime.now(timezone.utc), decay_lambda=0.015, model_version="m-0.6", **VERSION_FIELDS,
        )
        return authorize_evidence(
            proposal, evidence_id="bev_1", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )

    def test_evidence_with_known_event_ids_passes(self) -> None:
        errors = validate_belief_evidence_provenance([self._evidence(["evt_1"])], known_event_ids={"evt_1"})
        self.assertEqual(errors, [])

    def test_evidence_referencing_unknown_event_fails(self) -> None:
        errors = validate_belief_evidence_provenance([self._evidence(["evt_ghost"])], known_event_ids={"evt_1"})
        self.assertTrue(any("unknown event_id" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

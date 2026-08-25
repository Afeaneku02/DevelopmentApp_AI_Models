"""Tests for propose_evidence_from_observation_validated() in
src/beliefs/propose_evidence.py -- the cross-record-integrity-checked
wrapper around the pure propose_evidence_from_observation().

Covers exactly the cases the validation layer exists to catch:
- missing linked event rejected
- wrong-user linked event rejected
- no primary observation link rejected
- valid end-to-end path still passes

(evidence-side source_event_id validation against unknown/wrong-user events
is covered directly against validate_belief_evidence_provenance() in
tests/common/test_provenance.py, since no wrapper around authorize_evidence()
was requested for this phase.)
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.observations.models import ObservationEvent

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

_KWARGS = dict(
    belief_id="bel_88", direction="support", source_type="recorded_event",
    context_key="fitness", strength=0.9, model_version="pipeline-0.1",
    belief_type="behavioral_tendency", **VERSION_FIELDS,
)


def _event(event_id: str = "evt_1042", user_id: str = "usr_17", **overrides) -> UserEvent:
    base = dict(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=datetime(2026, 8, 22, 17, 20, tzinfo=timezone.utc), source="app", **VERSION_FIELDS,
    )
    base.update(overrides)
    return UserEvent(**base)


def _observation_and_links(event: UserEvent):
    return create_observation_from_event(
        event, observation_id="obs_1042", category="routine", observation_text="x",
        importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
    )


class ValidPathStillPassesTests(unittest.TestCase):
    def test_valid_end_to_end_path_passes(self) -> None:
        event = _event()
        observation, links = _observation_and_links(event)
        proposal = propose_evidence_from_observation_validated(observation, links, [event], **_KWARGS)
        self.assertEqual(proposal.source_event_ids, ["evt_1042"])
        self.assertEqual(proposal.event_id, "evt_1042")

    def test_valid_path_with_extra_supporting_link_still_passes(self) -> None:
        primary_event = _event("evt_1042")
        supporting_event = _event("evt_1041")
        observation, links = _observation_and_links(primary_event)
        links = links + [ObservationEvent(
            observation_id=observation.observation_id, event_id="evt_1041", link_role="supporting",
            created_at=primary_event.timestamp, **VERSION_FIELDS,
        )]
        proposal = propose_evidence_from_observation_validated(
            observation, links, [primary_event, supporting_event], **_KWARGS
        )
        self.assertEqual(proposal.source_event_ids, ["evt_1041", "evt_1042"])


class MissingLinkedEventIsRejectedTests(unittest.TestCase):
    def test_missing_linked_event_is_rejected(self) -> None:
        event = _event()
        observation, links = _observation_and_links(event)
        with self.assertRaises(ValueError) as ctx:
            propose_evidence_from_observation_validated(observation, links, [], **_KWARGS)
        self.assertIn("unknown event_id", str(ctx.exception))

    def test_propose_evidence_from_observation_is_never_called_on_a_missing_event(self) -> None:
        # The underlying pure function has no source_events to trust, so if
        # validation were skipped it would still succeed (it derives
        # source_event_ids purely from the links) -- proving the wrapper
        # actually gates the call, not just that an error happens somewhere.
        event = _event()
        observation, links = _observation_and_links(event)
        with self.assertRaises(ValueError):
            propose_evidence_from_observation_validated(observation, links, [], **_KWARGS)


class WrongUserLinkedEventIsRejectedTests(unittest.TestCase):
    def test_wrong_user_linked_event_is_rejected(self) -> None:
        event = _event(user_id="usr_17")
        observation, links = _observation_and_links(event)
        wrong_user_event = _event(user_id="usr_OTHER")
        with self.assertRaises(ValueError) as ctx:
            propose_evidence_from_observation_validated(observation, links, [wrong_user_event], **_KWARGS)
        self.assertIn("crosses users", str(ctx.exception))


class NoPrimaryLinkIsRejectedTests(unittest.TestCase):
    def test_observation_with_only_a_supporting_link_is_rejected(self) -> None:
        event = _event()
        observation, links = _observation_and_links(event)
        supporting_only = [
            ObservationEvent(
                observation_id=link.observation_id, event_id=link.event_id, link_role="supporting",
                created_at=link.created_at, **VERSION_FIELDS,
            )
            for link in links
        ]
        with self.assertRaises(ValueError) as ctx:
            propose_evidence_from_observation_validated(observation, supporting_only, [event], **_KWARGS)
        self.assertIn("no primary", str(ctx.exception))

    def test_observation_with_no_links_at_all_is_rejected(self) -> None:
        event = _event()
        observation, _ = _observation_and_links(event)
        with self.assertRaises(ValueError) as ctx:
            propose_evidence_from_observation_validated(observation, [], [event], **_KWARGS)
        self.assertIn("no primary", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

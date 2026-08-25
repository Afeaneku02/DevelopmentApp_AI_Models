from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.observations.models import ObservationEvent, UserObservation

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


class UserObservationTests(unittest.TestCase):
    def test_matches_blueprint_worked_example(self) -> None:
        observation = UserObservation(
            observation_id="obs_310",
            user_id="usr_17",
            category="routine",
            observation="User has repeatedly completed after-work workouts.",
            importance=0.72,
            confidence=0.86,
            created_at=datetime(2026, 8, 22, 17, 21, tzinfo=timezone.utc),
            **VERSION_FIELDS,
        )
        self.assertEqual(observation.importance, 0.72)

    def test_importance_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserObservation(
                observation_id="obs_1", user_id="usr_1", category="routine",
                observation="x", importance=1.5, confidence=0.5,
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )

    def test_has_no_event_id_or_event_ids_field(self) -> None:
        # Blueprint section 18 Definition of Done: "no event_id/event_ids
        # schema mismatch remains" -- proven structurally, not just by
        # documentation, since provenance must go through observation_events.
        self.assertNotIn("event_id", UserObservation.model_fields)
        self.assertNotIn("event_ids", UserObservation.model_fields)

    def test_importance_bool_is_rejected(self) -> None:
        # bool is a subclass of int in Python; lax Pydantic would otherwise
        # silently coerce True/False into 1.0/0.0.
        with self.assertRaises(ValidationError):
            UserObservation(
                observation_id="obs_1", user_id="usr_1", category="routine",
                observation="x", importance=True, confidence=0.5,
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )

    def test_importance_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserObservation(
                observation_id="obs_1", user_id="usr_1", category="routine",
                observation="x", importance="0.72", confidence=0.5,
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )

    def test_confidence_bool_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserObservation(
                observation_id="obs_1", user_id="usr_1", category="routine",
                observation="x", importance=0.5, confidence=False,
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )

    def test_confidence_numeric_string_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserObservation(
                observation_id="obs_1", user_id="usr_1", category="routine",
                observation="x", importance=0.5, confidence="0.86",
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )

    def test_plain_int_is_still_accepted_for_importance(self) -> None:
        observation = UserObservation(
            observation_id="obs_1", user_id="usr_1", category="routine",
            observation="x", importance=1, confidence=0.5,
            created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
        )
        self.assertEqual(observation.importance, 1.0)


class ObservationEventTests(unittest.TestCase):
    def test_matches_blueprint_worked_example(self) -> None:
        link = ObservationEvent(
            observation_id="obs_310", event_id="evt_1042", link_role="primary",
            created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
        )
        self.assertEqual(link.link_role.value, "primary")

    def test_invalid_link_role_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ObservationEvent(
                observation_id="obs_1", event_id="evt_1", link_role="secondary",
                created_at=datetime.now(timezone.utc), **VERSION_FIELDS,
            )


if __name__ == "__main__":
    unittest.main()

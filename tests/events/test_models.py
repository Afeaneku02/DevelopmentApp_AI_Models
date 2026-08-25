from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.events.models import UserEvent

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


class UserEventTests(unittest.TestCase):
    def test_matches_blueprint_worked_example(self) -> None:
        event = UserEvent(
            event_id="evt_1042",
            user_id="usr_17",
            event_type="goal_completed",
            timestamp=datetime(2026, 8, 22, 17, 20, tzinfo=timezone.utc),
            raw_content=None,
            structured_data={"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:20"},
            source="app",
            goal_id="goal_8",
            session_id="sess_55",
            **VERSION_FIELDS,
        )
        self.assertEqual(event.event_id, "evt_1042")
        self.assertIsNone(event.raw_content)

    def test_raw_content_only_event_is_valid(self) -> None:
        event = UserEvent(
            event_id="evt_2001",
            user_id="usr_17",
            event_type="chat_message",
            timestamp=datetime.now(timezone.utc),
            raw_content="I want to study every night after dinner.",
            source="app",
            **VERSION_FIELDS,
        )
        self.assertIsNone(event.structured_data)

    def test_missing_required_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserEvent(user_id="usr_17", event_type="x", timestamp=datetime.now(timezone.utc), source="app", **VERSION_FIELDS)

    def test_blank_event_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserEvent(
                event_id="", user_id="usr_17", event_type="x",
                timestamp=datetime.now(timezone.utc), source="app", **VERSION_FIELDS,
            )

    def test_extra_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UserEvent(
                event_id="evt_1", user_id="usr_17", event_type="x",
                timestamp=datetime.now(timezone.utc), source="app", event_ids=["evt_1"], **VERSION_FIELDS,
            )


if __name__ == "__main__":
    unittest.main()

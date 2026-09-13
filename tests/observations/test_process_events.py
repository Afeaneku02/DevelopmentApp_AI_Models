"""Tests for the user_events -> user_observations pipeline step
(src/observations/process_events.py + tools/process_user_events.py).

Proves the guarantees the task requires:
1. check_in_recorded maps to a conservative observation.
2. roadmap_step_completed maps to a conservative observation.
3. repeated runs create no duplicate observations.
4. unsupported event types (and malformed data on a supported type) are
   skipped safely, never raising.
5. no private/raw text is required, read, or persisted -- only
   structured_data, and only the specific keys each mapping expects.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.events.models import UserEvent
from src.observations.process_events import process_user_events
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
_CLI = Path(__file__).resolve().parents[2] / "tools" / "process_user_events.py"


def _event(
    event_id: str, *, user_id: str = "usr_1", event_type: str, structured_data: dict | None,
    raw_content: str | None = None, timestamp: datetime | None = None,
) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type=event_type,
        timestamp=timestamp or AS_OF - timedelta(days=1),
        structured_data=structured_data, raw_content=raw_content,
        source="better_you", **VERSION_FIELDS,
    )


class CheckInRecordedTests(unittest.TestCase):
    def test_maps_to_a_conservative_observation(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
            links = repo.list_observation_events_for([o.observation_id for o in observations])
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["created"])
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.category, "check_in")
        self.assertEqual(observation.observation, "User completed a scheduled check-in.")
        self.assertEqual(observation.importance, 0.3)
        self.assertEqual(observation.confidence, 0.9)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].event_id, "evt_1")
        self.assertEqual(links[0].link_role.value, "primary")

    def test_wording_varies_only_by_the_reported_response_enum(self) -> None:
        repo = Repository.in_memory()
        try:
            for index, response in enumerate(["yes", "no", "partly", "skipped"], start=1):
                repo.insert_event(_event(
                    f"evt_{index}", event_type="check_in_recorded",
                    structured_data={"goalId": "goal_1", "checkInId": f"ci_{index}", "response": response},
                ))
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = {
                o.observation for o in repo.list_observations(user_id="usr_1")
            }
        finally:
            repo.close()

        self.assertEqual(
            observations,
            {
                "User completed a scheduled check-in.",
                "User recorded a missed check-in.",
                "User recorded a partially completed check-in.",
                "User skipped a scheduled check-in.",
            },
        )


class RoadmapStepCompletedTests(unittest.TestCase):
    def test_maps_to_a_conservative_observation(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="roadmap_step_completed",
                structured_data={
                    "goalId": "goal_1", "roadmapId": "rm_1",
                    "milestoneId": "ms_1", "actionStepId": "step_1",
                },
            ))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["created"])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].category, "goal_progress")
        self.assertEqual(observations[0].observation, "User completed a roadmap action step.")
        self.assertEqual(observations[0].importance, 0.5)
        self.assertEqual(observations[0].confidence, 0.9)


class NoDuplicatesOnRerunTests(unittest.TestCase):
    def test_rerunning_creates_no_duplicate_observations(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            first = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            second = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in first.outcomes], ["created"])
        self.assertEqual([o.action for o in second.outcomes], ["skipped_already_processed"])
        self.assertEqual(second.created_observation_ids, [])
        self.assertEqual(len(observations), 1)

    def test_adding_a_new_event_between_runs_only_creates_the_new_one(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)

            repo.insert_event(_event(
                "evt_2", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_2", "response": "yes"},
            ))
            second = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.event_id, o.action) for o in second.outcomes),
            [("evt_1", "skipped_already_processed"), ("evt_2", "created")],
        )
        self.assertEqual(len(observations), 2)


class UnsupportedAndMalformedEventsTests(unittest.TestCase):
    def test_unsupported_event_types_are_skipped_safely(self) -> None:
        repo = Repository.in_memory()
        try:
            for index, event_type in enumerate(
                ["goal_created", "goal_paused", "goal_completed", "roadmap_generated", "totally_unknown_type"],
                start=1,
            ):
                repo.insert_event(_event(f"evt_{index}", event_type=event_type, structured_data={"anything": "x"}))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertTrue(all(o.action == "skipped_unsupported_type" for o in result.outcomes))
        self.assertEqual(observations, [])

    def test_malformed_structured_data_on_a_supported_type_is_skipped_safely(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event("evt_1", event_type="check_in_recorded", structured_data=None))
            repo.insert_event(_event(
                "evt_2", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "not_a_real_response"},
            ))
            repo.insert_event(_event(
                "evt_3", event_type="check_in_recorded", structured_data={"response": "yes"},  # missing ids
            ))
            repo.insert_event(_event(
                "evt_4", event_type="roadmap_step_completed",
                structured_data={"goalId": "goal_1"},  # missing roadmapId/milestoneId/actionStepId
            ))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        actions = {o.event_id: o.action for o in result.outcomes}
        self.assertEqual(
            actions,
            {
                "evt_1": "skipped_malformed_data",
                "evt_2": "skipped_malformed_data",
                "evt_3": "skipped_malformed_data",
                "evt_4": "skipped_malformed_data",
            },
        )
        self.assertEqual(observations, [])

    def test_never_raises_on_a_mix_of_supported_unsupported_and_malformed_events(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "g", "checkInId": "c", "response": "yes"},
            ))
            repo.insert_event(_event("evt_2", event_type="goal_created", structured_data={"goalId": "g"}))
            repo.insert_event(_event("evt_3", event_type="check_in_recorded", structured_data=None))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.event_id, o.action) for o in result.outcomes),
            [("evt_1", "created"), ("evt_2", "skipped_unsupported_type"), ("evt_3", "skipped_malformed_data")],
        )

    def test_unknown_or_other_users_event_ids_are_reported_as_not_found(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_owned", user_id="usr_1", event_type="check_in_recorded",
                structured_data={"goalId": "g", "checkInId": "c", "response": "yes"},
            ))
            repo.insert_event(_event(
                "evt_other_user", user_id="usr_2", event_type="check_in_recorded",
                structured_data={"goalId": "g", "checkInId": "c", "response": "yes"},
            ))
            result = process_user_events(
                repo, user_id="usr_1", as_of=AS_OF, persist=True,
                event_ids=["evt_owned", "evt_other_user", "evt_does_not_exist"],
            )
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.event_id, o.action) for o in result.outcomes),
            [
                ("evt_does_not_exist", "skipped_not_found"),
                ("evt_other_user", "skipped_not_found"),
                ("evt_owned", "created"),
            ],
        )


class NoPrivateOrRawTextTests(unittest.TestCase):
    def test_raw_content_is_never_required(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded", raw_content=None,
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
        finally:
            repo.close()
        self.assertEqual([o.action for o in result.outcomes], ["created"])

    def test_raw_content_is_never_read_or_persisted_even_when_present(self) -> None:
        secret = "PRIVATE-CHECK-IN-NOTE-should-never-appear-in-any-observation"
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded", raw_content=secret,
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual(len(observations), 1)
        self.assertNotIn(secret, observations[0].observation)
        self.assertNotIn(secret, observations[0].category)

    def test_unexpected_extra_structured_data_keys_never_leak_into_the_observation(self) -> None:
        # Simulates a hypothetical future event carrying more than this
        # pipeline's mapping expects - only the specific keys each drafter
        # reads may ever influence anything; everything else is ignored,
        # not incorporated.
        secret = "UNEXPECTED-EXTRA-FIELD-should-never-appear-in-any-observation"
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={
                    "goalId": "goal_1", "checkInId": "ci_1", "response": "yes",
                    "note": secret,
                },
            ))
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual(observations[0].observation, "User completed a scheduled check-in.")
        self.assertNotIn(secret, observations[0].observation)


class DryRunTests(unittest.TestCase):
    def test_dry_run_writes_nothing(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
            result = process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=False)
            observations = repo.list_observations(user_id="usr_1")
        finally:
            repo.close()

        self.assertFalse(result.persisted)
        self.assertEqual([o.action for o in result.outcomes], ["would_create"])
        self.assertEqual(result.created_observation_ids, ["obs_evt_1"])
        self.assertEqual(observations, [])


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class ProcessUserEventsCliTests(unittest.TestCase):
    def _seed(self, db_path: str) -> None:
        repo = Repository.at_path(db_path)
        try:
            repo.insert_event(_event(
                "evt_1", event_type="check_in_recorded",
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
            ))
        finally:
            repo.close()

    def test_missing_db_exits_nonzero_without_a_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.sqlite3")
            result = _run_cli(["--db", missing, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 1)
            self.assertIn("no such database file", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_dry_run_leaves_the_db_byte_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dry run", result.stderr)
            self.assertEqual(Path(db_path).read_bytes(), before)

    def test_persist_writes_the_observation(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("persisted 1 observation", result.stderr)

            payload = json.loads(result.stdout)
            self.assertEqual(payload["created_observation_ids"], ["obs_evt_1"])

            repo = Repository.at_path(db_path)
            try:
                observations = repo.list_observations(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(observations), 1)

    def test_persist_twice_creates_no_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            first = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("persisted 1 observation", first.stderr)

            second = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("persisted 0 observation", second.stderr)
            payload = json.loads(second.stdout)
            self.assertEqual(payload["created_observation_ids"], [])

            repo = Repository.at_path(db_path)
            try:
                observations = repo.list_observations(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(observations), 1)


if __name__ == "__main__":
    unittest.main()

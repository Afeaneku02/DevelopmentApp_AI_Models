"""Tests for tools/add_user_observation.py -- the manual observation-intake
CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for tools/add_user_event.py, so these tests exercise the actual
argparse wiring and process exit codes. Referenced events are seeded through
tools/add_user_event.py itself (the already-tested event-intake CLI) rather
than by hand-writing rows, and every persistence claim is double-checked by
reading the database back through src.storage.repository.Repository
directly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"


def _run_observation(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ADD_OBSERVATION_SCRIPT), *args], capture_output=True, text=True,
    )


def _seed_event(db_path: str, *, user_id: str, event_id: str) -> None:
    result = subprocess.run(
        [
            sys.executable, str(_ADD_EVENT_SCRIPT), "--db", db_path, "--user-id", user_id,
            "--event-id", event_id, "--event-type", "goal_completed", "--source", "app",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


class ValidSingleEventInsertTests(unittest.TestCase):
    def test_single_event_observation_is_saved_with_primary_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--category", "routine",
                "--observation", "User completed a workout.",
                "--importance", "0.6", "--confidence", "0.6",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["observation"]["observation_id"], "obs_1")
            self.assertEqual(printed["observation"]["user_id"], "usr_1")
            self.assertEqual(len(printed["observation_events"]), 1)
            self.assertEqual(printed["observation_events"][0]["event_id"], "evt_1")
            self.assertEqual(printed["observation_events"][0]["link_role"], "primary")

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_observation("obs_1")
                links = repo.list_observation_events("obs_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0].link_role.value, "primary")


class ValidMultiEventInsertTests(unittest.TestCase):
    def test_multi_event_observation_uses_explicit_primary_and_marks_rest_supporting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_1", event_id="evt_2")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--event-id", "evt_2", "--primary-event-id", "evt_2",
                "--category", "routine", "--observation", "pattern",
                "--importance", "0.6", "--confidence", "0.6",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            links_by_event = {link["event_id"]: link["link_role"] for link in printed["observation_events"]}
            self.assertEqual(links_by_event, {"evt_1": "supporting", "evt_2": "primary"})
            primary_links = [link for link in printed["observation_events"] if link["link_role"] == "primary"]
            self.assertEqual(len(primary_links), 1)

            repo = Repository.at_path(db_path)
            try:
                links = repo.list_observation_events("obs_1")
            finally:
                repo.close()
            roles_by_event = {link.event_id: link.link_role.value for link in links}
            self.assertEqual(roles_by_event, {"evt_1": "supporting", "evt_2": "primary"})
            self.assertEqual(sum(1 for link in links if link.link_role.value == "primary"), 1)


class DuplicateEventIdRejectionTests(unittest.TestCase):
    def test_duplicate_event_id_is_rejected_and_nothing_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--event-id", "evt_1", "--primary-event-id", "evt_1",
                "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate --event-id", result.stderr)
            self.assertIn("evt_1", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
                self.assertEqual(repo.list_observation_events("obs_1"), [])
            finally:
                repo.close()

    def test_duplicate_event_id_among_multiple_distinct_ids_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_1", event_id="evt_2")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--event-id", "evt_2", "--event-id", "evt_1",
                "--primary-event-id", "evt_1", "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate --event-id", result.stderr)
            self.assertIn("evt_1", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
                self.assertEqual(repo.list_observation_events("obs_1"), [])
            finally:
                repo.close()


class UnknownEventRejectionTests(unittest.TestCase):
    def test_unknown_event_id_is_rejected_and_nothing_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_ghost", "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown event_id", result.stderr)
            self.assertIn("evt_ghost", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
            finally:
                repo.close()


class UserMismatchRejectionTests(unittest.TestCase):
    def test_event_belonging_to_a_different_user_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_OTHER", event_id="evt_other")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_other", "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different user", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
            finally:
                repo.close()


class PrimaryLinkRejectionTests(unittest.TestCase):
    def test_ambiguous_primary_with_multiple_event_ids_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_1", event_id="evt_2")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--event-id", "evt_2", "--category", "routine",
                "--observation", "x", "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--primary-event-id is required", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
            finally:
                repo.close()

    def test_primary_event_id_not_among_event_ids_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_1", event_id="evt_2")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--primary-event-id", "evt_2", "--category", "routine",
                "--observation", "x", "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is not one of the --event-id values", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
            finally:
                repo.close()

    def test_missing_event_id_entirely_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("At least one --event-id is required", result.stderr)


class InvalidNumericValueTests(unittest.TestCase):
    def test_out_of_range_importance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--category", "routine", "--observation", "x",
                "--importance", "1.5", "--confidence", "0.5",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation", result.stderr.lower())

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_observation("obs_1"))
            finally:
                repo.close()

    def test_non_numeric_confidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "not-a-number",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--confidence", result.stderr)


class CliOutputShapeTests(unittest.TestCase):
    def test_output_contains_observation_and_observation_events_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")

            result = _run_observation([
                "--db", db_path, "--observation-id", "obs_1", "--user-id", "usr_1",
                "--event-id", "evt_1", "--category", "routine", "--observation", "x",
                "--importance", "0.5", "--confidence", "0.5",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(set(printed.keys()), {"observation", "observation_events"})
            self.assertIsInstance(printed["observation"], dict)
            self.assertIsInstance(printed["observation_events"], list)
            for expected_field in (
                "observation_id", "user_id", "category", "observation",
                "importance", "confidence", "created_at",
            ):
                self.assertIn(expected_field, printed["observation"])
            for link in printed["observation_events"]:
                for expected_field in ("observation_id", "event_id", "link_role", "created_at"):
                    self.assertIn(expected_field, link)


if __name__ == "__main__":
    unittest.main()

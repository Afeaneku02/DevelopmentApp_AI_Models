"""Tests for tools/add_user_event.py -- the manual event-intake CLI.

Runs the CLI as a real subprocess (matching tests/demo_user_model's approach
for its own CLI test) so these tests exercise the actual argparse wiring and
process exit codes, not just an in-process function call. Every persistence
claim is double-checked by reading the on-disk database back through
src.storage.repository.Repository directly, not by trusting the CLI's own
stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args], capture_output=True, text=True,
    )


class ValidInsertTests(unittest.TestCase):
    def test_valid_event_is_saved_and_printed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data", '{"goal": "workout"}', "--raw-content", "did it",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["event_id"], "evt_1")
            self.assertEqual(printed["user_id"], "usr_1")
            self.assertEqual(printed["event_type"], "goal_completed")
            self.assertEqual(printed["structured_data"], {"goal": "workout"})
            self.assertEqual(printed["raw_content"], "did it")

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_event("evt_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.user_id, "usr_1")
            self.assertEqual(stored.structured_data, {"goal": "workout"})

    def test_timestamp_defaults_to_now_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertIsNotNone(printed["timestamp"])

    def test_explicit_iso_timestamp_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--timestamp", "2026-01-01T12:00:00Z",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertTrue(printed["timestamp"].startswith("2026-01-01T12:00:00"))


class InvalidJsonTests(unittest.TestCase):
    def test_malformed_structured_data_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data", "{bad json",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid JSON", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()

    def test_structured_data_that_is_valid_json_but_not_an_object_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data", "[1, 2, 3]",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation", result.stderr.lower())


class MissingRequiredFieldTests(unittest.TestCase):
    def test_missing_required_cli_argument_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--source", result.stderr)

    def test_empty_required_field_value_fails_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "", "--source", "app",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation", result.stderr.lower())

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()


class StructuredDataFileTests(unittest.TestCase):
    def test_valid_structured_data_file_saves_and_prints_structured_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            data_path = Path(tmp) / "data.json"
            data_path.write_text('{"goal": "workout"}', encoding="utf-8")

            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data-file", str(data_path),
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["structured_data"], {"goal": "workout"})

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_event("evt_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.structured_data, {"goal": "workout"})

    def test_malformed_json_in_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            data_path = Path(tmp) / "data.json"
            data_path.write_text("{bad json", encoding="utf-8")

            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data-file", str(data_path),
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid JSON", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()

    def test_json_list_in_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            data_path = Path(tmp) / "data.json"
            data_path.write_text("[1, 2, 3]", encoding="utf-8")

            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data-file", str(data_path),
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation", result.stderr.lower())

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()

    def test_both_structured_data_and_structured_data_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            data_path = Path(tmp) / "data.json"
            data_path.write_text('{"goal": "workout"}', encoding="utf-8")

            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data", "{}", "--structured-data-file", str(data_path),
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--structured-data", result.stderr)
            self.assertIn("--structured-data-file", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()

    def test_missing_structured_data_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            missing_path = Path(tmp) / "does_not_exist.json"

            result = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
                "--structured-data-file", str(missing_path),
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot read", result.stderr)
            self.assertIn(missing_path.name, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_event("evt_1"))
            finally:
                repo.close()


class DuplicateEventTests(unittest.TestCase):
    def test_second_insert_with_same_event_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            first = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
            ])
            self.assertEqual(first.returncode, 0, first.stderr)

            second = _run([
                "--db", db_path, "--user-id", "usr_1", "--event-id", "evt_1",
                "--event-type", "goal_completed", "--source", "app",
            ])
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Duplicate event_id", second.stderr)
            self.assertIn("evt_1", second.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_event("evt_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)


if __name__ == "__main__":
    unittest.main()

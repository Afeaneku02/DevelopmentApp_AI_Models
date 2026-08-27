"""Tests for tools/recompute_belief.py -- the manual belief-recompute CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake CLIs. Events, observations, and belief_evidence
are seeded through those already-tested CLIs rather than by hand-writing
rows, and every persistence claim is double-checked by reading the database
back through src.storage.repository.Repository directly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_RECOMPUTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "recompute_belief.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"
_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"
_ADD_EVIDENCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_belief_evidence.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
    return result


def _run_recompute(args: list[str]) -> subprocess.CompletedProcess:
    return _run(_RECOMPUTE_SCRIPT, args)


def _seed_event(db_path: str, *, user_id: str, event_id: str, timestamp: str) -> None:
    result = _run(_ADD_EVENT_SCRIPT, [
        "--db", db_path, "--user-id", user_id, "--event-id", event_id,
        "--event-type", "goal_completed", "--source", "app", "--timestamp", timestamp,
    ])
    assert result.returncode == 0, result.stderr


def _seed_observation(db_path: str, *, user_id: str, observation_id: str, event_id: str, created_at: str) -> None:
    result = _run(_ADD_OBSERVATION_SCRIPT, [
        "--db", db_path, "--observation-id", observation_id, "--user-id", user_id,
        "--event-id", event_id, "--category", "routine", "--observation", "did a workout",
        "--importance", "0.6", "--confidence", "0.6", "--created-at", created_at,
    ])
    assert result.returncode == 0, result.stderr


def _seed_evidence(
    db_path: str, *, evidence_id: str, observation_id: str, belief_id: str, created_at: str,
) -> None:
    result = _run(_ADD_EVIDENCE_SCRIPT, [
        "--db", db_path, "--evidence-id", evidence_id, "--observation-id", observation_id,
        "--belief-id", belief_id, "--belief-type", "behavioral_tendency", "--direction", "support",
        "--source-type", "recorded_event", "--context-key", "fitness", "--strength", "0.9",
        "--model-version", "demo-0.1", "--created-at", created_at,
    ])
    assert result.returncode == 0, result.stderr


def _seed_evidence_chain(db_path: str, *, user_id: str, belief_id: str) -> None:
    """One belief backed by two evidence rows dated 5 and 3 days before
    2026-08-27T00:00:00Z, so the earliest active evidence's observed_at is
    unambiguous for the --first-observed inference tests."""
    _seed_event(db_path, user_id=user_id, event_id="evt_1", timestamp="2026-08-22T18:00:00Z")
    _seed_event(db_path, user_id=user_id, event_id="evt_2", timestamp="2026-08-24T18:00:00Z")
    _seed_observation(
        db_path, user_id=user_id, observation_id="obs_1", event_id="evt_1", created_at="2026-08-22T18:00:00Z"
    )
    _seed_observation(
        db_path, user_id=user_id, observation_id="obs_2", event_id="evt_2", created_at="2026-08-24T18:00:00Z"
    )
    _seed_evidence(
        db_path, evidence_id="bev_1", observation_id="obs_1", belief_id=belief_id,
        created_at="2026-08-22T18:00:00Z",
    )
    _seed_evidence(
        db_path, evidence_id="bev_2", observation_id="obs_2", belief_id=belief_id,
        created_at="2026-08-24T18:00:00Z",
    )


_BELIEF_ARGS = [
    "--belief-type", "behavioral_tendency", "--belief-key", "higher_adherence_after_work",
    "--belief-value-json", "true",
]


class ValidRecomputeFromEvidenceTests(unittest.TestCase):
    def test_recompute_from_stored_evidence_saves_positive_confidence_belief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--as-of", "2026-08-27T00:00:00Z",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertGreater(printed["confidence"], 0.0)
            self.assertEqual(printed["supporting_evidence_count"], 2)
            self.assertFalse(printed["locked_until_recompute"])


class ReadBackTests(unittest.TestCase):
    def test_saved_belief_can_be_read_back_from_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--as-of", "2026-08-27T00:00:00Z",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.confidence, printed["confidence"])
            self.assertEqual(stored.belief_key, "higher_adherence_after_work")


class NoActiveEvidenceRejectionTests(unittest.TestCase):
    def test_no_active_evidence_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_none", "--user-id", "usr_1", *_BELIEF_ARGS,
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No active evidence", result.stderr)
            self.assertIn("--allow-no-evidence", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_none"))
            finally:
                repo.close()


class NoActiveEvidenceAllowedTests(unittest.TestCase):
    def test_no_active_evidence_with_allow_flag_produces_zero_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_none", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["confidence"], 0.0)
            self.assertEqual(printed["total_evidence_count"], 0)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief(user_id="usr_1", belief_id="bel_none")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.confidence, 0.0)


class InvalidBeliefTypeRejectionTests(unittest.TestCase):
    def test_invalid_belief_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "not_a_real_belief_type", "--belief-key", "x",
                "--belief-value-json", "true", "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("--belief-type", result.stderr)


class InvalidBeliefValueJsonRejectionTests(unittest.TestCase):
    def test_malformed_belief_value_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json", "{bad", "--allow-no-evidence",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid JSON", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
            finally:
                repo.close()


class FirstObservedInferenceTests(unittest.TestCase):
    def test_first_observed_defaults_to_earliest_active_evidence_observed_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--as-of", "2026-08-27T00:00:00Z",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertTrue(printed["first_observed"].startswith("2026-08-22T18:00:00"))


class DeletionReasonNoEvidenceTests(unittest.TestCase):
    def test_recompute_reason_deletion_with_no_evidence_produces_status_outdated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--allow-no-evidence", "--recompute-reason", "deletion",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["confidence"], 0.0)
            self.assertEqual(printed["status"], "outdated")


class BeliefValueJsonFileTests(unittest.TestCase):
    def test_valid_object_from_file_is_saved_and_printed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            value_path = Path(tmp) / "value.json"
            value_path.write_text('{"habit": "workout", "count": 3}', encoding="utf-8")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json-file", str(value_path), "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["belief_value"], {"habit": "workout", "count": 3})

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(stored.belief_value, {"habit": "workout", "count": 3})

    def test_valid_string_from_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            value_path = Path(tmp) / "value.json"
            value_path.write_text('"just a string"', encoding="utf-8")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json-file", str(value_path), "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["belief_value"], "just a string")

    def test_valid_array_from_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            value_path = Path(tmp) / "value.json"
            value_path.write_text("[1, 2, 3]", encoding="utf-8")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json-file", str(value_path), "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["belief_value"], [1, 2, 3])

    def test_malformed_json_in_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            value_path = Path(tmp) / "value.json"
            value_path.write_text("{bad", encoding="utf-8")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json-file", str(value_path), "--allow-no-evidence",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid JSON", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
            finally:
                repo.close()

    def test_missing_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            missing_path = Path(tmp) / "does_not_exist.json"

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json-file", str(missing_path), "--allow-no-evidence",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot read", result.stderr)
            self.assertIn(missing_path.name, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
            finally:
                repo.close()

    def test_both_belief_value_json_and_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            value_path = Path(tmp) / "value.json"
            value_path.write_text("true", encoding="utf-8")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--belief-value-json", "true", "--belief-value-json-file", str(value_path),
                "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("not allowed with argument", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
            finally:
                repo.close()

    def test_neither_belief_value_json_nor_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x",
                "--allow-no-evidence",
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("--belief-value-json", result.stderr)


class OutputShapeTests(unittest.TestCase):
    def test_output_contains_confidence_status_counts_and_lock_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1")

            result = _run_recompute([
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1", *_BELIEF_ARGS,
                "--as-of", "2026-08-27T00:00:00Z",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            for expected_field in (
                "confidence", "status", "supporting_evidence_count", "contradicting_evidence_count",
                "total_evidence_count", "effective_support_count", "effective_evidence_count",
                "evidence_for", "evidence_against", "locked_until_recompute", "first_observed",
                "last_validated", "last_successful_recompute_at",
            ):
                self.assertIn(expected_field, printed)


if __name__ == "__main__":
    unittest.main()

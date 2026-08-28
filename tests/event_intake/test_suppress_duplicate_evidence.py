"""Tests for tools/suppress_duplicate_evidence.py -- the manual
duplicate-evidence suppression CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake/recompute/invalidation CLIs. Events, observations,
belief_evidence, and beliefs are seeded through those already-tested CLIs
rather than by hand-writing rows, and every persistence claim is
double-checked by reading the database back through
src.storage.repository.Repository directly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_SUPPRESS_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "suppress_duplicate_evidence.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"
_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"
_ADD_EVIDENCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_belief_evidence.py"
_RECOMPUTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "recompute_belief.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def _run_suppress(args: list[str]) -> subprocess.CompletedProcess:
    return _run(_SUPPRESS_SCRIPT, args)


def _seed_event(db_path: str, *, user_id: str, event_id: str) -> None:
    result = _run(_ADD_EVENT_SCRIPT, [
        "--db", db_path, "--user-id", user_id, "--event-id", event_id,
        "--event-type", "goal_completed", "--source", "app",
    ])
    assert result.returncode == 0, result.stderr


def _seed_observation(db_path: str, *, user_id: str, observation_id: str, event_id: str) -> None:
    result = _run(_ADD_OBSERVATION_SCRIPT, [
        "--db", db_path, "--observation-id", observation_id, "--user-id", user_id,
        "--event-id", event_id, "--category", "routine", "--observation", "did a workout",
        "--importance", "0.6", "--confidence", "0.6",
    ])
    assert result.returncode == 0, result.stderr


def _seed_evidence(
    db_path: str, *, evidence_id: str, observation_id: str, belief_id: str, direction: str = "support",
) -> None:
    result = _run(_ADD_EVIDENCE_SCRIPT, [
        "--db", db_path, "--evidence-id", evidence_id, "--observation-id", observation_id,
        "--belief-id", belief_id, "--belief-type", "behavioral_tendency", "--direction", direction,
        "--source-type", "recorded_event", "--context-key", "fitness", "--strength", "0.9",
        "--model-version", "demo-0.1",
    ])
    assert result.returncode == 0, result.stderr


def _seed_recompute(db_path: str, *, belief_id: str, user_id: str) -> None:
    result = _run(_RECOMPUTE_SCRIPT, [
        "--db", db_path, "--belief-id", belief_id, "--user-id", user_id,
        "--belief-type", "behavioral_tendency", "--belief-key", "x", "--belief-value-json", "true",
    ])
    assert result.returncode == 0, result.stderr


class ExactDuplicateSuppressedTests(unittest.TestCase):
    def test_duplicate_evidence_from_the_same_event_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_1")

            result = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual([row["evidence_id"] for row in printed["suppressed_evidence"]], ["bev_2"])

            repo = Repository.at_path(db_path)
            try:
                active = repo.list_active_evidence(user_id="usr_1", belief_id="bel_1")
                audited = repo.get_evidence("bev_2")
            finally:
                repo.close()
            self.assertEqual([row.evidence_id for row in active], ["bev_1"])
            self.assertTrue(audited.is_duplicate_suppressed)
            self.assertTrue(audited.is_active)


class DifferentBeliefNotSuppressedTests(unittest.TestCase):
    def test_same_event_different_belief_is_not_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_2")

            result = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["suppressed_evidence"], [])

            repo = Repository.at_path(db_path)
            try:
                self.assertFalse(repo.get_evidence("bev_1").is_duplicate_suppressed)
                self.assertFalse(repo.get_evidence("bev_2").is_duplicate_suppressed)
            finally:
                repo.close()


class OppositeDirectionNotSuppressedTests(unittest.TestCase):
    def test_same_belief_opposite_direction_is_not_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1", direction="support")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_1", direction="contradict")

            result = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["suppressed_evidence"], [])


class SuppressedExcludedFromConfidenceTests(unittest.TestCase):
    def test_suppressed_evidence_no_longer_counts_toward_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_1")
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            repo = Repository.at_path(db_path)
            try:
                before = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(before.supporting_evidence_count, 2)

            result = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertTrue(printed["latest_belief"]["locked_until_recompute"])

            recompute_result = _run(_RECOMPUTE_SCRIPT, [
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "x", "--belief-value-json", "true",
            ])
            self.assertEqual(recompute_result.returncode, 0, recompute_result.stderr)

            repo = Repository.at_path(db_path)
            try:
                after = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(after.supporting_evidence_count, 1)
            self.assertFalse(after.locked_until_recompute)


class IdempotentReRunTests(unittest.TestCase):
    def test_running_suppression_twice_is_a_safe_no_op_the_second_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_1")

            first = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])
            self.assertEqual(first.returncode, 0, first.stderr)
            first_printed = json.loads(first.stdout)
            self.assertEqual(len(first_printed["suppressed_evidence"]), 1)

            second = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])
            self.assertEqual(second.returncode, 0, second.stderr)
            second_printed = json.loads(second.stdout)
            self.assertEqual(second_printed["suppressed_evidence"], [])

            repo = Repository.at_path(db_path)
            try:
                all_evidence = repo.list_evidence(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            suppressed = [row.evidence_id for row in all_evidence if row.is_duplicate_suppressed]
            self.assertEqual(suppressed, ["bev_2"])


class AuditFieldsIntactTests(unittest.TestCase):
    def test_suppressed_row_keeps_its_other_fields_and_stays_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_2", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_2", belief_id="bel_1")

            repo = Repository.at_path(db_path)
            try:
                before = repo.get_evidence("bev_2")
            finally:
                repo.close()

            result = _run_suppress(["--db", db_path, "--user-id", "usr_1", "--belief-id", "bel_1"])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                after = repo.get_evidence("bev_2")
            finally:
                repo.close()

            self.assertTrue(after.is_duplicate_suppressed)
            self.assertEqual(after.suppression_reason, "duplicate_evidence")
            before_fields = before.model_dump(exclude={"is_duplicate_suppressed", "suppression_reason"})
            after_fields = after.model_dump(exclude={"is_duplicate_suppressed", "suppression_reason"})
            self.assertEqual(before_fields, after_fields)


if __name__ == "__main__":
    unittest.main()

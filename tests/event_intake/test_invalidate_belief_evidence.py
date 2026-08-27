"""Tests for tools/invalidate_belief_evidence.py -- the manual belief-
evidence invalidation CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake/recompute CLIs. Events, observations,
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

_INVALIDATE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "invalidate_belief_evidence.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"
_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"
_ADD_EVIDENCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_belief_evidence.py"
_RECOMPUTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "recompute_belief.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def _run_invalidate(args: list[str]) -> subprocess.CompletedProcess:
    return _run(_INVALIDATE_SCRIPT, args)


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


def _seed_evidence(db_path: str, *, evidence_id: str, observation_id: str, belief_id: str) -> None:
    result = _run(_ADD_EVIDENCE_SCRIPT, [
        "--db", db_path, "--evidence-id", evidence_id, "--observation-id", observation_id,
        "--belief-id", belief_id, "--belief-type", "behavioral_tendency", "--direction", "support",
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


def _seed_evidence_chain(db_path: str, *, user_id: str, belief_id: str, evidence_id: str) -> None:
    _seed_event(db_path, user_id=user_id, event_id="evt_1")
    _seed_observation(db_path, user_id=user_id, observation_id="obs_1", event_id="evt_1")
    _seed_evidence(db_path, evidence_id=evidence_id, observation_id="obs_1", belief_id=belief_id)


class ValidInvalidationTests(unittest.TestCase):
    def test_valid_invalidation_marks_evidence_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_1", "--reason", "deletion"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertFalse(printed["evidence"]["is_active"])
            self.assertEqual(printed["evidence"]["invalidation_reason"], "deletion")

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_1")
            finally:
                repo.close()
            self.assertFalse(stored.is_active)
            self.assertEqual(stored.invalidation_reason, "deletion")


class LocksLatestBeliefTests(unittest.TestCase):
    def test_invalidation_locks_the_latest_belief_when_one_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            before = Repository.at_path(db_path)
            try:
                prior_belief = before.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                before.close()
            self.assertIsNotNone(prior_belief)
            self.assertFalse(prior_belief.locked_until_recompute)

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_1", "--reason", "manual_review"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertIsNotNone(printed["latest_belief"])
            self.assertTrue(printed["latest_belief"]["locked_until_recompute"])

            repo = Repository.at_path(db_path)
            try:
                stored_belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertTrue(stored_belief.locked_until_recompute)


class NoBeliefYetTests(unittest.TestCase):
    def test_latest_belief_is_null_when_no_belief_exists_yet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_1", "--reason", "deletion"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertIsNone(printed["latest_belief"])


class UnknownEvidenceRejectionTests(unittest.TestCase):
    def test_unknown_evidence_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_ghost", "--reason", "deletion"])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bev_ghost", result.stderr)


class InvalidReasonRejectionTests(unittest.TestCase):
    def test_invalid_reason_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_1", "--reason", "bogus"])

            self.assertEqual(result.returncode, 2)
            self.assertIn("--reason", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_1")
            finally:
                repo.close()
            self.assertTrue(stored.is_active)


class InvalidatedAtRespectedTests(unittest.TestCase):
    def test_explicit_invalidated_at_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")

            result = _run_invalidate([
                "--db", db_path, "--evidence-id", "bev_1", "--reason", "deletion",
                "--invalidated-at", "2026-01-01T00:00:00Z",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertTrue(printed["evidence"]["invalidated_at"].startswith("2026-01-01T00:00:00"))


class NoRecomputePerformedTests(unittest.TestCase):
    def test_confidence_stays_stale_but_belief_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_evidence_chain(db_path, user_id="usr_1", belief_id="bel_1", evidence_id="bev_1")
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            before = Repository.at_path(db_path)
            try:
                prior_belief = before.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                before.close()
            prior_confidence = prior_belief.confidence
            self.assertGreater(prior_confidence, 0.0)

            result = _run_invalidate(["--db", db_path, "--evidence-id", "bev_1", "--reason", "deletion"])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored_belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            # No recompute happened -- the stale confidence value is left
            # exactly as it was; only the lock flag communicates it is no
            # longer authoritative.
            self.assertEqual(stored_belief.confidence, prior_confidence)
            self.assertTrue(stored_belief.locked_until_recompute)


if __name__ == "__main__":
    unittest.main()

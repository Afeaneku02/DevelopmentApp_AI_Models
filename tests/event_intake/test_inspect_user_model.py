"""Tests for tools/inspect_user_model.py -- the read-only inspection CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake/recompute/invalidation CLIs. Records are seeded
through those already-tested CLIs rather than by hand-writing rows. Every
"the inspector didn't change anything" claim is double-checked by comparing
a full Repository read-back (events, observations, observation_events,
evidence, beliefs) taken before and after running the inspector.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_INSPECT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "inspect_user_model.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"
_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"
_ADD_EVIDENCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_belief_evidence.py"
_RECOMPUTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "recompute_belief.py"
_INVALIDATE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "invalidate_belief_evidence.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def _run_inspect(args: list[str]) -> subprocess.CompletedProcess:
    return _run(_INSPECT_SCRIPT, args)


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


def _invalidate(db_path: str, *, evidence_id: str) -> None:
    result = _run(_INVALIDATE_SCRIPT, ["--db", db_path, "--evidence-id", evidence_id, "--reason", "deletion"])
    assert result.returncode == 0, result.stderr


def _seed_full_chain(db_path: str, *, user_id: str, belief_id: str, event_id: str, observation_id: str,
                      evidence_id: str) -> None:
    _seed_event(db_path, user_id=user_id, event_id=event_id)
    _seed_observation(db_path, user_id=user_id, observation_id=observation_id, event_id=event_id)
    _seed_evidence(db_path, evidence_id=evidence_id, observation_id=observation_id, belief_id=belief_id)


def _dump_all(db_path: str) -> dict:
    repo = Repository.at_path(db_path)
    try:
        return {
            "events": [e.model_dump_json() for e in repo.list_events()],
            "observations": [o.model_dump_json() for o in repo.list_observations()],
            "evidence": [ev.model_dump_json() for ev in repo.list_all_evidence()],
            "beliefs": [b.model_dump_json() for b in repo.list_latest_beliefs()],
        }
    finally:
        repo.close()


class MissingDbRejectionTests(unittest.TestCase):
    def test_inspecting_a_missing_db_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "does_not_exist.sqlite3"

            result = _run_inspect(["--db", str(db_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no such database file", result.stderr)
            self.assertIn(db_path.name, result.stderr)

    def test_missing_db_file_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "does_not_exist.sqlite3"

            result = _run_inspect(["--db", str(db_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(db_path.exists())


class EmptyDbOutputShapeTests(unittest.TestCase):
    def test_empty_db_returns_all_empty_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            Repository.at_path(db_path).close()  # creates an empty, schema-only database

            result = _run_inspect(["--db", db_path])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(
                printed, {"events": [], "observations": [], "observation_events": [], "evidence": [], "beliefs": []}
            )


class FullPipelineAppearsTests(unittest.TestCase):
    def test_full_pipeline_records_appear_after_using_existing_clis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            result = _run_inspect(["--db", db_path])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual([e["event_id"] for e in printed["events"]], ["evt_1"])
            self.assertEqual([o["observation_id"] for o in printed["observations"]], ["obs_1"])
            self.assertEqual(len(printed["observation_events"]), 1)
            self.assertEqual([ev["evidence_id"] for ev in printed["evidence"]], ["bev_1"])
            self.assertEqual([b["belief_id"] for b in printed["beliefs"]], ["bel_1"])


class UserIdFilterTests(unittest.TestCase):
    def test_user_id_filter_excludes_other_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_event(db_path, user_id="usr_2", event_id="evt_2")

            result = _run_inspect(["--db", db_path, "--user-id", "usr_2"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual([e["event_id"] for e in printed["events"]], ["evt_2"])
            self.assertEqual(printed["observations"], [])
            self.assertEqual(printed["evidence"], [])


class BeliefIdFilterTests(unittest.TestCase):
    def test_belief_id_filter_excludes_other_beliefs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_id="evt_1")
            _seed_evidence(db_path, evidence_id="bev_1", observation_id="obs_1", belief_id="bel_1")
            _seed_evidence(db_path, evidence_id="bev_2", observation_id="obs_1", belief_id="bel_2")
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")
            _seed_recompute(db_path, belief_id="bel_2", user_id="usr_1")

            result = _run_inspect(["--db", db_path, "--belief-id", "bel_1"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual([ev["evidence_id"] for ev in printed["evidence"]], ["bev_1"])
            self.assertEqual([b["belief_id"] for b in printed["beliefs"]], ["bel_1"])
            # events/observations are not belief-scoped, so --belief-id must
            # not filter them out.
            self.assertEqual(len(printed["events"]), 1)
            self.assertEqual(len(printed["observations"]), 1)


class InactiveEvidenceHiddenByDefaultTests(unittest.TestCase):
    def test_inactive_evidence_hidden_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _invalidate(db_path, evidence_id="bev_1")

            result = _run_inspect(["--db", db_path])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["evidence"], [])


class InactiveEvidenceIncludedWithFlagTests(unittest.TestCase):
    def test_inactive_evidence_included_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _invalidate(db_path, evidence_id="bev_1")

            result = _run_inspect(["--db", db_path, "--include-inactive-evidence"])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(len(printed["evidence"]), 1)
            self.assertFalse(printed["evidence"][0]["is_active"])
            self.assertEqual(printed["evidence"][0]["invalidation_reason"], "deletion")


class LatestBeliefShownTests(unittest.TestCase):
    def test_latest_belief_shown_after_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            repo = Repository.at_path(db_path)
            try:
                expected = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()

            result = _run_inspect(["--db", db_path])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(len(printed["beliefs"]), 1)
            self.assertEqual(printed["beliefs"][0]["belief_id"], "bel_1")
            self.assertEqual(printed["beliefs"][0]["confidence"], expected.confidence)
            self.assertFalse(printed["beliefs"][0]["locked_until_recompute"])


class InspectorDoesNotMutateTests(unittest.TestCase):
    def test_inspector_does_not_mutate_db_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")

            before = _dump_all(db_path)

            result = _run_inspect(["--db", db_path, "--include-inactive-evidence", "--pretty"])
            self.assertEqual(result.returncode, 0, result.stderr)

            after = _dump_all(db_path)
            self.assertEqual(before, after)


class PrettyFlagTests(unittest.TestCase):
    def test_pretty_flag_produces_multiline_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            Repository.at_path(db_path).close()  # creates an empty, schema-only database

            compact = _run_inspect(["--db", db_path])
            pretty = _run_inspect(["--db", db_path, "--pretty"])

            self.assertEqual(compact.returncode, 0, compact.stderr)
            self.assertEqual(pretty.returncode, 0, pretty.stderr)
            self.assertNotIn("\n", compact.stdout.strip())
            self.assertIn("\n", pretty.stdout.strip())
            self.assertEqual(json.loads(compact.stdout), json.loads(pretty.stdout))


if __name__ == "__main__":
    unittest.main()

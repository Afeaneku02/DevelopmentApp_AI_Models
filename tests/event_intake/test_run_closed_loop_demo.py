"""Tests for tools/run_closed_loop_demo.py -- the one-command end-to-end demo.

Proves the demo builds the whole chain (event -> ... -> promoted evidence ->
recomputed belief), is deterministic and re-run safe, and that the viewer and
inspector can then show every section populated.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.common.enums import SourceType
from src.storage.repository import Repository
from src.viewer.user_model_view import collect_view_model, render_html

import tools.run_closed_loop_demo as demo

_DEMO_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "run_closed_loop_demo.py"
_INSPECT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "inspect_user_model.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


class BuildsTheFullChainTests(unittest.TestCase):
    def test_build_closed_loop_creates_every_record_and_moves_confidence(self) -> None:
        repo = Repository.in_memory()
        try:
            summary = demo.build_closed_loop(
                repo, run_id="t", as_of=datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
            )
            user_id = summary["user_id"]
            belief_id = summary["belief_id"]

            self.assertEqual(len(repo.list_events(user_id=user_id)), 1)
            self.assertEqual(len(repo.list_observations(user_id=user_id)), 1)
            self.assertEqual(len(repo.list_observation_events_for([summary["observation_id"]])), 1)
            self.assertEqual(len(repo.list_belief_key_canonicalizations(user_id=user_id)), 1)

            evidence = repo.list_evidence(user_id=user_id, belief_id=belief_id)
            source_types = sorted(e.source_type for e in evidence)
            self.assertEqual(
                source_types, [SourceType.RECORDED_EVENT, SourceType.REPEATED_PATTERN_SUMMARY]
            )

            self.assertEqual(len(repo.list_recommendations(user_id=user_id)), 4)
            outcomes = repo.list_recommendation_outcomes()
            self.assertEqual(len(outcomes), 4)
            signals = repo.list_outcome_learning_signals(user_id=user_id)
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].kind.value, "support")

            # promoted evidence exists and carries the signal's independence group
            promoted = [e for e in evidence if e.source_type is SourceType.REPEATED_PATTERN_SUMMARY]
            self.assertEqual(promoted[0].independence_group, signals[0].independence_group)
            self.assertEqual(promoted[0].aggregation_authorized_by, "outcome_learning_promotion")

            # the belief was recomputed: a second belief row, higher confidence, not locked
            belief_rows = repo.list_latest_beliefs(user_id=user_id, belief_id=belief_id)
            latest = repo.get_latest_belief(user_id=user_id, belief_id=belief_id)
            self.assertFalse(latest.locked_until_recompute)
            self.assertEqual(summary["initial_confidence"], 0.13711449558366265)
            self.assertGreater(summary["final_confidence"], summary["initial_confidence"])
            self.assertEqual(latest.confidence, summary["final_confidence"])
        finally:
            repo.close()

    def test_cli_runs_and_prints_a_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            result = _run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "demo1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["run_id"], "demo1")
            self.assertEqual(summary["signal_kind"], "support")
            self.assertEqual(len(summary["promoted_evidence_ids"]), 1)
            self.assertGreater(summary["final_confidence"], summary["initial_confidence"])
            # the printed next-steps reference the real user id
            self.assertIn(f"--user-id {summary['user_id']}", result.stderr)


class DeterministicAndReRunSafeTests(unittest.TestCase):
    def test_same_run_id_into_two_fresh_databases_is_byte_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            first = _run(_DEMO_SCRIPT, ["--db", str(Path(tmp) / "a.sqlite3"), "--run-id", "demoX"])
            second = _run(_DEMO_SCRIPT, ["--db", str(Path(tmp) / "b.sqlite3"), "--run-id", "demoX"])
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))

    def test_re_running_the_same_run_id_fails_clearly_and_changes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self.assertEqual(_run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "d"]).returncode, 0)

            repo = Repository.at_path(db_path)
            try:
                before = repo.list_events(user_id="usr_d")
            finally:
                repo.close()

            again = _run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "d"])
            self.assertEqual(again.returncode, 1)
            self.assertIn("already has data", again.stderr)
            self.assertNotIn("Traceback", again.stderr)

            repo = Repository.at_path(db_path)
            try:
                after = repo.list_events(user_id="usr_d")
            finally:
                repo.close()
            self.assertEqual(
                [e.model_dump_json() for e in before], [e.model_dump_json() for e in after]
            )

    def test_a_second_run_id_adds_an_independent_chain_to_the_same_db(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self.assertEqual(_run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "one"]).returncode, 0)
            self.assertEqual(_run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "two"]).returncode, 0)

            repo = Repository.at_path(db_path)
            try:
                self.assertEqual(len(repo.list_events(user_id="usr_one")), 1)
                self.assertEqual(len(repo.list_events(user_id="usr_two")), 1)
                self.assertEqual(len(repo.list_outcome_learning_signals()), 2)
            finally:
                repo.close()


class ViewerAndInspectorShowEverySectionTests(unittest.TestCase):
    def _demo_db(self, tmp: str) -> str:
        db_path = str(Path(tmp) / "canonical.sqlite3")
        result = _run(_DEMO_SCRIPT, ["--db", db_path, "--run-id", "d"])
        assert result.returncode == 0, result.stderr
        return db_path

    def test_viewer_renders_every_section_populated(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = self._demo_db(tmp)
            repo = Repository.readonly_at_path(db_path)
            try:
                view_model = collect_view_model(repo, db_path=db_path, user_id="usr_d")
            finally:
                repo.close()

        for section in (
            "events", "observations", "observation_events", "evidence", "beliefs",
            "canonicalizations", "recommendations", "recommendation_outcomes",
            "outcome_learning_signals",
        ):
            self.assertTrue(getattr(view_model, section), f"{section} is empty")
        self.assertEqual(view_model.summary()["outcome_learning_signals_promoted"], 1)

        page = render_html(view_model)
        for heading in (
            "Events", "Observations", "Observation-event links", "Evidence", "Beliefs",
            "Belief-key canonicalization decisions", "Recommendations", "Recommendation outcomes",
            "Outcome learning signals",
        ):
            self.assertIn(heading, page)
        # no section rendered its empty-state placeholder
        self.assertNotIn("No events stored.", page)
        self.assertNotIn("No outcome-learning signals stored.", page)
        self.assertNotIn("No recommendations stored.", page)

    def test_inspector_json_has_every_section_populated(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = self._demo_db(tmp)
            result = _run(_INSPECT_SCRIPT, ["--db", db_path, "--user-id", "usr_d"])
            self.assertEqual(result.returncode, 0, result.stderr)
            dump = json.loads(result.stdout)

        for key in (
            "events", "observations", "observation_events", "evidence", "beliefs",
            "canonicalizations", "recommendations", "recommendation_outcomes",
            "outcome_learning_signals",
        ):
            self.assertTrue(dump[key], f"{key} is empty")
        self.assertTrue(dump["outcome_learning_signals"][0]["promoted"])


if __name__ == "__main__":
    unittest.main()

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
_MAKE_REC_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "make_recommendation.py"
_ADD_OUTCOME_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_recommendation_outcome.py"
_LEARN_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "learn_from_recommendation_outcomes.py"
_PROMOTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "promote_outcome_learning_signal.py"


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
                printed,
                {
                    "events": [], "observations": [], "observation_events": [], "evidence": [],
                    "beliefs": [], "canonicalizations": [], "recommendations": [],
                    "recommendation_outcomes": [], "outcome_learning_signals": [],
                },
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


def _seed_recommendation(db_path: str, *, recommendation_id: str, user_id: str, context_key: str) -> None:
    result = _run(_MAKE_REC_SCRIPT, [
        "--db", db_path, "--user-id", user_id, "--context-key", context_key,
        "--recommendation-id", recommendation_id,
    ])
    assert result.returncode == 0, result.stderr


def _seed_outcome(db_path: str, *, outcome_id: str, recommendation_id: str) -> None:
    result = _run(_ADD_OUTCOME_SCRIPT, [
        "--db", db_path, "--outcome-id", outcome_id, "--recommendation-id", recommendation_id,
        "--followed", "followed", "--result", "successful", "--source", "app_event",
        "--user-feedback", "did the after-work slot",
    ])
    assert result.returncode == 0, result.stderr


def _seed_learning_signal(db_path: str, *, user_id: str, belief_id: str, event_prefix: str) -> str:
    """Seed a full chain, a belief, then 4 recommendations + 4 successful
    outcomes for one context, persist the resulting outcome-learning signal
    via the learn CLI, and return its signal_id."""
    _seed_full_chain(
        db_path, user_id=user_id, belief_id=belief_id, event_id=f"{event_prefix}_evt",
        observation_id=f"{event_prefix}_obs", evidence_id=f"{event_prefix}_bev",
    )
    _seed_recompute(db_path, belief_id=belief_id, user_id=user_id)
    for index in range(1, 5):
        rec_id = f"{event_prefix}_rec_{index}"
        _seed_recommendation(
            db_path, recommendation_id=rec_id, user_id=user_id, context_key="fitness_scheduling"
        )
        _seed_outcome(db_path, outcome_id=f"{event_prefix}_out_{index}", recommendation_id=rec_id)
    result = _run(_LEARN_SCRIPT, ["--db", db_path, "--persist"])
    assert result.returncode == 0, result.stderr

    repo = Repository.at_path(db_path)
    try:
        signals = repo.list_outcome_learning_signals(user_id=user_id)
    finally:
        repo.close()
    assert signals, "learn CLI produced no signal"
    return signals[-1].signal_id


def _promote_signal(db_path: str, *, signal_id: str) -> None:
    result = _run(_PROMOTE_SCRIPT, ["--db", db_path, "--signal-id", signal_id, "--persist"])
    assert result.returncode == 0, result.stderr


class RecommendationsAndOutcomesAppearTests(unittest.TestCase):
    def test_inspector_includes_persisted_recommendations_and_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")
            _seed_recommendation(
                db_path, recommendation_id="rec_1", user_id="usr_1", context_key="fitness_scheduling"
            )
            _seed_outcome(db_path, outcome_id="out_1", recommendation_id="rec_1")

            result = _run_inspect(["--db", db_path])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)

            self.assertEqual([r["recommendation_id"] for r in printed["recommendations"]], ["rec_1"])
            self.assertEqual(printed["recommendations"][0]["user_id"], "usr_1")
            self.assertEqual(
                printed["recommendations"][0]["recommendation_context"], "fitness_scheduling"
            )
            self.assertIn("risk_tier", printed["recommendations"][0])
            self.assertEqual(
                [o["outcome_id"] for o in printed["recommendation_outcomes"]], ["out_1"]
            )
            self.assertEqual(printed["recommendation_outcomes"][0]["recommendation_id"], "rec_1")
            self.assertEqual(printed["recommendation_outcomes"][0]["followed"], "followed")

    def test_inspecting_does_not_mutate_the_frozen_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_full_chain(
                db_path, user_id="usr_1", belief_id="bel_1", event_id="evt_1",
                observation_id="obs_1", evidence_id="bev_1",
            )
            _seed_recompute(db_path, belief_id="bel_1", user_id="usr_1")
            _seed_recommendation(
                db_path, recommendation_id="rec_1", user_id="usr_1", context_key="fitness_scheduling"
            )
            _seed_outcome(db_path, outcome_id="out_1", recommendation_id="rec_1")

            repo = Repository.at_path(db_path)
            try:
                before = repo.get_recommendation("rec_1").model_dump_json()
            finally:
                repo.close()

            self.assertEqual(_run_inspect(["--db", db_path, "--pretty"]).returncode, 0)

            repo = Repository.at_path(db_path)
            try:
                after = repo.get_recommendation("rec_1").model_dump_json()
                outcomes = repo.list_recommendation_outcomes(recommendation_id="rec_1")
            finally:
                repo.close()
            self.assertEqual(before, after)
            self.assertEqual([o.outcome_id for o in outcomes], ["out_1"])


class OutcomeLearningSignalsAppearTests(unittest.TestCase):
    def test_inspector_includes_persisted_outcome_learning_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_learning_signal(db_path, user_id="usr_1", belief_id="bel_1", event_prefix="a")

            result = _run_inspect(["--db", db_path])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)

            self.assertEqual(len(printed["outcome_learning_signals"]), 1)
            signal = printed["outcome_learning_signals"][0]
            self.assertEqual(signal["user_id"], "usr_1")
            self.assertEqual(signal["recommendation_context"], "fitness_scheduling")
            self.assertEqual(signal["kind"], "support")
            self.assertEqual(signal["trial_count"], 4)
            self.assertFalse(signal["causal_claim"])
            self.assertEqual(signal["belief_ids"], ["bel_1"])
            self.assertEqual(len(signal["proposed_evidence"]), 1)
            self.assertFalse(signal["promoted"])
            self.assertEqual(signal["promoted_evidence_ids"], [])

    def test_a_promoted_signal_is_shown_as_promoted_with_its_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = _seed_learning_signal(
                db_path, user_id="usr_1", belief_id="bel_1", event_prefix="a"
            )
            _promote_signal(db_path, signal_id=signal_id)

            printed = json.loads(_run_inspect(["--db", db_path]).stdout)
            signal = printed["outcome_learning_signals"][0]
            self.assertTrue(signal["promoted"])
            self.assertEqual(signal["promoted_evidence_ids"], [f"bev-ols-{signal_id}-bel_1"])
            # the promoted belief_evidence row is itself in the dump
            promoted_evidence = [
                e for e in printed["evidence"]
                if e["source_type"] == "repeated_pattern_summary"
            ]
            self.assertEqual(len(promoted_evidence), 1)
            self.assertEqual(promoted_evidence[0]["independence_group"], signal["independence_group"])

    def test_user_scoped_inspection_does_not_leak_another_users_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_learning_signal(db_path, user_id="usr_1", belief_id="bel_1", event_prefix="a")
            _seed_learning_signal(db_path, user_id="usr_2", belief_id="bel_2", event_prefix="b")

            everyone = json.loads(_run_inspect(["--db", db_path]).stdout)
            self.assertEqual(
                sorted(s["user_id"] for s in everyone["outcome_learning_signals"]),
                ["usr_1", "usr_2"],
            )

            scoped = json.loads(_run_inspect(["--db", db_path, "--user-id", "usr_1"]).stdout)
            self.assertEqual(
                [s["user_id"] for s in scoped["outcome_learning_signals"]], ["usr_1"]
            )
            serialized = json.dumps(scoped["outcome_learning_signals"])
            self.assertNotIn("usr_2", serialized)
            self.assertNotIn("bel_2", serialized)

    def test_inspecting_does_not_mutate_the_stored_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_learning_signal(db_path, user_id="usr_1", belief_id="bel_1", event_prefix="a")

            repo = Repository.at_path(db_path)
            try:
                before = [s.model_dump_json() for s in repo.list_outcome_learning_signals()]
            finally:
                repo.close()

            self.assertEqual(_run_inspect(["--db", db_path]).returncode, 0)

            repo = Repository.at_path(db_path)
            try:
                after = [s.model_dump_json() for s in repo.list_outcome_learning_signals()]
            finally:
                repo.close()
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

"""Tests for tools/run_manifest.py -- the batch import / demo-manifest CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake/recompute/invalidation CLIs. This runner is pure
orchestration over those already-tested CLIs' own main() functions, so these
tests focus on manifest parsing/validation, the snake_case-key -> CLI-flag
conversion rules, stop-on-first-failure, and that the shipped demo manifest
reproduces tools/demo_user_model.py's own scenario end to end.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_RUN_MANIFEST_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "run_manifest.py"
_DEMO_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "manifests" / "demo_after_work_workout.json"
)
_CANONICALIZED_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "manifests" / "canonicalized_after_work_workout.json"
)


def _run_manifest(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_RUN_MANIFEST_SCRIPT), *args], capture_output=True, text=True)


def _write_manifest(tmp: str, steps: list[dict]) -> str:
    manifest_path = Path(tmp) / "manifest.json"
    manifest_path.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    return str(manifest_path)


class ValidMultiStepRunTests(unittest.TestCase):
    def test_all_steps_run_in_order_and_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_1",
                    "event_type": "goal_completed", "source": "app",
                },
                {
                    "type": "add_user_observation", "observation_id": "obs_1", "user_id": "usr_1",
                    "event_id": "evt_1", "category": "routine", "observation": "did a workout",
                    "importance": 0.6, "confidence": 0.6,
                },
                {
                    "type": "add_belief_evidence", "evidence_id": "bev_1", "observation_id": "obs_1",
                    "belief_id": "bel_1", "belief_type": "behavioral_tendency", "direction": "support",
                    "source_type": "recorded_event", "context_key": "fitness", "strength": 0.9,
                    "model_version": "demo-0.1",
                },
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency", "belief_key": "x", "belief_value_json": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("All 4 step(s) completed successfully.", result.stdout)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNotNone(repo.get_event("evt_1"))
                self.assertIsNotNone(repo.get_observation("obs_1"))
                self.assertIsNotNone(repo.get_evidence("bev_1"))
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertGreater(belief.confidence, 0.0)


class StopOnFirstFailureTests(unittest.TestCase):
    def test_execution_stops_at_the_first_failing_step_and_earlier_steps_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_1",
                    "event_type": "goal_completed", "source": "app",
                },
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_ghost",
                    "event_type": "goal_completed",  # missing required --source: this step fails
                },
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_never_reached",
                    "event_type": "goal_completed", "source": "app",
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Manifest stopped at step 2", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNotNone(repo.get_event("evt_1"))
                self.assertIsNone(repo.get_event("evt_ghost"))
                self.assertIsNone(repo.get_event("evt_never_reached"))
            finally:
                repo.close()


class UnknownStepTypeRejectionTests(unittest.TestCase):
    def test_unknown_step_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [{"type": "not_a_real_step_type"}])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown step type", result.stderr)
            self.assertIn("not_a_real_step_type", result.stderr)


class ForbiddenDbKeyTests(unittest.TestCase):
    def test_step_setting_db_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "db": "sneaky.sqlite3", "user_id": "usr_1",
                    "event_id": "evt_1", "event_type": "goal_completed", "source": "app",
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must not set "db"', result.stderr)
            self.assertFalse(Path(tmp, "sneaky.sqlite3").exists())


class ManifestStructureValidationTests(unittest.TestCase):
    def test_manifest_without_steps_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps({"not_steps": []}), encoding="utf-8")

            result = _run_manifest(["--db", str(Path(tmp) / "events.sqlite3"), "--manifest", str(manifest_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must be a JSON object with a "steps" array', result.stderr)

    def test_empty_steps_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write_manifest(tmp, [])

            result = _run_manifest(["--db", str(Path(tmp) / "events.sqlite3"), "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has no steps", result.stderr)

    def test_non_object_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps({"steps": ["not-an-object"]}), encoding="utf-8")

            result = _run_manifest(["--db", str(Path(tmp) / "events.sqlite3"), "--manifest", str(manifest_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be a JSON object", result.stderr)

    def test_malformed_manifest_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("{bad json", encoding="utf-8")

            result = _run_manifest(["--db", str(Path(tmp) / "events.sqlite3"), "--manifest", str(manifest_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid JSON", result.stderr)

    def test_missing_manifest_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "does_not_exist.json"

            result = _run_manifest(["--db", str(Path(tmp) / "events.sqlite3"), "--manifest", str(missing_path)])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot read", result.stderr)


class StepValueConversionTests(unittest.TestCase):
    def test_list_value_repeats_the_flag_for_a_multi_event_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_1",
                    "event_type": "goal_completed", "source": "app",
                },
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_2",
                    "event_type": "goal_completed", "source": "app",
                },
                {
                    "type": "add_user_observation", "observation_id": "obs_1", "user_id": "usr_1",
                    "event_id": ["evt_1", "evt_2"], "primary_event_id": "evt_2",
                    "category": "routine", "observation": "did a workout",
                    "importance": 0.6, "confidence": 0.6,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                links = repo.list_observation_events("obs_1")
            finally:
                repo.close()
            roles_by_event = {link.event_id: link.link_role.value for link in links}
            self.assertEqual(roles_by_event, {"evt_1": "supporting", "evt_2": "primary"})

    def test_boolean_true_sets_a_store_true_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency", "belief_key": "x", "belief_value_json": True,
                    "allow_no_evidence": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertEqual(belief.confidence, 0.0)

    def test_boolean_false_omits_the_flag(self) -> None:
        # allow_no_evidence: false must behave exactly like omitting it --
        # i.e. the CLI's own default rejection of zero active evidence still
        # applies.
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency", "belief_key": "x", "belief_value_json": True,
                    "allow_no_evidence": False,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No active evidence", result.stderr)

    def test_structured_data_dict_value_is_json_encoded_for_the_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_1",
                    "event_type": "goal_completed", "source": "app",
                    "structured_data": {"goal": "workout", "count": 3},
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                event = repo.get_event("evt_1")
            finally:
                repo.close()
            self.assertEqual(event.structured_data, {"goal": "workout", "count": 3})

    def test_null_value_omits_the_flag_so_the_step_clis_own_default_applies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_1",
                    "event_type": "goal_completed", "source": "app", "timestamp": None,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                event = repo.get_event("evt_1")
            finally:
                repo.close()
            self.assertIsNotNone(event.timestamp)


class DemoManifestEndToEndTests(unittest.TestCase):
    def test_shipped_demo_manifest_reproduces_the_confidence_and_lock_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_manifest(["--db", db_path, "--manifest", str(_DEMO_MANIFEST_PATH)])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("All 14 step(s) completed successfully.", result.stdout)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_17", belief_id="bel_2001")
                active = repo.list_active_evidence(user_id="usr_17", belief_id="bel_2001")
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertEqual(belief.confidence, 0.0)
            self.assertEqual(belief.status.value, "outdated")
            self.assertFalse(belief.locked_until_recompute)
            self.assertEqual(active, [])


def _evidence_prefix_steps(*, user_id: str, belief_id: str) -> list[dict]:
    """The minimal add_user_event -> add_user_observation -> add_belief_evidence
    chain that gives a belief one active evidence row, so a later
    recompute_belief step has something to score."""
    return [
        {
            "type": "add_user_event", "user_id": user_id, "event_id": "evt_c1",
            "event_type": "goal_completed", "source": "app",
        },
        {
            "type": "add_user_observation", "observation_id": "obs_c1", "user_id": user_id,
            "event_id": "evt_c1", "category": "routine", "observation": "did an after-work workout",
            "importance": 0.6, "confidence": 0.6,
        },
        {
            "type": "add_belief_evidence", "evidence_id": "bev_c1", "observation_id": "obs_c1",
            "belief_id": belief_id, "belief_type": "behavioral_tendency", "direction": "support",
            "source_type": "recorded_event", "context_key": "fitness", "strength": 0.9,
            "model_version": "test-0.1",
        },
    ]


class ResolveBeliefKeyStepTests(unittest.TestCase):
    """resolve_belief_key wired into run_manifest.py as orchestration only:
    the step runs the already-tested CLI, its authorized canonical_key can be
    referenced by a later step, and a raw proposed key is never silently used
    when a canonical one was decided."""

    def test_resolve_belief_key_step_runs_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency",
                    "proposed_key": "prefers_evening_exercise_sessions",
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief_key_canonicalization(
                    user_id="usr_1", belief_type="behavioral_tendency",
                    proposed_key="prefers_evening_exercise_sessions",
                )
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.canonicalization_id, "can_x1")
            self.assertEqual(stored.canonical_key, "higher_adherence_after_work")
            self.assertEqual(stored.decision.value, "alias")

    def test_a_later_step_references_the_authorized_canonical_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                *_evidence_prefix_steps(user_id="usr_1", belief_id="bel_1"),
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency",
                    "proposed_key": "prefers_evening_exercise_sessions",
                },
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency",
                    "belief_key": "$steps.can_1.canonical_key", "belief_value_json": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertEqual(belief.belief_key, "higher_adherence_after_work")

    def test_manifest_does_not_silently_use_the_proposed_key_when_a_canonical_key_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                *_evidence_prefix_steps(user_id="usr_1", belief_id="bel_1"),
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency",
                    "proposed_key": "prefers_evening_exercise_sessions",
                },
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency",
                    "belief_key": "$steps.can_1.canonical_key", "belief_value_json": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
                # No belief was ever stored under the raw proposed key.
                under_proposed = repo.list_latest_beliefs(user_id="usr_1")
            finally:
                repo.close()
            self.assertNotEqual(belief.belief_key, "prefers_evening_exercise_sessions")
            self.assertEqual(
                [b.belief_key for b in under_proposed], ["higher_adherence_after_work"],
            )

    def test_unknown_key_keeps_separate_and_passes_its_own_key_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                *_evidence_prefix_steps(user_id="usr_1", belief_id="bel_1"),
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency",
                    "proposed_key": "a_totally_novel_key",
                },
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency",
                    "belief_key": "$steps.can_1.canonical_key", "belief_value_json": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])
            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief_key_canonicalization(
                    user_id="usr_1", belief_type="behavioral_tendency", proposed_key="a_totally_novel_key",
                )
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(stored.decision.value, "keep_separate")
            self.assertEqual(stored.canonical_key, "a_totally_novel_key")
            self.assertEqual(belief.belief_key, "a_totally_novel_key")

    def test_reference_to_an_unknown_step_id_fails_clearly_and_stops_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                *_evidence_prefix_steps(user_id="usr_1", belief_id="bel_1"),
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency",
                    "belief_key": "$steps.can_typo.canonical_key", "belief_value_json": True,
                },
                {
                    "type": "add_user_event", "user_id": "usr_1", "event_id": "evt_never",
                    "event_type": "goal_completed", "source": "app",
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("$steps.can_typo.canonical_key", result.stderr)
            self.assertIn("Manifest stopped at step 4", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
                self.assertIsNone(repo.get_event("evt_never"))
            finally:
                repo.close()

    def test_reference_to_a_missing_field_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency",
                    "proposed_key": "prefers_evening_exercise_sessions",
                },
                {
                    "type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_1",
                    "belief_type": "behavioral_tendency",
                    "belief_key": "$steps.can_1.not_a_field", "belief_value_json": True,
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no field 'not_a_field'", result.stderr)
            self.assertIn("Manifest stopped at step 2", result.stderr)

    def test_duplicate_step_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            manifest_path = _write_manifest(tmp, [
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x1",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency", "proposed_key": "key_a",
                },
                {
                    "type": "resolve_belief_key", "id": "can_1", "canonicalization_id": "can_x2",
                    "user_id": "usr_1", "belief_type": "behavioral_tendency", "proposed_key": "key_b",
                },
            ])

            result = _run_manifest(["--db", db_path, "--manifest", manifest_path])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate step id", result.stderr)


class CanonicalizedManifestEndToEndTests(unittest.TestCase):
    def test_shipped_canonicalized_manifest_stores_the_canonical_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_manifest(["--db", db_path, "--manifest", str(_CANONICALIZED_MANIFEST_PATH)])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("All 5 step(s) completed successfully.", result.stdout)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_31", belief_id="bel_3001")
                canonicalization = repo.get_latest_belief_key_canonicalization(
                    user_id="usr_31", belief_type="behavioral_tendency",
                    proposed_key="prefers_evening_exercise_sessions",
                )
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertEqual(belief.belief_key, "higher_adherence_after_work")
            self.assertEqual(canonicalization.canonical_key, "higher_adherence_after_work")
            self.assertEqual(canonicalization.decision.value, "alias")


if __name__ == "__main__":
    unittest.main()

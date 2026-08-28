"""Tests for tools/add_belief_evidence.py -- the manual belief-evidence-intake
CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for tools/add_user_event.py and tools/add_user_observation.py.
Referenced events/observations are seeded through those already-tested
intake CLIs rather than by hand-writing rows. Two rejection scenarios --
"observation has no observation_events" and "a linked event is missing" --
cannot arise through Repository's own public API (insert_observation()
already refuses to save an observation with no primary link, and
insert_evidence()/insert_observation() both check every referenced event
exists), so those two tests reach into Repository._conn directly to write a
row Repository's own API would never have allowed, the same way
tests/storage/test_repository.py does for its own edge-case setup -- this is
deliberately simulating data that reached the database some other way, not
exercising a bug in the intake CLIs it seeded from.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.storage.repository import Repository

_ADD_EVIDENCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_belief_evidence.py"
_ADD_EVENT_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_event.py"
_ADD_OBSERVATION_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "add_user_observation.py"

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

_VALID_ARGS = [
    "--belief-id", "bel_1", "--belief-type", "behavioral_tendency", "--direction", "support",
    "--source-type", "recorded_event", "--context-key", "fitness", "--strength", "0.9",
    "--model-version", "demo-0.1",
]


def _run_evidence(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ADD_EVIDENCE_SCRIPT), *args], capture_output=True, text=True,
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


def _seed_observation(
    db_path: str, *, user_id: str, observation_id: str, event_ids: list[str],
    primary_event_id: str | None = None,
) -> None:
    args = [
        sys.executable, str(_ADD_OBSERVATION_SCRIPT), "--db", db_path, "--observation-id", observation_id,
        "--user-id", user_id, "--category", "routine", "--observation", "did a workout",
        "--importance", "0.6", "--confidence", "0.6",
    ]
    for event_id in event_ids:
        args += ["--event-id", event_id]
    if primary_event_id is not None:
        args += ["--primary-event-id", primary_event_id]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _seed_evidence(db_path: str, *, evidence_id: str, observation_id: str, belief_id: str) -> None:
    result = subprocess.run(
        [
            sys.executable, str(_ADD_EVIDENCE_SCRIPT), "--db", db_path, "--evidence-id", evidence_id,
            "--observation-id", observation_id, "--belief-id", belief_id,
            "--belief-type", "behavioral_tendency", "--direction", "support",
            "--source-type", "recorded_event", "--context-key", "fitness", "--strength", "0.9",
            "--model-version", "demo-0.1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


class ValidInsertTests(unittest.TestCase):
    def test_valid_evidence_is_saved_and_printed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["evidence_id"], "bev_1")
            self.assertEqual(printed["belief_id"], "bel_1")
            self.assertEqual(printed["source_event_ids"], ["evt_1"])
            self.assertEqual(printed["authorized_aggregation_mode"], "leaf_default")
            self.assertTrue(printed["is_active"])

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.source_event_ids, ["evt_1"])


class UnknownObservationRejectionTests(unittest.TestCase):
    def test_unknown_observation_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_ghost", *_VALID_ARGS,
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown observation_id", result.stderr)
            self.assertIn("obs_ghost", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()


class NoObservationEventsRejectionTests(unittest.TestCase):
    def test_observation_with_no_links_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                # Bypasses insert_observation()'s own "must have a primary
                # link" guard on purpose -- see module docstring.
                repo._conn.execute(
                    "INSERT INTO user_observations (observation_id, user_id, data) VALUES (?, ?, ?)",
                    ("obs_nolinks", "usr_1", json.dumps({
                        "observation_id": "obs_nolinks", "user_id": "usr_1", "category": "routine",
                        "observation": "x", "importance": 0.5, "confidence": 0.5,
                        "created_at": datetime.now(timezone.utc).isoformat(), **VERSION_FIELDS,
                    })),
                )
                repo._conn.commit()
            finally:
                repo.close()

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_nolinks", *_VALID_ARGS,
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no observation_events links", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()


class MissingLinkedEventRejectionTests(unittest.TestCase):
    def test_dangling_link_to_a_deleted_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            repo = Repository.at_path(db_path)
            try:
                # Simulates the event having been removed some other way,
                # leaving obs_1's link dangling -- see module docstring.
                repo._conn.execute("DELETE FROM user_events WHERE event_id = ?", ("evt_1",))
                repo._conn.commit()
            finally:
                repo.close()

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Linked event_id", result.stderr)
            self.assertIn("evt_1", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()


class InvalidChoiceRejectionTests(unittest.TestCase):
    def _base_args(self, db_path: str) -> list[str]:
        return [
            "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1",
            "--belief-id", "bel_1", "--context-key", "fitness", "--strength", "0.9",
            "--model-version", "demo-0.1",
        ]

    def test_invalid_direction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence(
                self._base_args(db_path) + [
                    "--belief-type", "behavioral_tendency", "--direction", "sideways",
                    "--source-type", "recorded_event",
                ]
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--direction", result.stderr)

    def test_invalid_source_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence(
                self._base_args(db_path) + [
                    "--belief-type", "behavioral_tendency", "--direction", "support",
                    "--source-type", "not_a_real_source_type",
                ]
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--source-type", result.stderr)

    def test_invalid_belief_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence(
                self._base_args(db_path) + [
                    "--belief-type", "not_a_real_belief_type", "--direction", "support",
                    "--source-type", "recorded_event",
                ]
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--belief-type", result.stderr)


class InvalidStrengthRejectionTests(unittest.TestCase):
    def test_out_of_range_strength_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1",
                "--belief-id", "bel_1", "--belief-type", "behavioral_tendency", "--direction", "support",
                "--source-type", "recorded_event", "--context-key", "fitness", "--strength", "1.5",
                "--model-version", "demo-0.1",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation", result.stderr.lower())

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()


class DuplicateEvidenceRejectionTests(unittest.TestCase):
    def test_second_insert_with_same_evidence_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            first = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
            ])
            self.assertEqual(first.returncode, 0, first.stderr)

            second = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
            ])
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Duplicate evidence_id", second.stderr)
            self.assertIn("bev_1", second.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)


class CliOutputShapeTests(unittest.TestCase):
    def test_output_is_a_single_belief_evidence_object_with_no_forbidden_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            for expected_field in (
                "evidence_id", "belief_id", "user_id", "direction", "source_event_ids",
                "source_type", "context_key", "strength", "source_reliability",
                "authorized_aggregation_mode", "independence_group", "is_active",
            ):
                self.assertIn(expected_field, printed)
            # This CLI has no flag for any of these -- they must never appear
            # as something the CLI itself set beyond authorize_evidence()'s
            # own safe defaults, and no UserBelief field belongs here at all.
            for forbidden_belief_field in (
                "confidence", "status", "belief_type", "belief_key", "belief_value",
                "supporting_evidence_count", "locked_until_recompute",
            ):
                self.assertNotIn(forbidden_belief_field, printed)


class ReplacementModeExclusivityTests(unittest.TestCase):
    def test_leaf_default_rejects_replaces_evidence_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_old", observation_id="obs_1", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
                "--replaces-evidence-id", "bev_old",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--replaces-evidence-id is only allowed with", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()

    def test_leaf_default_rejects_backend_validation_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
                "--backend-validation-passed",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--backend-validation-passed is only allowed with", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()

    def test_aggregate_replacement_requires_replaces_evidence_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("At least one --replaces-evidence-id is required", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()

    def test_duplicate_replaces_evidence_id_is_rejected_and_nothing_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_old", observation_id="obs_1", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement",
                "--replaces-evidence-id", "bev_old", "--replaces-evidence-id", "bev_old",
                "--backend-validation-passed",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate --replaces-evidence-id", result.stderr)
            self.assertIn("bev_old", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_new"))
                # The row named twice must also be untouched.
                old = repo.get_evidence("bev_old")
            finally:
                repo.close()
            self.assertTrue(old.is_active)
            self.assertFalse(old.is_duplicate_suppressed)


class UnknownReplacedEvidenceRejectionTests(unittest.TestCase):
    def test_unknown_replaces_evidence_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_1", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement", "--replaces-evidence-id", "bev_ghost",
                "--backend-validation-passed",
            ])

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown --replaces-evidence-id", result.stderr)
            self.assertIn("bev_ghost", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_evidence("bev_1"))
            finally:
                repo.close()


class ReplacementWithoutBackendValidationTests(unittest.TestCase):
    def test_valid_replacement_without_backend_validation_passed_stays_leaf_default_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_old", observation_id="obs_1", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement", "--replaces-evidence-id", "bev_old",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["proposed_aggregation_mode"], "aggregate_replacement")
            self.assertEqual(printed["authorized_aggregation_mode"], "leaf_default")
            self.assertEqual(printed["aggregation_review_status"], "pending")
            self.assertTrue(printed["aggregation_review_required"])

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_new")
            finally:
                repo.close()
            self.assertEqual(stored.authorized_aggregation_mode.value, "leaf_default")


class ReplacementWithBackendValidationTests(unittest.TestCase):
    def test_valid_replacement_with_backend_validation_passed_is_authorized_as_aggregate_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_old", observation_id="obs_1", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement", "--replaces-evidence-id", "bev_old",
                "--backend-validation-passed",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["authorized_aggregation_mode"], "aggregate_replacement")
            self.assertEqual(printed["aggregation_review_status"], "approved")
            self.assertFalse(printed["aggregation_review_required"])

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_evidence("bev_new")
            finally:
                repo.close()
            self.assertEqual(stored.authorized_aggregation_mode.value, "aggregate_replacement")


class ReplacementCrossUserRejectionTests(unittest.TestCase):
    def test_replacement_from_a_different_user_is_not_granted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_2", event_id="evt_2")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_observation(db_path, user_id="usr_2", observation_id="obs_2", event_ids=["evt_2"])
            _seed_evidence(db_path, evidence_id="bev_other_user", observation_id="obs_2", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement",
                "--replaces-evidence-id", "bev_other_user", "--backend-validation-passed",
            ])

            # authorize_evidence() decides this, not a CLI-level rejection --
            # the CLI still succeeds, but the replacement itself is refused.
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["authorized_aggregation_mode"], "leaf_default")
            self.assertEqual(printed["aggregation_review_status"], "rejected")


class ReplacementCrossBeliefRejectionTests(unittest.TestCase):
    def test_replacement_from_a_different_belief_is_not_granted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_other_belief", observation_id="obs_1", belief_id="bel_2")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement",
                "--replaces-evidence-id", "bev_other_belief", "--backend-validation-passed",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["authorized_aggregation_mode"], "leaf_default")
            self.assertEqual(printed["aggregation_review_status"], "rejected")


class ReplacementCoverageNarrowingRejectionTests(unittest.TestCase):
    def test_replacement_that_narrows_source_event_ids_coverage_is_not_granted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_event(db_path, user_id="usr_1", event_id="evt_2")
            _seed_observation(
                db_path, user_id="usr_1", observation_id="obs_multi", event_ids=["evt_1", "evt_2"],
                primary_event_id="evt_1",
            )
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_single", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_multi", observation_id="obs_multi", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_single", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement",
                "--replaces-evidence-id", "bev_multi", "--backend-validation-passed",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["source_event_ids"], ["evt_1"])
            self.assertEqual(printed["authorized_aggregation_mode"], "leaf_default")
            self.assertEqual(printed["aggregation_review_status"], "rejected")


class ReplacementDoesNotRecomputeBeliefsTests(unittest.TestCase):
    def test_cli_does_not_create_or_save_a_belief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_event(db_path, user_id="usr_1", event_id="evt_1")
            _seed_observation(db_path, user_id="usr_1", observation_id="obs_1", event_ids=["evt_1"])
            _seed_evidence(db_path, evidence_id="bev_old", observation_id="obs_1", belief_id="bel_1")

            result = _run_evidence([
                "--db", db_path, "--evidence-id", "bev_new", "--observation-id", "obs_1", *_VALID_ARGS,
                "--proposed-aggregation-mode", "aggregate_replacement", "--replaces-evidence-id", "bev_old",
                "--backend-validation-passed",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertIsNone(repo.get_latest_belief(user_id="usr_1", belief_id="bel_1"))
                # The replaced row itself is untouched -- this CLI creates
                # the new replacement evidence row only.
                old = repo.get_evidence("bev_old")
            finally:
                repo.close()
            self.assertTrue(old.is_active)
            self.assertFalse(old.is_duplicate_suppressed)


if __name__ == "__main__":
    unittest.main()

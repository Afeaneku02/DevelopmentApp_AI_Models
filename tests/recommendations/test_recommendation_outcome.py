"""Tests for the recommendation_outcomes MVP -- RecommendationOutcome,
Repository.insert/get/list_recommendation_outcome(s), and
tools/add_recommendation_outcome.py.

Proves the three guarantees the task requires:
- outcomes are append-only (many per recommendation; no update/delete path);
- an outcome must reference an existing recommendation;
- recording an outcome never mutates the frozen recommendation state.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.beliefs.models import UserBelief
from src.recommendations.engine import generate_recommendation
from src.recommendations.models import RecommendationOutcome
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
_CLI = Path(__file__).resolve().parents[2] / "tools" / "add_recommendation_outcome.py"


def _belief(belief_id: str = "b1") -> UserBelief:
    return UserBelief(
        belief_id=belief_id, user_id="usr_1", belief_type="behavioral_tendency",
        belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
        belief_value=True, confidence=0.7, supporting_evidence_count=3, contradicting_evidence_count=0,
        total_evidence_count=3, effective_support_count=2.0, effective_evidence_count=2.0,
        evidence_for=3, evidence_against=0, allowed_contexts=[], disallowed_contexts=[],
        sensitivity_class="normal", persistence_policy="retained",
        first_observed=AS_OF - timedelta(days=5), last_validated=AS_OF, status="validated",
        **VERSION_FIELDS,
    )


def _recommendation(recommendation_id: str = "rec_1"):
    return generate_recommendation(
        recommendation_id=recommendation_id, user_id="usr_1", context_key="fitness_scheduling",
        beliefs=[_belief()], created_at=AS_OF,
    )


def _outcome(outcome_id: str, *, recommendation_id: str = "rec_1", followed: str = "followed",
             result: str = "successful", source: str = "app_event", created_at: datetime = AS_OF,
             **extra) -> RecommendationOutcome:
    return RecommendationOutcome(
        outcome_id=outcome_id, recommendation_id=recommendation_id, followed=followed,
        result=result, source=source, created_at=created_at, **extra, **VERSION_FIELDS,
    )


class OutcomePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Repository.in_memory()
        self.repo.insert_recommendation(_recommendation("rec_1"))

    def tearDown(self) -> None:
        self.repo.close()

    def test_round_trips_every_field(self) -> None:
        outcome = _outcome(
            "out_1", followed="partially_followed", result="mixed", source="user_survey",
            user_feedback="tried the earlier slot twice", measured_result="2 of 5 sessions on time",
            observed_at=AS_OF + timedelta(days=3),
        )
        self.repo.insert_recommendation_outcome(outcome)

        loaded = self.repo.get_recommendation_outcome("out_1")
        self.assertEqual(loaded.model_dump(), outcome.model_dump())
        self.assertEqual(loaded.followed.value, "partially_followed")
        self.assertEqual(loaded.user_feedback, "tried the earlier slot twice")
        self.assertEqual(loaded.measured_result, "2 of 5 sessions on time")
        self.assertEqual(loaded.source, "user_survey")

    def test_outcomes_are_append_only_many_per_recommendation(self) -> None:
        self.repo.insert_recommendation_outcome(
            _outcome("out_1", followed="followed", result="not_yet_known", source="app_event")
        )
        self.repo.insert_recommendation_outcome(
            _outcome("out_2", followed="followed", result="successful", source="user_survey",
                     created_at=AS_OF + timedelta(days=7))
        )
        self.repo.insert_recommendation_outcome(
            _outcome("out_3", followed="not_followed", result="unknown", source="manual",
                     created_at=AS_OF + timedelta(days=14))
        )

        trail = self.repo.list_recommendation_outcomes(recommendation_id="rec_1")
        self.assertEqual([o.outcome_id for o in trail], ["out_1", "out_2", "out_3"])
        # No update/delete surface exists on the repository.
        self.assertFalse(hasattr(self.repo, "update_recommendation_outcome"))
        self.assertFalse(hasattr(self.repo, "delete_recommendation_outcome"))

    def test_duplicate_outcome_id_is_rejected_and_the_original_is_unchanged(self) -> None:
        self.repo.insert_recommendation_outcome(_outcome("out_1", result="successful"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo.insert_recommendation_outcome(_outcome("out_1", result="unsuccessful"))

        stored = self.repo.list_recommendation_outcomes(recommendation_id="rec_1")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].result.value, "successful")

    def test_outcome_must_reference_an_existing_recommendation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.repo.insert_recommendation_outcome(
                _outcome("out_ghost", recommendation_id="rec_does_not_exist")
            )
        self.assertIn("no recommendation", str(ctx.exception))
        self.assertIsNone(self.repo.get_recommendation_outcome("out_ghost"))
        self.assertEqual(self.repo.list_recommendation_outcomes(), [])

    def test_recording_an_outcome_does_not_mutate_the_frozen_recommendation(self) -> None:
        before = self.repo.get_recommendation("rec_1").model_dump_json()

        self.repo.insert_recommendation_outcome(_outcome("out_1"))
        self.repo.insert_recommendation_outcome(
            _outcome("out_2", followed="not_followed", result="unsuccessful",
                     user_feedback="forgot", created_at=AS_OF + timedelta(days=2))
        )

        after = self.repo.get_recommendation("rec_1").model_dump_json()
        self.assertEqual(after, before)

    def test_list_without_scope_returns_outcomes_across_recommendations(self) -> None:
        self.repo.insert_recommendation(_recommendation("rec_2"))
        self.repo.insert_recommendation_outcome(_outcome("out_1", recommendation_id="rec_1"))
        self.repo.insert_recommendation_outcome(_outcome("out_2", recommendation_id="rec_2"))

        self.assertEqual(
            [o.outcome_id for o in self.repo.list_recommendation_outcomes()], ["out_1", "out_2"]
        )
        self.assertEqual(
            [o.outcome_id for o in self.repo.list_recommendation_outcomes(recommendation_id="rec_2")],
            ["out_2"],
        )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class AddRecommendationOutcomeCliTests(unittest.TestCase):
    def _seed_recommendation(self, db_path: str, recommendation_id: str = "rec_1") -> None:
        repo = Repository.at_path(db_path)
        try:
            repo.save_belief(_belief())
            repo.insert_recommendation(_recommendation(recommendation_id))
        finally:
            repo.close()

    def test_end_to_end_appends_an_outcome(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed_recommendation(db_path)

            result = _run_cli([
                "--db", db_path, "--outcome-id", "out_1", "--recommendation-id", "rec_1",
                "--followed", "followed", "--result", "successful", "--source", "app_event",
                "--user-feedback", "did the after-work slot",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["recommendation_id"], "rec_1")
            self.assertEqual(printed["followed"], "followed")

            repo = Repository.at_path(db_path)
            try:
                stored = repo.list_recommendation_outcomes(recommendation_id="rec_1")
                rec_json = repo.get_recommendation("rec_1").model_dump_json()
            finally:
                repo.close()
            self.assertEqual([o.outcome_id for o in stored], ["out_1"])
            self.assertEqual(stored[0].user_feedback, "did the after-work slot")
            # recommendation still exactly as issued
            self.assertEqual(json.loads(rec_json)["recommendation_id"], "rec_1")

    def test_unknown_recommendation_id_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed_recommendation(db_path)
            result = _run_cli([
                "--db", db_path, "--outcome-id", "out_1", "--recommendation-id", "rec_missing",
                "--followed", "ignored", "--result", "unknown", "--source", "manual",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no recommendation", result.stderr)

    def test_duplicate_outcome_id_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed_recommendation(db_path)
            args = [
                "--db", db_path, "--outcome-id", "out_1", "--recommendation-id", "rec_1",
                "--followed", "followed", "--result", "successful", "--source", "app_event",
            ]
            self.assertEqual(_run_cli(args).returncode, 0)
            second = _run_cli(args)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Duplicate outcome_id", second.stderr)

    def test_invalid_followed_value_is_rejected_by_argparse(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed_recommendation(db_path)
            result = _run_cli([
                "--db", db_path, "--outcome-id", "out_1", "--recommendation-id", "rec_1",
                "--followed", "sort_of", "--result", "successful", "--source", "app_event",
            ])
            self.assertEqual(result.returncode, 2)
            self.assertIn("--followed", result.stderr)


if __name__ == "__main__":
    unittest.main()

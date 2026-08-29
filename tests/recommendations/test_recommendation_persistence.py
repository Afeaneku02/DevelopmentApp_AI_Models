"""Tests for Repository.insert_recommendation / get_recommendation /
list_recommendations and the tools/make_recommendation.py CLI.

The recommendation table is append-only with a UNIQUE recommendation_id,
matching every other externally-supplied record id in this project.
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
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
_MAKE_REC_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "make_recommendation.py"


def _belief(belief_id: str, *, user_id: str = "usr_1", belief_key: str = "higher_adherence_after_work",
            belief_type: str = "behavioral_tendency", status: str = "validated",
            confidence: float = 0.7, belief_value: object = True,
            sensitivity_class: str = "normal", persistence_policy: str = "retained") -> UserBelief:
    return UserBelief(
        belief_id=belief_id, user_id=user_id, belief_type=belief_type,
        belief_type_registry_version="belief-types-0.6", belief_key=belief_key,
        belief_value=belief_value, confidence=confidence, supporting_evidence_count=3,
        contradicting_evidence_count=0, total_evidence_count=3, effective_support_count=2.0,
        effective_evidence_count=2.0, evidence_for=3, evidence_against=0,
        allowed_contexts=[], disallowed_contexts=[], sensitivity_class=sensitivity_class,
        persistence_policy=persistence_policy, first_observed=AS_OF - timedelta(days=5),
        last_validated=AS_OF, status=status, **VERSION_FIELDS,
    )


def _rec(recommendation_id: str, beliefs, context_key: str = "fitness_scheduling"):
    return generate_recommendation(
        recommendation_id=recommendation_id, user_id="usr_1", context_key=context_key,
        beliefs=beliefs, created_at=AS_OF,
    )


class RecommendationPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Repository.in_memory()

    def tearDown(self) -> None:
        self.repo.close()

    def test_insert_then_get_round_trips_every_field(self) -> None:
        rec = _rec("rec_1", [_belief("b1")])
        self.repo.insert_recommendation(rec)

        loaded = self.repo.get_recommendation("rec_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.model_dump(), rec.model_dump())
        self.assertEqual(loaded.belief_ids_used, ["b1"])
        self.assertEqual(loaded.risk_tier.value, "low")

    def test_list_can_be_scoped_by_user_and_context(self) -> None:
        self.repo.insert_recommendation(_rec("rec_1", [_belief("b1")]))
        self.repo.insert_recommendation(
            _rec("rec_2", [_belief("b1", belief_key="prefers_short_messages",
                                   belief_type="communication_or_learning_preference")],
                 context_key="habit_nudge")
        )

        self.assertEqual(
            [r.recommendation_id for r in self.repo.list_recommendations(user_id="usr_1")],
            ["rec_1", "rec_2"],
        )
        scoped = self.repo.list_recommendations(
            user_id="usr_1", recommendation_context="habit_nudge"
        )
        self.assertEqual([r.recommendation_id for r in scoped], ["rec_2"])

    def test_duplicate_recommendation_id_is_rejected_and_original_unchanged(self) -> None:
        self.repo.insert_recommendation(_rec("rec_1", [_belief("b1")]))
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo.insert_recommendation(_rec("rec_1", [_belief("b2")]))

        stored = self.repo.list_recommendations(user_id="usr_1")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].belief_ids_used, ["b1"])

    def test_a_review_gated_recommendation_persists_its_review_fields(self) -> None:
        # blueprint section 14.1: persisted recommendations record
        # review_required, review_status, risk_resolution_path, and the exact
        # risk policy/domain policy versions used.
        belief = _belief("b1", belief_key="prefers_short_messages",
                         belief_type="communication_or_learning_preference")
        rec = _rec("rec_hi", [belief], context_key="mental_health_support")
        self.repo.insert_recommendation(rec)

        loaded = self.repo.get_recommendation("rec_hi")
        self.assertTrue(loaded.review_required)
        self.assertEqual(loaded.review_status.value, "pending")
        self.assertEqual(loaded.required_resolution_mode.value, "reviewer")
        # mental_health_support is an exact context-policy row.
        self.assertEqual(loaded.risk_resolution_path.value, "exact_context")
        self.assertTrue(loaded.risk_policy_version)
        self.assertTrue(loaded.risk_domain_policy_version)
        self.assertFalse(loaded.exploration_applied)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_MAKE_REC_SCRIPT), *args], capture_output=True, text=True
    )


class MakeRecommendationCliTests(unittest.TestCase):
    def _seed(self, db_path: str, beliefs) -> None:
        repo = Repository.at_path(db_path)
        try:
            for belief in beliefs:
                repo.save_belief(belief)
        finally:
            repo.close()

    def test_end_to_end_issues_and_persists_a_low_risk_recommendation(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [_belief("b1")])

            result = _run_cli([
                "--db", db_path, "--user-id", "usr_1", "--context-key", "fitness_scheduling",
                "--recommendation-id", "rec_1", "--goal", "be consistent",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["belief_ids_used"], ["b1"])
            self.assertFalse(printed["review_required"])

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_recommendation("rec_1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.goal, "be consistent")
            self.assertEqual(stored.risk_tier.value, "low")

    def test_high_risk_context_persists_a_pending_review_and_excludes_sensitive_belief(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [
                _belief("b_ok", belief_key="prefers_short_messages",
                        belief_type="communication_or_learning_preference"),
                _belief("b_sensitive", belief_key="infers_condition",
                        belief_type="sensitive_or_high_impact_inference",
                        sensitivity_class="restricted", persistence_policy="do_not_persist"),
            ])

            result = _run_cli([
                "--db", db_path, "--user-id", "usr_1", "--context-key", "mental_health_support",
                "--recommendation-id", "rec_hi",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("manual review", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_recommendation("rec_hi")
            finally:
                repo.close()
            self.assertTrue(stored.review_required)
            self.assertNotIn("b_sensitive", stored.belief_ids_used)
            blocked_ids = {b.belief_id for b in stored.blocked_beliefs}
            self.assertIn("b_sensitive", blocked_ids)

    def test_duplicate_recommendation_id_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [_belief("b1")])
            args = [
                "--db", db_path, "--user-id", "usr_1", "--context-key", "fitness_scheduling",
                "--recommendation-id", "rec_1",
            ]
            self.assertEqual(_run_cli(args).returncode, 0)
            second = _run_cli(args)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Duplicate recommendation_id", second.stderr)


if __name__ == "__main__":
    unittest.main()

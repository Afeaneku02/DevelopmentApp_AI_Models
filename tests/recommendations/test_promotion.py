"""Tests for backend promotion of outcome-learning signals into belief_evidence
(src/recommendations/promotion.py + tools/promote_outcome_learning_signal.py).

Proves the seven guarantees the task requires:
1. learning never auto-promotes;
2. promotion is rejected for no_signal / causal_claim=true / missing proposals;
3. the authorized evidence carries the signal's independence_group;
4. re-running promotion cannot add a second row for the same signal;
5. a dry run writes nothing;
6. persist without --recompute locks the affected beliefs;
7. persist with --recompute updates confidence through the real recompute path.
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

from src.beliefs.models import authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.common.enums import SourceType
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.recommendations.engine import generate_recommendation
from src.recommendations.models import OutcomeLearningSignal, RecommendationOutcome
from src.recommendations.outcome_learning import analyze_recommendation_outcomes
from src.recommendations.promotion import (
    evaluate_signal_for_promotion,
    promote_outcome_learning_signal,
)
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
LEARNED_AT = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
PROMOTED_AT = LEARNED_AT + timedelta(days=30)
_CLI = Path(__file__).resolve().parents[2] / "tools" / "promote_outcome_learning_signal.py"


def _seed_model(repo: Repository, *, user_id: str = "usr_1", belief_id: str = "bel_1",
                event_id: str = "evt_1", specs: list[tuple[str, str]] | None = None):
    """A real event->observation->evidence->belief chain plus 4 (or len(specs))
    recommendations + outcomes for one context, then the learning signal."""
    specs = specs or [("followed", "successful")] * 4

    event = UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=LEARNED_AT - timedelta(days=20), source="app", **VERSION_FIELDS,
    )
    repo.insert_event(event)
    observation, links = create_observation_from_event(
        event, observation_id=f"obs_{event_id}", category="routine", observation_text="x",
        importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
    )
    repo.insert_observation(observation, links)
    proposal = propose_evidence_from_observation_validated(
        observation, links, [event], belief_id=belief_id, direction="support",
        source_type="recorded_event", context_key="fitness", strength=0.9,
        model_version="pipeline-0.1", belief_type="behavioral_tendency", **VERSION_FIELDS,
    )
    repo.insert_evidence(authorize_evidence(
        proposal, evidence_id=f"bev_{event_id}", created_at=event.timestamp,
        aggregation_policy_version="evidence-aggregation-0.6",
    ))
    belief = recompute_belief(
        belief_id=belief_id, user_id=user_id, belief_type="behavioral_tendency",
        belief_key="higher_adherence_after_work", belief_value=True,
        evidence=repo.list_active_evidence(user_id=user_id, belief_id=belief_id),
        as_of=LEARNED_AT, first_observed=event.timestamp, **VERSION_FIELDS,
    )
    repo.save_belief(belief)

    recs = []
    outcomes = []
    for index, (followed, result) in enumerate(specs, start=1):
        rec_id = f"rec_{user_id}_{index}"
        rec = generate_recommendation(
            recommendation_id=rec_id, user_id=user_id, context_key="fitness_scheduling",
            beliefs=[belief], created_at=LEARNED_AT,
        )
        repo.insert_recommendation(rec)
        recs.append(rec)
        outcome = RecommendationOutcome(
            outcome_id=f"out_{user_id}_{index}", recommendation_id=rec_id, followed=followed,
            result=result, source="app_event", created_at=LEARNED_AT + timedelta(days=index),
            **VERSION_FIELDS,
        )
        repo.insert_recommendation_outcome(outcome)
        outcomes.append(outcome)

    signal = analyze_recommendation_outcomes(
        recs, outcomes, as_of=LEARNED_AT, belief_source_events={belief_id: [event_id]}
    )[0]
    repo.insert_outcome_learning_signal(signal)
    return belief, signal


def _rps_rows(repo: Repository, *, user_id: str, belief_id: str):
    return [
        row for row in repo.list_evidence(user_id=user_id, belief_id=belief_id)
        if row.source_type is SourceType.REPEATED_PATTERN_SUMMARY
    ]


class LearningNeverAutoPromotesTests(unittest.TestCase):
    def test_analyzing_and_storing_a_signal_writes_no_belief_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            self.assertEqual(_rps_rows(repo, user_id="usr_1", belief_id="bel_1"), [])
            # the signal exists but nothing has been promoted
            self.assertIsNotNone(repo.get_outcome_learning_signal(signal.signal_id))
            self.assertEqual(len(repo.list_evidence(user_id="usr_1", belief_id="bel_1")), 1)
        finally:
            repo.close()


class PromotionGateTests(unittest.TestCase):
    def _signal(self, repo: Repository):
        _, signal = _seed_model(repo)
        return signal

    def test_a_valid_support_signal_passes_the_gate(self) -> None:
        repo = Repository.in_memory()
        try:
            ok, reason = evaluate_signal_for_promotion(self._signal(repo))
        finally:
            repo.close()
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_no_signal_kind_is_rejected(self) -> None:
        repo = Repository.in_memory()
        try:
            base = self._signal(repo)
            tampered = OutcomeLearningSignal(
                **{**base.model_dump(), "signal_id": "ns", "kind": "no_signal",
                   "direction": None, "proposed_evidence": []}
            )
            repo.insert_outcome_learning_signal(tampered)
            result = promote_outcome_learning_signal(repo, signal_id="ns", as_of=PROMOTED_AT)
        finally:
            repo.close()
        self.assertFalse(result.authorized)
        self.assertIn("not promotable", result.rejected_reason)
        self.assertEqual(result.proposals, [])

    def test_causal_claim_true_is_rejected(self) -> None:
        repo = Repository.in_memory()
        try:
            base = self._signal(repo)
            # bypass the model validator that pins causal_claim False, the same
            # way other tests simulate data that reached the DB some other way
            row = json.loads(base.model_dump_json())
            row["signal_id"] = "cc"
            row["causal_claim"] = True
            with repo._conn:
                repo._conn.execute(
                    "INSERT INTO outcome_learning_signals "
                    "(signal_id, user_id, recommendation_context, data) VALUES (?, ?, ?, ?)",
                    ("cc", row["user_id"], row["recommendation_context"], json.dumps(row)),
                )
            with self.assertRaises(Exception):
                # model_validate_json inside get_outcome_learning_signal rejects it
                repo.get_outcome_learning_signal("cc")
        finally:
            repo.close()

    def test_missing_proposed_evidence_is_rejected(self) -> None:
        repo = Repository.in_memory()
        try:
            base = self._signal(repo)
            tampered = OutcomeLearningSignal(
                **{**base.model_dump(), "signal_id": "empty", "kind": "no_signal",
                   "direction": None, "proposed_evidence": []}
            )
            repo.insert_outcome_learning_signal(tampered)
            ok, reason = evaluate_signal_for_promotion(tampered)
        finally:
            repo.close()
        self.assertFalse(ok)

    def test_below_threshold_trial_count_is_rejected(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo, specs=[("followed", "successful")] * 3)
            # force the stored count low without touching proposals
            tampered = signal.model_copy(update={
                "signal_id": "low", "trial_count": 2, "supportive_count": 2,
                "adverse_count": 0, "neutral_count": 0,
            })
            repo.insert_outcome_learning_signal(tampered)
            result = promote_outcome_learning_signal(repo, signal_id="low", as_of=PROMOTED_AT)
        finally:
            repo.close()
        self.assertFalse(result.authorized)
        self.assertIn("threshold", result.rejected_reason)

    def test_missing_signal_id_raises(self) -> None:
        repo = Repository.in_memory()
        try:
            with self.assertRaises(ValueError):
                promote_outcome_learning_signal(repo, signal_id="does-not-exist", as_of=PROMOTED_AT)
        finally:
            repo.close()


class PromotionUsesIndependenceGroupTests(unittest.TestCase):
    def test_authorized_evidence_carries_the_signals_independence_group(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            result = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT, persist=True
            )
            rows = _rps_rows(repo, user_id="usr_1", belief_id="bel_1")
        finally:
            repo.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].independence_group, signal.independence_group)
        self.assertEqual(result.independence_group, signal.independence_group)
        self.assertEqual(rows[0].source_type, SourceType.REPEATED_PATTERN_SUMMARY)
        self.assertEqual(rows[0].aggregation_authorized_by, "outcome_learning_promotion")


class ReRunningPromotionCannotDoubleCountTests(unittest.TestCase):
    def test_second_promotion_of_the_same_signal_adds_nothing(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            first = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT, persist=True
            )
            second = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT + timedelta(days=1), persist=True
            )
            rows = _rps_rows(repo, user_id="usr_1", belief_id="bel_1")
        finally:
            repo.close()
        self.assertEqual([p.action for p in first.proposals], ["inserted"])
        self.assertEqual([p.action for p in second.proposals], ["skipped_existing"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(second.locked_belief_ids, [])


class DryRunModifiesNothingTests(unittest.TestCase):
    def test_dry_run_writes_no_evidence_and_does_not_lock(self) -> None:
        repo = Repository.in_memory()
        try:
            belief, signal = _seed_model(repo)
            evidence_before = [e.model_dump_json() for e in repo.list_all_evidence(user_id="usr_1")]
            belief_before = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1").model_dump_json()

            result = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT
            )

            evidence_after = [e.model_dump_json() for e in repo.list_all_evidence(user_id="usr_1")]
            belief_after = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1").model_dump_json()
        finally:
            repo.close()

        self.assertTrue(result.authorized)
        self.assertFalse(result.persisted)
        self.assertEqual([p.action for p in result.proposals], ["authorized"])
        self.assertEqual(evidence_before, evidence_after)
        self.assertEqual(belief_before, belief_after)

    def test_cli_dry_run_leaves_the_db_byte_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                _, signal = _seed_model(repo)
            finally:
                repo.close()
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path, "--signal-id", signal.signal_id])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dry run", result.stderr)
            self.assertEqual(Path(db_path).read_bytes(), before)


class PersistWithoutRecomputeLocksTests(unittest.TestCase):
    def test_persist_locks_the_affected_belief(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            result = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT, persist=True
            )
            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            rps_count = len(_rps_rows(repo, user_id="usr_1", belief_id="bel_1"))
        finally:
            repo.close()
        self.assertTrue(result.persisted)
        self.assertFalse(result.recomputed)
        self.assertEqual(result.locked_belief_ids, ["bel_1"])
        self.assertTrue(latest.locked_until_recompute)
        # the promoted row is in the ledger, active
        self.assertEqual(rps_count, 1)


class PersistWithRecomputeUpdatesConfidenceTests(unittest.TestCase):
    def test_recompute_runs_the_real_path_and_clears_the_lock(self) -> None:
        repo = Repository.in_memory()
        try:
            baseline_belief, signal = _seed_model(repo)
            result = promote_outcome_learning_signal(
                repo, signal_id=signal.signal_id, as_of=PROMOTED_AT, persist=True, recompute=True
            )
            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
        finally:
            repo.close()

        self.assertTrue(result.recomputed)
        self.assertEqual(result.locked_belief_ids, [])
        self.assertEqual(len(result.recomputed_beliefs), 1)
        self.assertFalse(latest.locked_until_recompute)
        # the weak support row nudged confidence up through recompute_belief
        self.assertGreater(latest.confidence, baseline_belief.confidence)
        self.assertEqual(result.recomputed_beliefs[0]["confidence"], latest.confidence)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class PromoteCliTests(unittest.TestCase):
    def _seed(self, db_path: str):
        repo = Repository.at_path(db_path)
        try:
            _, signal = _seed_model(repo)
        finally:
            repo.close()
        return signal

    def test_missing_db_exits_nonzero_without_a_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.sqlite3")
            result = _run_cli(["--db", missing, "--signal-id", "whatever"])
            self.assertEqual(result.returncode, 1)
            self.assertIn("no such database file", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_missing_db_is_not_created_in_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "typo.sqlite3"
            result = _run_cli(["--db", str(missing), "--signal-id", "whatever"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(missing.exists())

    def test_missing_db_is_not_created_with_persist(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "typo.sqlite3"
            result = _run_cli(["--db", str(missing), "--signal-id", "whatever", "--persist"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(missing.exists())

    def test_recompute_without_persist_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal = self._seed(db_path)
            result = _run_cli(["--db", db_path, "--signal-id", signal.signal_id, "--recompute"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--recompute requires --persist", result.stderr)

    def test_persist_then_recompute_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal = self._seed(db_path)

            persisted = _run_cli([
                "--db", db_path, "--signal-id", signal.signal_id, "--persist",
                "--as-of", PROMOTED_AT.isoformat(),
            ])
            self.assertEqual(persisted.returncode, 0, persisted.stderr)
            self.assertIn("locked", persisted.stderr)

            repo = Repository.at_path(db_path)
            try:
                self.assertTrue(
                    repo.get_latest_belief(user_id="usr_1", belief_id="bel_1").locked_until_recompute
                )
                rps = _rps_rows(repo, user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(len(rps), 1)

            recomputed = _run_cli([
                "--db", db_path, "--signal-id", signal.signal_id, "--persist", "--recompute",
                "--as-of", PROMOTED_AT.isoformat(),
            ])
            self.assertEqual(recomputed.returncode, 0, recomputed.stderr)

            repo = Repository.at_path(db_path)
            try:
                latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertFalse(latest.locked_until_recompute)

    def test_rejected_signal_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal = self._seed(db_path)
            repo = Repository.at_path(db_path)
            try:
                tampered = OutcomeLearningSignal(
                    **{**signal.model_dump(), "signal_id": "ns", "kind": "no_signal",
                       "direction": None, "proposed_evidence": []}
                )
                repo.insert_outcome_learning_signal(tampered)
            finally:
                repo.close()
            result = _run_cli(["--db", db_path, "--signal-id", "ns", "--persist"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not promoted", result.stderr)


if __name__ == "__main__":
    unittest.main()

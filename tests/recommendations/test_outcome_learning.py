"""Tests for the outcome-learning MVP -- src/recommendations/outcome_learning.py,
the outcome_learning_signals repository methods, and
tools/learn_from_recommendation_outcomes.py.

Proves the five guarantees the task requires:
1. below the repeated-trial threshold, no evidence is produced;
2. repeated followed-and-successful outcomes produce conservative support;
3. a minority of bad/ignored outcomes never over-penalises a belief;
4. nothing generated claims causality;
5. running the CLI without --persist cannot modify the database.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.beliefs.models import UserBelief
from src.common.enums import Direction, OutcomeLearningSignalKind
from src.common.registry import OUTCOME_LEARNING_POLICY, OUTCOME_LEARNING_VERSION, SOURCE_TYPE_RELIABILITY
from src.common.enums import SourceType
from src.recommendations.engine import generate_recommendation
from src.recommendations.models import RecommendationOutcome
from src.recommendations.outcome_learning import analyze_recommendation_outcomes
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
MIN_TRIALS = OUTCOME_LEARNING_POLICY.min_trials
_CLI = Path(__file__).resolve().parents[2] / "tools" / "learn_from_recommendation_outcomes.py"


def _belief(belief_id: str = "b1") -> UserBelief:
    return UserBelief(
        belief_id=belief_id, user_id="usr_1", belief_type="behavioral_tendency",
        belief_type_registry_version="belief-types-0.6", belief_key="higher_adherence_after_work",
        belief_value=True, confidence=0.7, supporting_evidence_count=3, contradicting_evidence_count=0,
        total_evidence_count=3, effective_support_count=2.0, effective_evidence_count=2.0,
        evidence_for=3, evidence_against=0, allowed_contexts=[], disallowed_contexts=[],
        sensitivity_class="normal", persistence_policy="retained",
        first_observed=AS_OF - timedelta(days=30), last_validated=AS_OF, status="validated",
        **VERSION_FIELDS,
    )


def _recommendation(recommendation_id: str, *, context_key: str = "fitness_scheduling"):
    return generate_recommendation(
        recommendation_id=recommendation_id, user_id="usr_1", context_key=context_key,
        beliefs=[_belief("b1")], created_at=AS_OF,
    )


def _outcome(index: int, recommendation_id: str, followed: str, result: str) -> RecommendationOutcome:
    return RecommendationOutcome(
        outcome_id=f"out_{index}", recommendation_id=recommendation_id, followed=followed,
        result=result, source="app_event", created_at=AS_OF + timedelta(days=index), **VERSION_FIELDS,
    )


def _scenario(specs: list[tuple[str, str]], *, context_key: str = "fitness_scheduling"):
    """specs: list of (followed, result); one recommendation + one outcome each."""
    recommendations = []
    outcomes = []
    for index, (followed, result) in enumerate(specs, start=1):
        rec_id = f"rec_{index}"
        recommendations.append(_recommendation(rec_id, context_key=context_key))
        outcomes.append(_outcome(index, rec_id, followed, result))
    return recommendations, outcomes


class ThresholdTests(unittest.TestCase):
    def test_below_threshold_produces_no_signal_at_all(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * (MIN_TRIALS - 1))
        self.assertEqual(analyze_recommendation_outcomes(recs, outs, as_of=AS_OF), [])

    def test_exactly_at_threshold_produces_a_signal(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
        signals = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].trial_count, MIN_TRIALS)


class ConservativeSupportTests(unittest.TestCase):
    def test_repeated_successful_outcomes_produce_weak_support_evidence(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * 4 + [("partially_followed", "mixed")])
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]

        self.assertEqual(signal.kind, OutcomeLearningSignalKind.SUPPORT)
        self.assertEqual(signal.direction, Direction.SUPPORT)
        self.assertEqual(len(signal.proposed_evidence), 1)
        proposal = signal.proposed_evidence[0]
        self.assertEqual(proposal.direction, Direction.SUPPORT)
        self.assertEqual(proposal.source_type, SourceType.REPEATED_PATTERN_SUMMARY)
        self.assertEqual(
            proposal.source_reliability, SOURCE_TYPE_RELIABILITY[SourceType.REPEATED_PATTERN_SUMMARY]
        )
        # "conservative" = far below a recorded_event leaf; capped low.
        self.assertLessEqual(proposal.strength, OUTCOME_LEARNING_POLICY.support_strength_cap)
        self.assertLess(proposal.strength, 0.4)
        self.assertEqual(proposal.belief_id, "b1")

    def test_support_strength_is_capped_no_matter_how_many_successes(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * 40)
        proposal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0].proposed_evidence[0]
        self.assertEqual(proposal.strength, OUTCOME_LEARNING_POLICY.support_strength_cap)

    def test_provenance_is_preserved_on_the_signal(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
        self.assertEqual(signal.recommendation_ids, [f"rec_{i}" for i in range(1, MIN_TRIALS + 1)])
        self.assertEqual(signal.outcome_ids, [f"out_{i}" for i in range(1, MIN_TRIALS + 1)])
        self.assertIn("outcome-learning:fitness_scheduling", signal.independence_group)

    def test_real_backing_events_are_used_when_supplied(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
        signal = analyze_recommendation_outcomes(
            recs, outs, as_of=AS_OF, belief_source_events={"b1": ["evt_9", "evt_7", "evt_7"]}
        )[0]
        self.assertEqual(signal.proposed_evidence[0].source_event_ids, ["evt_7", "evt_9"])


class DoesNotOverPenalizeTests(unittest.TestCase):
    def test_one_failure_among_many_successes_still_yields_support(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * 5 + [("not_followed", "unsuccessful")])
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
        self.assertEqual(signal.kind, OutcomeLearningSignalKind.SUPPORT)
        self.assertEqual(signal.adverse_count, 1)

    def test_a_couple_of_bad_outcomes_below_threshold_produce_no_signal(self) -> None:
        recs, outs = _scenario(
            [("not_followed", "unsuccessful"), ("ignored", "unknown"), ("followed", "successful")]
        )
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
        self.assertEqual(signal.kind, OutcomeLearningSignalKind.NO_SIGNAL)
        self.assertIsNone(signal.direction)
        self.assertEqual(signal.proposed_evidence, [])

    def test_only_a_clearly_dominant_repeated_failure_pattern_makes_weak_contradiction(self) -> None:
        recs, outs = _scenario(
            [("not_followed", "unsuccessful")] * 4 + [("followed", "successful")]
        )
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
        self.assertEqual(signal.kind, OutcomeLearningSignalKind.WEAK_CONTRADICTION)
        self.assertEqual(signal.direction, Direction.CONTRADICT)
        proposal = signal.proposed_evidence[0]
        self.assertEqual(proposal.direction, Direction.CONTRADICT)
        # even weaker than support
        self.assertLessEqual(proposal.strength, OUTCOME_LEARNING_POLICY.contradiction_strength_cap)


class NoCausalClaimTests(unittest.TestCase):
    def test_signal_never_claims_causality(self) -> None:
        recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
        signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]

        self.assertFalse(signal.causal_claim)
        lowered = signal.rationale.lower()
        self.assertIn("correlational", lowered)
        self.assertTrue(
            "does not establish" in lowered or "asserts no causation" in lowered
        )
        # no affirmative causal assertion (the disclaimer negates causation,
        # so match whole affirmative phrasings, not the bare word "caused")
        for affirmative in (
            "the recommendation caused",
            "the recommendation worked",
            "because the recommendation",
            "proves that",
        ):
            self.assertNotIn(affirmative, lowered)
        # the proposal is about the belief, not "the recommendation worked"
        self.assertEqual(signal.proposed_evidence[0].belief_id, "b1")

    def test_model_rejects_a_signal_that_sets_causal_claim_true(self) -> None:
        from pydantic import ValidationError

        from src.recommendations.models import OutcomeLearningSignal

        recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
        good = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
        with self.assertRaises(ValidationError):
            OutcomeLearningSignal(**{**good.model_dump(), "causal_claim": True})


class PersistenceTests(unittest.TestCase):
    def test_signal_round_trips_and_is_append_only(self) -> None:
        repo = Repository.in_memory()
        try:
            recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
            signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]
            repo.insert_outcome_learning_signal(signal)

            loaded = repo.get_outcome_learning_signal(signal.signal_id)
            self.assertEqual(loaded.model_dump(), signal.model_dump())

            import sqlite3
            with self.assertRaises(sqlite3.IntegrityError):
                repo.insert_outcome_learning_signal(signal)
            self.assertFalse(hasattr(repo, "update_outcome_learning_signal"))
            self.assertFalse(hasattr(repo, "delete_outcome_learning_signal"))
        finally:
            repo.close()

    def test_persisting_a_signal_does_not_touch_beliefs_or_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.save_belief(_belief("b1"))
            recs, outs = _scenario([("followed", "successful")] * MIN_TRIALS)
            for rec in recs:
                repo.insert_recommendation(rec)
            for outcome in outs:
                repo.insert_recommendation_outcome(outcome)
            signal = analyze_recommendation_outcomes(recs, outs, as_of=AS_OF)[0]

            belief_before = repo.get_latest_belief(user_id="usr_1", belief_id="b1").model_dump_json()
            repo.insert_outcome_learning_signal(signal)

            self.assertEqual(
                repo.get_latest_belief(user_id="usr_1", belief_id="b1").model_dump_json(), belief_before
            )
            self.assertEqual(repo.list_all_evidence(user_id="usr_1", belief_id="b1"), [])
        finally:
            repo.close()


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class LearnFromOutcomesCliTests(unittest.TestCase):
    def _seed(self, db_path: str, specs: list[tuple[str, str]]) -> None:
        repo = Repository.at_path(db_path)
        try:
            repo.save_belief(_belief("b1"))
            recs, outs = _scenario(specs)
            for rec in recs:
                repo.insert_recommendation(rec)
            for outcome in outs:
                repo.insert_recommendation_outcome(outcome)
        finally:
            repo.close()

    def test_prints_proposals_and_does_not_modify_db_without_persist(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [("followed", "successful")] * (MIN_TRIALS + 1))
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path])
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(len(printed), 1)
            self.assertEqual(printed[0]["kind"], "support")
            self.assertTrue(printed[0]["proposed_evidence"])

            self.assertEqual(Path(db_path).read_bytes(), before)
            repo = Repository.at_path(db_path)
            try:
                self.assertEqual(repo.list_outcome_learning_signals(), [])
            finally:
                repo.close()

    def test_persist_appends_signals(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [("followed", "successful")] * (MIN_TRIALS + 1))

            result = _run_cli(["--db", db_path, "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("persisted 1 new signal", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                stored = repo.list_outcome_learning_signals()
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="b1")
            finally:
                repo.close()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].model_version, OUTCOME_LEARNING_VERSION)
            self.assertEqual(belief.confidence, 0.7)  # unchanged -- no recompute

    def test_below_threshold_prints_empty_and_persists_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            self._seed(db_path, [("followed", "successful")] * (MIN_TRIALS - 1))

            result = _run_cli(["--db", db_path, "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [])

            repo = Repository.at_path(db_path)
            try:
                self.assertEqual(repo.list_outcome_learning_signals(), [])
            finally:
                repo.close()


if __name__ == "__main__":
    unittest.main()

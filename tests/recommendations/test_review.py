"""Tests for the manual review workflow for outcome-learning signal promotion
(src/recommendations/review.py + tools/review_outcome_learning_signal.py, plus
the review status the viewer/inspector surface).

Proves the seven guarantees the task requires:
1. a rejected review stores the decision and promotes nothing;
2. an approved review without --promote stores the decision only;
3. an approved review with --promote creates belief_evidence;
4. an approved review with --promote --recompute recomputes the belief;
5. a duplicate approval is blocked unless explicitly allowed (and then clearly
   versioned as a second, separately-identified review);
6. the viewer and inspector show each signal's review decision / status;
7. an LLM cannot set reviewer-only fields through any proposal object.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from src.recommendations.models import (
    OutcomeLearningSignalReviewProposal,
)
from src.recommendations.review import review_outcome_learning_signal, review_status_for
from src.storage.repository import Repository
from src.viewer.user_model_view import collect_view_model, render_html

from tests.recommendations.test_promotion import (
    LEARNED_AT,
    PROMOTED_AT,
    _rps_rows,
    _seed_model,
)

_REVIEW_CLI = Path(__file__).resolve().parents[2] / "tools" / "review_outcome_learning_signal.py"
_INSPECT_CLI = Path(__file__).resolve().parents[2] / "tools" / "inspect_user_model.py"
REVIEWED_AT = PROMOTED_AT + timedelta(days=1)


def _run(cli: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(cli), *args], capture_output=True, text=True)


class RejectedReviewTests(unittest.TestCase):
    def test_rejected_review_stores_the_decision_and_promotes_nothing(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            outcome = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="rejected", as_of=REVIEWED_AT, notes="self-report only",
            )
            self.assertTrue(outcome.stored)
            self.assertIsNone(outcome.promotion)
            self.assertEqual(outcome.review.decision.value, "rejected")
            self.assertFalse(outcome.review.promotion_requested)
            self.assertEqual(outcome.review.promoted_evidence_ids, [])

            stored = repo.list_outcome_learning_signal_reviews(signal_id=signal.signal_id)
            self.assertEqual([r.review_id for r in stored], ["rev_1"])
            self.assertEqual(_rps_rows(repo, user_id="usr_1", belief_id="bel_1"), [])
            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            self.assertFalse(latest.locked_until_recompute)
        finally:
            repo.close()

    def test_promote_flag_with_a_rejection_is_refused(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            with self.assertRaises(ValueError):
                review_outcome_learning_signal(
                    repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                    decision="rejected", as_of=REVIEWED_AT, promote=True,
                )
            self.assertEqual(repo.list_outcome_learning_signal_reviews(), [])
        finally:
            repo.close()


class ApprovedWithoutPromoteTests(unittest.TestCase):
    def test_approved_without_promote_stores_the_decision_only(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            outcome = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="approved", as_of=REVIEWED_AT, notes="looks fine, hold promotion",
            )
            self.assertTrue(outcome.stored)
            self.assertIsNone(outcome.promotion)
            self.assertFalse(outcome.review.promotion_requested)
            self.assertFalse(outcome.review.recompute_requested)

            self.assertEqual(_rps_rows(repo, user_id="usr_1", belief_id="bel_1"), [])
            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            self.assertFalse(latest.locked_until_recompute)
            self.assertEqual(
                review_status_for(
                    repo.list_outcome_learning_signal_reviews(signal_id=signal.signal_id)
                ),
                "approved",
            )
        finally:
            repo.close()


class ApprovedWithPromoteTests(unittest.TestCase):
    def test_approved_with_promote_creates_evidence_and_locks_the_belief(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            outcome = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="approved", as_of=PROMOTED_AT, promote=True,
            )
            self.assertTrue(outcome.stored)
            self.assertIsNotNone(outcome.promotion)
            self.assertTrue(outcome.promotion.persisted)
            self.assertFalse(outcome.promotion.recomputed)

            rps = _rps_rows(repo, user_id="usr_1", belief_id="bel_1")
            self.assertEqual(len(rps), 1)
            self.assertEqual(
                outcome.review.promoted_evidence_ids, [row.evidence_id for row in rps]
            )
            self.assertTrue(outcome.review.promotion_requested)
            self.assertFalse(outcome.review.recompute_requested)
            self.assertEqual(outcome.review.recomputed_belief_ids, [])

            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            self.assertTrue(latest.locked_until_recompute)
        finally:
            repo.close()


class ApprovedWithPromoteAndRecomputeTests(unittest.TestCase):
    def test_approved_with_promote_and_recompute_recomputes_the_belief(self) -> None:
        repo = Repository.in_memory()
        try:
            baseline, signal = _seed_model(repo)
            outcome = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="approved", as_of=PROMOTED_AT, promote=True, recompute=True,
            )
            self.assertTrue(outcome.promotion.recomputed)
            self.assertTrue(outcome.review.recompute_requested)
            self.assertEqual(outcome.review.recomputed_belief_ids, ["bel_1"])

            latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            self.assertFalse(latest.locked_until_recompute)
            self.assertGreater(latest.confidence, baseline.confidence)
        finally:
            repo.close()

    def test_recompute_without_promote_is_refused(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            with self.assertRaises(ValueError):
                review_outcome_learning_signal(
                    repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                    decision="approved", as_of=PROMOTED_AT, recompute=True,
                )
        finally:
            repo.close()


class DuplicateReviewTests(unittest.TestCase):
    def test_second_approval_is_blocked_unless_explicitly_allowed(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            first = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="approved", as_of=REVIEWED_AT,
            )
            self.assertTrue(first.stored)

            blocked = review_outcome_learning_signal(
                repo, review_id="rev_2", signal_id=signal.signal_id, reviewer_id="bob",
                decision="approved", as_of=REVIEWED_AT + timedelta(days=1),
            )
            self.assertFalse(blocked.stored)
            self.assertIn("already has an approved review", blocked.blocked_reason)
            self.assertEqual(
                [r.review_id for r in repo.list_outcome_learning_signal_reviews()], ["rev_1"]
            )

            allowed = review_outcome_learning_signal(
                repo, review_id="rev_3", signal_id=signal.signal_id, reviewer_id="bob",
                decision="approved", as_of=REVIEWED_AT + timedelta(days=2),
                allow_duplicate=True,
            )
            self.assertTrue(allowed.stored)
            trail = repo.list_outcome_learning_signal_reviews(signal_id=signal.signal_id)
            self.assertEqual([r.review_id for r in trail], ["rev_1", "rev_3"])
            # each review is separately identified and timestamped
            self.assertLess(trail[0].created_at, trail[1].created_at)
        finally:
            repo.close()

    def test_reusing_a_review_id_is_rejected(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="rejected", as_of=REVIEWED_AT,
            )
            with self.assertRaises(ValueError):
                review_outcome_learning_signal(
                    repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="bob",
                    decision="approved", as_of=REVIEWED_AT,
                )
        finally:
            repo.close()


class ViewerAndInspectorShowReviewStatusTests(unittest.TestCase):
    def test_viewer_view_model_and_html_carry_the_review_decision(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="rejected", as_of=REVIEWED_AT, notes="not yet",
            )
            view_model = collect_view_model(repo, db_path=":memory:")
        finally:
            repo.close()

        signal_row = view_model.outcome_learning_signals[0]
        self.assertEqual(signal_row["review_status"], "rejected")
        self.assertEqual(signal_row["review_count"], 1)
        self.assertEqual(signal_row["review_decisions"][0]["reviewer_id"], "alice")

        self.assertEqual(len(view_model.outcome_learning_signal_reviews), 1)
        self.assertEqual(
            view_model.outcome_learning_signal_reviews[0]["decision"], "rejected"
        )
        self.assertEqual(
            view_model.summary()["outcome_learning_signals_by_review_status"],
            {"rejected": 1},
        )

        html = render_html(view_model)
        self.assertIn("Outcome learning signal reviews", html)
        self.assertIn("rev_1", html)
        self.assertIn(">rejected<", html)

    def test_inspector_json_reports_review_status_per_signal(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                _, signal = _seed_model(repo)
                review_outcome_learning_signal(
                    repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                    decision="approved", as_of=PROMOTED_AT, promote=True,
                )
            finally:
                repo.close()

            result = _run(_INSPECT_CLI, ["--db", db_path, "--pretty"])
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

        self.assertEqual(payload["outcome_learning_signals"][0]["review_status"], "approved")
        self.assertEqual(payload["outcome_learning_signals"][0]["review_count"], 1)
        self.assertEqual(
            payload["outcome_learning_signals"][0]["reviews"][0]["reviewer_id"], "alice"
        )
        self.assertEqual(len(payload["outcome_learning_signal_reviews"]), 1)
        self.assertEqual(
            payload["outcome_learning_signal_reviews"][0]["decision"], "approved"
        )

    def test_inspector_does_not_leak_another_users_reviews(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                _, signal_a = _seed_model(repo, user_id="usr_a", belief_id="bel_a", event_id="evt_a")
                _, signal_b = _seed_model(repo, user_id="usr_b", belief_id="bel_b", event_id="evt_b")
                review_outcome_learning_signal(
                    repo, review_id="rev_a", signal_id=signal_a.signal_id, reviewer_id="alice",
                    decision="rejected", as_of=REVIEWED_AT,
                )
                review_outcome_learning_signal(
                    repo, review_id="rev_b", signal_id=signal_b.signal_id, reviewer_id="bob",
                    decision="rejected", as_of=REVIEWED_AT,
                )
            finally:
                repo.close()

            result = _run(_INSPECT_CLI, ["--db", db_path, "--user-id", "usr_a", "--pretty"])
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

        review_ids = {r["review_id"] for r in payload["outcome_learning_signal_reviews"]}
        self.assertEqual(review_ids, {"rev_a"})


class ReviewCliTests(unittest.TestCase):
    def _seed(self, db_path: str) -> str:
        repo = Repository.at_path(db_path)
        try:
            _, signal = _seed_model(repo)
        finally:
            repo.close()
        return signal.signal_id

    def test_cli_rejection_stores_and_promotes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = self._seed(db_path)
            result = _run(_REVIEW_CLI, [
                "--db", db_path, "--signal-id", signal_id, "--review-id", "rev_1",
                "--reviewer-id", "alice", "--decision", "rejected", "--notes", "no",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["stored"])
            self.assertNotIn("promotion", payload)
            repo = Repository.at_path(db_path)
            try:
                self.assertEqual(_rps_rows(repo, user_id="usr_1", belief_id="bel_1"), [])
            finally:
                repo.close()

    def test_cli_approved_with_promote_and_recompute(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = self._seed(db_path)
            result = _run(_REVIEW_CLI, [
                "--db", db_path, "--signal-id", signal_id, "--review-id", "rev_1",
                "--reviewer-id", "alice", "--decision", "approved", "--promote", "--recompute",
                "--as-of", PROMOTED_AT.isoformat(),
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["review"]["recomputed_belief_ids"], ["bel_1"])
            repo = Repository.at_path(db_path)
            try:
                latest = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
                self.assertFalse(latest.locked_until_recompute)
                self.assertEqual(len(_rps_rows(repo, user_id="usr_1", belief_id="bel_1")), 1)
            finally:
                repo.close()

    def test_cli_recompute_requires_promote(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = self._seed(db_path)
            result = _run(_REVIEW_CLI, [
                "--db", db_path, "--signal-id", signal_id, "--review-id", "rev_1",
                "--reviewer-id", "alice", "--decision", "approved", "--recompute",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--recompute requires --promote", result.stderr)

    def test_cli_promote_with_rejection_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = self._seed(db_path)
            result = _run(_REVIEW_CLI, [
                "--db", db_path, "--signal-id", signal_id, "--review-id", "rev_1",
                "--reviewer-id", "alice", "--decision", "rejected", "--promote",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rejected", result.stderr)

    def test_cli_missing_db_exits_nonzero_without_creating_it(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.sqlite3"
            result = _run(_REVIEW_CLI, [
                "--db", str(missing), "--signal-id", "x", "--review-id", "rev_1",
                "--reviewer-id", "alice", "--decision", "approved",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no such database file", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(missing.exists())

    def test_cli_duplicate_approval_blocked_then_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            signal_id = self._seed(db_path)
            base = [
                "--db", db_path, "--signal-id", signal_id, "--reviewer-id", "alice",
                "--decision", "approved",
            ]
            first = _run(_REVIEW_CLI, [*base, "--review-id", "rev_1"])
            self.assertEqual(first.returncode, 0, first.stderr)

            blocked = _run(_REVIEW_CLI, [*base, "--review-id", "rev_2"])
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("already has an approved review", blocked.stdout + blocked.stderr)

            allowed = _run(_REVIEW_CLI, [*base, "--review-id", "rev_3", "--allow-duplicate"])
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            repo = Repository.at_path(db_path)
            try:
                trail = repo.list_outcome_learning_signal_reviews(signal_id=signal_id)
            finally:
                repo.close()
            self.assertEqual([r.review_id for r in trail], ["rev_1", "rev_3"])


class LlmCannotSetReviewerFieldsTests(unittest.TestCase):
    def test_proposal_object_has_no_reviewer_only_fields(self) -> None:
        allowed = set(OutcomeLearningSignalReviewProposal.model_fields)
        # the reviewer-owned decision surface is absent from the proposal
        for banned in ("decision", "reviewer_id", "review_id", "promotion_requested",
                       "recompute_requested", "promoted_evidence_ids", "recomputed_belief_ids"):
            self.assertNotIn(banned, allowed)

    def test_proposal_rejects_smuggled_reviewer_fields(self) -> None:
        for banned in ("decision", "reviewer_id", "review_id", "promotion_requested"):
            with self.assertRaises(ValidationError):
                OutcomeLearningSignalReviewProposal(
                    signal_id="ols-x",
                    **{banned: "approved" if banned == "decision" else "x"},
                    schema_version="6", scoring_version="belief-score-0.6",
                    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
                )

    def test_a_model_drafted_proposal_only_supplies_suggested_notes(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo)
            proposal = OutcomeLearningSignalReviewProposal(
                signal_id=signal.signal_id, suggested_notes="model thinks trials look positive",
                schema_version="6", scoring_version="belief-score-0.6",
                canonicalizer_version="canon-0.6", policy_version="policy-0.6",
            )
            # the human reviewer still supplies decision + identity via the call
            outcome = review_outcome_learning_signal(
                repo, review_id="rev_1", signal_id=signal.signal_id, reviewer_id="alice",
                decision="approved", as_of=REVIEWED_AT, proposal=proposal,
            )
            self.assertEqual(outcome.review.reviewer_id, "alice")
            self.assertEqual(outcome.review.notes, "model thinks trials look positive")
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()

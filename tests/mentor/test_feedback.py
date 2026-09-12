"""Tests for src/mentor/feedback.py -- the first mentor feedback engine.

Covers the required behaviours:

- a strong grounded belief -> cautious, useful, grounded feedback
- no evidence / no beliefs -> needs_more_data
- contradictory evidence -> needs_more_data
- a locked belief -> not used
- a rejected / outdated belief -> not used
- a high-risk context -> no directive advice, capped confidence
- every feedback item carries confidence + grounded belief ids
- feedback never contains unsupported / high-stakes claims
- generation is read-only (the database is not mutated)
"""
from __future__ import annotations

import unittest

from src.common.enums import BeliefStatus
from src.mentor.feedback import (
    _HIGH_STAKES_PATTERNS,
    generate_mentor_feedback,
)
from src.storage.repository import Repository

from tests.mentor._helpers import AS_OF, VERSION_FIELDS, seed_belief


def _invalidate_all_and_recompute(repo: Repository, *, user_id: str, belief_id: str, status: BeliefStatus) -> None:
    """Turn a live belief into a rejected / outdated one the honest way:
    invalidate its evidence, then recompute through the invalidation branch."""
    from src.beliefs.recompute import recompute_belief

    for row in repo.list_evidence(user_id=user_id, belief_id=belief_id):
        repo.mark_evidence_inactive(row.evidence_id, reason="policy invalidation", invalidated_at=AS_OF)
    belief = recompute_belief(
        belief_id=belief_id, user_id=user_id, belief_type="behavioral_tendency",
        belief_key="higher_adherence_after_work", belief_value=True,
        evidence=repo.list_active_evidence(user_id=user_id, belief_id=belief_id),
        as_of=AS_OF, first_observed=AS_OF, recompute_reason="policy_invalidation",
        no_active_evidence_status=status, **VERSION_FIELDS,
    )
    repo.save_belief(belief)


class StrongGroundedBeliefTests(unittest.TestCase):
    def test_produces_cautious_grounded_feedback(self) -> None:
        repo = Repository.in_memory()
        try:
            belief = seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.feedback), 1)
        item = result.feedback[0]
        self.assertEqual(item.grounded_in_belief_ids, ["b1"])
        self.assertGreater(item.confidence, 0.0)
        self.assertLessEqual(item.confidence, 0.70)  # medium-tier cap (no context)
        self.assertIn("b1", item.why)
        self.assertIn(belief.status.value, item.why)
        self.assertIn("higher adherence after work", item.message)
        self.assertIn(item.risk_tier, ("low", "medium", "high"))
        self.assertIsNone(result.needs_more_data_reason)

    def test_low_risk_context_allows_one_small_next_action(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            result = generate_mentor_feedback(repo, user_id="u1", context_key="fitness_scheduling")
        finally:
            repo.close()
        item = result.feedback[0]
        self.assertEqual(item.risk_tier, "low")
        self.assertIsNotNone(item.recommended_next_action)
        self.assertIn("small", item.recommended_next_action.lower())


class NeedsMoreDataTests(unittest.TestCase):
    def test_no_beliefs_at_all(self) -> None:
        repo = Repository.in_memory()
        try:
            result = generate_mentor_feedback(repo, user_id="nobody")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertEqual(result.feedback, [])
        self.assertIn("no beliefs", result.needs_more_data_reason)

    def test_belief_with_no_active_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            # a provisional belief, then every evidence row invalidated but
            # the belief left provisional (stale cache) -- no active support.
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            for row in repo.list_evidence(user_id="u1", belief_id="b1"):
                repo.mark_evidence_inactive(row.evidence_id, reason="x", invalidated_at=AS_OF)
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        # the belief was auto-locked by the invalidation, so it fails the
        # lifecycle gate before the evidence gate is even reached.
        self.assertIsNotNone(result.needs_more_data_reason)

    def test_contradictory_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3, contradict=3)
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertEqual(result.feedback, [])

    def test_unknown_context_blocks_everything(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            result = generate_mentor_feedback(repo, user_id="u1", context_key="totally_unknown_zzz")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertIn("context", result.needs_more_data_reason)


class ExcludedBeliefTests(unittest.TestCase):
    def test_locked_belief_is_not_used(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            repo.lock_belief_until_recompute(user_id="u1", belief_id="b1")
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertIn("locked", result.needs_more_data_reason)

    def test_locked_belief_is_skipped_when_another_is_usable(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b_ok", support=3,
                        belief_key="higher_adherence_after_work")
            seed_belief(repo, user_id="u1", belief_id="b_locked", support=3,
                        belief_key="prefers_evening_exercise_sessions")
            repo.lock_belief_until_recompute(user_id="u1", belief_id="b_locked")
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "ready")
        used = {bid for item in result.feedback for bid in item.grounded_in_belief_ids}
        self.assertEqual(used, {"b_ok"})

    def test_rejected_belief_is_not_used(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            _invalidate_all_and_recompute(repo, user_id="u1", belief_id="b1", status=BeliefStatus.REJECTED)
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertEqual(result.feedback, [])

    def test_outdated_belief_is_not_used(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            _invalidate_all_and_recompute(repo, user_id="u1", belief_id="b1", status=BeliefStatus.OUTDATED)
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertEqual(result.feedback, [])

    def test_candidate_belief_is_not_used(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=1)  # -> candidate
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")


class HighRiskContextTests(unittest.TestCase):
    def _validated_belief_repo(self) -> Repository:
        repo = Repository.in_memory()
        seed_belief(
            repo, user_id="u1", belief_id="b1", support=8, strength=0.95,
            belief_type="routine_or_preference", belief_key="prefers_short_focused_sessions",
        )
        return repo

    def test_no_directive_advice_in_a_high_risk_context(self) -> None:
        repo = self._validated_belief_repo()
        try:
            result = generate_mentor_feedback(repo, user_id="u1", context_key="mental_health_support")
        finally:
            repo.close()
        self.assertEqual(result.status, "ready")
        for item in result.feedback:
            self.assertEqual(item.risk_tier, "high")
            self.assertIsNone(item.recommended_next_action)
            self.assertLessEqual(item.confidence, 0.55)
            lowered = item.message.lower()
            for directive in ("you should", "you must", "do this", "start ", "quit "):
                self.assertNotIn(directive, lowered)

    def test_medium_risk_context_stays_reflective(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(
                repo, user_id="u1", belief_id="b1", support=6, strength=0.9,
                belief_type="routine_or_preference", belief_key="prefers_short_focused_sessions",
            )
            result = generate_mentor_feedback(repo, user_id="u1", context_key="nutrition_guidance")
        finally:
            repo.close()
        self.assertEqual(result.status, "ready")
        item = result.feedback[0]
        self.assertEqual(item.risk_tier, "medium")
        self.assertLessEqual(item.confidence, 0.70)
        # medium may carry a soft prompt, but not a concrete directive
        if item.recommended_next_action is not None:
            self.assertNotIn("consider one small", item.recommended_next_action.lower())


class OutputContractTests(unittest.TestCase):
    def test_every_item_carries_confidence_and_grounded_ids(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b1", support=3,
                        belief_key="higher_adherence_after_work")
            seed_belief(repo, user_id="u1", belief_id="b2", support=4, strength=0.92,
                        belief_key="responds_to_streak_framing",
                        belief_type="recommendation_response_pattern")
            result = generate_mentor_feedback(repo, user_id="u1")
            stored_ids = {b.belief_id for b in repo.list_latest_beliefs(user_id="u1")}
        finally:
            repo.close()

        self.assertEqual(result.status, "ready")
        self.assertTrue(result.feedback)
        for item in result.feedback:
            self.assertIsInstance(item.confidence, float)
            self.assertGreater(item.confidence, 0.0)
            self.assertLess(item.confidence, 0.98)
            self.assertTrue(item.grounded_in_belief_ids)
            for bid in item.grounded_in_belief_ids:
                self.assertIn(bid, stored_ids)  # never a fabricated id
            self.assertTrue(item.why.strip())

    def test_feedback_never_contains_high_stakes_advice(self) -> None:
        repo = Repository.in_memory()
        try:
            # a belief_key that itself mentions a high-stakes topic must not
            # produce a feedback item -- the item is dropped, not emitted.
            seed_belief(repo, user_id="u1", belief_id="b_ok", support=3,
                        belief_key="higher_adherence_after_work")
            seed_belief(repo, user_id="u1", belief_id="b_risky", support=4, strength=0.92,
                        belief_key="wants_to_quit_your_job_and_go_to_college",
                        belief_type="goal_or_intention")
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()

        self.assertEqual(result.status, "ready")
        used = {bid for item in result.feedback for bid in item.grounded_in_belief_ids}
        self.assertNotIn("b_risky", used)
        for item in result.feedback:
            haystack = f"{item.message} {item.why} {item.recommended_next_action or ''}".lower()
            for pattern in _HIGH_STAKES_PATTERNS:
                self.assertNotIn(pattern, haystack)

    def test_all_high_stakes_beliefs_collapse_to_needs_more_data(self) -> None:
        repo = Repository.in_memory()
        try:
            seed_belief(repo, user_id="u1", belief_id="b_risky", support=4, strength=0.92,
                        belief_key="should_take_out_a_loan_for_a_degree",
                        belief_type="goal_or_intention")
            result = generate_mentor_feedback(repo, user_id="u1")
        finally:
            repo.close()
        self.assertEqual(result.status, "needs_more_data")
        self.assertIn("high-stakes", result.needs_more_data_reason)


class ReadOnlyTests(unittest.TestCase):
    def test_generation_does_not_mutate_the_database(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "mentor.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                seed_belief(repo, user_id="u1", belief_id="b1", support=3)
            finally:
                repo.close()

            before = Path(db_path).read_bytes()
            readonly = Repository.readonly_at_path(db_path)
            try:
                result = generate_mentor_feedback(readonly, user_id="u1")
                # run it a few times through different paths
                generate_mentor_feedback(readonly, user_id="u1", context_key="fitness_scheduling")
                generate_mentor_feedback(readonly, user_id="u1", context_key="mental_health_support")
            finally:
                readonly.close()

            self.assertEqual(result.status, "ready")
            self.assertEqual(Path(db_path).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

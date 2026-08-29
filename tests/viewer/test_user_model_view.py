"""Tests for src/viewer/user_model_view.py -- the pure data-shaping and
HTML-rendering helpers behind the read-only user-model viewer.

These cover the pure functions (``evidence_state``, the ``*_row`` projections,
``summarize``) and ``collect_view_model``/``render_html`` at a smoke level.
The viewer's browser/UI behaviour is deliberately not tested here.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.beliefs.canonicalization import (
    BeliefKeyCanonicalizationProposal,
    authorize_belief_key_canonicalization,
)
from src.beliefs.models import authorize_evidence, invalidate_evidence, suppress_duplicate_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.storage.repository import Repository
from src.viewer.user_model_view import (
    EVIDENCE_STATE_ACTIVE,
    EVIDENCE_STATE_DUPLICATE_SUPPRESSED,
    EVIDENCE_STATE_INACTIVE,
    belief_row,
    canonicalization_row,
    collect_view_model,
    evidence_row,
    evidence_state,
    observation_event_row,
    render_html,
    summarize,
)

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

USER_ID = "usr_17"
BELIEF_ID = "bel_88"
BELIEF_TYPE = "behavioral_tendency"
BELIEF_KEY = "higher_adherence_after_work"
AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


def _event(event_id: str, user_id: str = USER_ID) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=AS_OF - timedelta(days=1), source="app", **VERSION_FIELDS,
    )


def _make_evidence(
    evidence_id: str, event_id: str, *, user_id: str = USER_ID, belief_id: str = BELIEF_ID,
    event: UserEvent | None = None,
):
    event = event or _event(event_id, user_id=user_id)
    observation, links = create_observation_from_event(
        event, observation_id=f"obs_{evidence_id}", category="routine",
        observation_text="did an after-work workout", importance=0.6, confidence=0.6,
        created_at=event.timestamp, **VERSION_FIELDS,
    )
    proposal = propose_evidence_from_observation_validated(
        observation, links, [event], belief_id=belief_id, direction="support",
        source_type="recorded_event", context_key="fitness", strength=0.9,
        model_version="test-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
    )
    return event, observation, links, authorize_evidence(
        proposal, evidence_id=evidence_id, created_at=event.timestamp,
        aggregation_policy_version="evidence-aggregation-0.6",
    )


class EvidenceStateTests(unittest.TestCase):
    def test_a_plain_evidence_row_is_active(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        self.assertEqual(evidence_state(evidence), EVIDENCE_STATE_ACTIVE)

    def test_an_invalidated_row_is_inactive(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        invalidated = invalidate_evidence(evidence, reason="deletion", invalidated_at=AS_OF)
        self.assertEqual(evidence_state(invalidated), EVIDENCE_STATE_INACTIVE)

    def test_a_live_duplicate_suppressed_row_is_duplicate_suppressed(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        suppressed = suppress_duplicate_evidence(evidence, reason="duplicate_evidence")
        self.assertEqual(evidence_state(suppressed), EVIDENCE_STATE_DUPLICATE_SUPPRESSED)

    def test_inactive_takes_precedence_over_duplicate_suppressed(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        both = invalidate_evidence(
            suppress_duplicate_evidence(evidence, reason="duplicate_evidence"),
            reason="deletion", invalidated_at=AS_OF,
        )
        self.assertEqual(evidence_state(both), EVIDENCE_STATE_INACTIVE)


class RowProjectionTests(unittest.TestCase):
    def test_belief_row_surfaces_the_highlighted_fields(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=[evidence], as_of=AS_OF, first_observed=AS_OF - timedelta(days=1),
            **VERSION_FIELDS,
        )
        row = belief_row(belief)
        self.assertEqual(row["belief_key"], BELIEF_KEY)
        self.assertEqual(row["confidence"], belief.confidence)
        self.assertEqual(row["status"], belief.status.value)
        self.assertEqual(row["locked_until_recompute"], belief.locked_until_recompute)
        for count_field in (
            "supporting_evidence_count", "contradicting_evidence_count",
            "total_evidence_count", "effective_evidence_count",
        ):
            self.assertIn(count_field, row)

    def test_evidence_row_reports_state_and_a_reason(self) -> None:
        _, _, _, evidence = _make_evidence("bev_1", "evt_1")
        invalidated = invalidate_evidence(evidence, reason="policy_reset", invalidated_at=AS_OF)
        row = evidence_row(invalidated)
        self.assertEqual(row["state"], EVIDENCE_STATE_INACTIVE)
        self.assertEqual(row["invalidation_reason"], "policy_reset")
        self.assertFalse(row["is_active"])

    def test_canonicalization_row_shows_proposed_vs_canonical_and_decision(self) -> None:
        proposal = BeliefKeyCanonicalizationProposal(
            user_id=USER_ID, belief_type=BELIEF_TYPE,
            proposed_key="prefers_evening_exercise_sessions", **VERSION_FIELDS,
        )
        decision = authorize_belief_key_canonicalization(
            proposal, canonicalization_id="can_1", authorized_at=AS_OF,
        )
        row = canonicalization_row(decision)
        self.assertEqual(row["proposed_key"], "prefers_evening_exercise_sessions")
        self.assertEqual(row["canonical_key"], "higher_adherence_after_work")
        self.assertEqual(row["decision"], "alias")
        self.assertIn("registry", row["decision_reason"])

    def test_observation_event_row_projects_the_link(self) -> None:
        _, observation, links, _ = _make_evidence("bev_1", "evt_1")
        row = observation_event_row(links[0])
        self.assertEqual(row["observation_id"], observation.observation_id)
        self.assertEqual(row["event_id"], "evt_1")
        self.assertEqual(row["link_role"], links[0].link_role.value)


class SummarizeTests(unittest.TestCase):
    def test_counts_by_evidence_state_status_and_decision(self) -> None:
        repo = _seeded_repo()
        try:
            view_model = collect_view_model(repo, db_path=":memory:")
        finally:
            repo.close()
        summary = summarize(view_model)

        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["observations"], 3)
        self.assertEqual(summary["observation_events"], 3)
        self.assertEqual(summary["evidence"], 3)
        self.assertEqual(summary["evidence_by_state"][EVIDENCE_STATE_ACTIVE], 1)
        self.assertEqual(summary["evidence_by_state"][EVIDENCE_STATE_INACTIVE], 1)
        self.assertEqual(summary["evidence_by_state"][EVIDENCE_STATE_DUPLICATE_SUPPRESSED], 1)
        self.assertEqual(summary["beliefs"], 1)
        self.assertEqual(summary["locked_beliefs"], 1)
        self.assertEqual(summary["canonicalizations"], 1)
        self.assertEqual(summary["canonicalizations_by_decision"]["alias"], 1)


class CollectViewModelTests(unittest.TestCase):
    def test_respects_user_scope(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_event("evt_a", user_id="usr_1"))
            repo.insert_event(_event("evt_b", user_id="usr_2"))
            scoped = collect_view_model(repo, db_path=":memory:", user_id="usr_1")
        finally:
            repo.close()
        self.assertEqual([r["event_id"] for r in scoped.events], ["evt_a"])

    def test_observation_rows_carry_their_linked_event_ids(self) -> None:
        repo = Repository.in_memory()
        try:
            event, observation, links, _ = _make_evidence("bev_1", "evt_1")
            repo.insert_event(event)
            repo.insert_observation(observation, links)
            view_model = collect_view_model(repo, db_path=":memory:")
        finally:
            repo.close()
        self.assertEqual(view_model.observations[0]["linked_event_ids"], ["evt_1"])
        self.assertEqual(
            [(l["observation_id"], l["event_id"]) for l in view_model.observation_events],
            [(observation.observation_id, "evt_1")],
        )


class RenderHtmlTests(unittest.TestCase):
    def test_renders_a_full_page_and_escapes_untrusted_text(self) -> None:
        repo = Repository.in_memory()
        try:
            event = _event("evt_1")
            repo.insert_event(event)
            observation, links = create_observation_from_event(
                event, observation_id="obs_1", category="routine",
                observation_text="<script>alert('xss')</script>", importance=0.5, confidence=0.5,
                created_at=event.timestamp, **VERSION_FIELDS,
            )
            repo.insert_observation(observation, links)
            view_model = collect_view_model(repo, db_path="demo.sqlite3")
        finally:
            repo.close()

        page = render_html(view_model)
        self.assertTrue(page.lstrip().startswith("<!doctype html>"))
        self.assertIn("READ-ONLY", page)
        self.assertIn("demo.sqlite3", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>alert", page)

    def test_empty_database_still_renders_every_section(self) -> None:
        repo = Repository.in_memory()
        try:
            view_model = collect_view_model(repo, db_path="empty.sqlite3")
        finally:
            repo.close()
        page = render_html(view_model)
        for heading in (
            "Events", "Observations", "Observation-event links", "Evidence", "Beliefs",
            "Belief-key canonicalization",
        ):
            self.assertIn(heading, page)
        self.assertIn("No events stored.", page)
        self.assertIn("No observation-event links stored.", page)


def _seeded_repo() -> Repository:
    """One user with: an active evidence row, an invalidated one, a
    duplicate-suppressed one, a locked belief, and one canonicalization
    decision."""
    repo = Repository.in_memory()

    shared_event = _event("evt_1")
    repo.insert_event(shared_event)
    gone_event = _event("evt_2")
    repo.insert_event(gone_event)

    # bev_active and bev_dupe both trace to evt_1 through different
    # observations -> find_duplicate_evidence() flags the later one.
    for name, event in (("bev_active", shared_event), ("bev_dupe", shared_event), ("bev_gone", gone_event)):
        _, observation, links, evidence = _make_evidence(name, event.event_id, event=event)
        repo.insert_observation(observation, links)
        repo.insert_evidence(evidence)

    belief = recompute_belief(
        belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
        belief_value=True, evidence=repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
        as_of=AS_OF, first_observed=AS_OF - timedelta(days=1), **VERSION_FIELDS,
    )
    repo.save_belief(belief)

    # Invalidating one row fail-closes (locks) the belief -- exactly the
    # state the viewer needs to display distinctly.
    repo.mark_evidence_inactive("bev_gone", reason="deletion", invalidated_at=AS_OF)
    repo.suppress_duplicate_evidence(user_id=USER_ID, belief_id=BELIEF_ID, reason="duplicate_evidence")

    proposal = BeliefKeyCanonicalizationProposal(
        user_id=USER_ID, belief_type=BELIEF_TYPE,
        proposed_key="prefers_evening_exercise_sessions", **VERSION_FIELDS,
    )
    repo.insert_belief_key_canonicalization(
        authorize_belief_key_canonicalization(proposal, canonicalization_id="can_1", authorized_at=AS_OF)
    )
    return repo


if __name__ == "__main__":
    unittest.main()

"""Tests for src/storage/repository.py -- the minimal local persistence
layer over the Phase 1-3 records.

Covers exactly the five things this phase must prove:
1. event -> observation -> evidence -> recompute -> persisted belief round trip.
2. source_event_ids survives persistence exactly.
3. wrong-user/unknown event provenance validation still fails before save.
4. inactive evidence is excluded when listing active evidence.
5. no stale belief is returned after all evidence is invalidated and recomputed.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.beliefs.models import UserBelief, authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.observations.models import ObservationEvent, UserObservation
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

BELIEF_ID = "bel_88"
USER_ID = "usr_17"
BELIEF_TYPE = "behavioral_tendency"
BELIEF_KEY = "higher_adherence_after_work"
AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


def _event(event_id: str, user_id: str = USER_ID, timestamp: datetime | None = None) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=timestamp or (AS_OF - timedelta(days=1)), source="app", **VERSION_FIELDS,
    )


class RepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Repository.in_memory()

    def tearDown(self) -> None:
        self.repo.close()


class EndToEndRoundTripTests(RepositoryTestCase):
    def test_event_through_observation_through_evidence_through_recompute_through_persisted_belief(self) -> None:
        event = _event("evt_1042")
        self.repo.insert_event(event)

        observation, links = create_observation_from_event(
            event, observation_id="obs_1042", category="routine",
            observation_text="User completed an after-work workout.",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)

        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id="bev_1042", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)

        active = self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID)
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=active, as_of=AS_OF, first_observed=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.save_belief(belief)

        persisted = self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.confidence, belief.confidence)
        self.assertGreater(persisted.confidence, 0.0)
        self.assertEqual(persisted.supporting_evidence_count, 1)


class SourceEventIdsRoundTripTests(RepositoryTestCase):
    def test_source_event_ids_survives_persistence_exactly(self) -> None:
        events = [_event("evt_1040"), _event("evt_1041"), _event("evt_1042")]
        for event in events:
            self.repo.insert_event(event)

        observation, links = create_observation_from_event(
            events[-1], observation_id="obs_310", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=AS_OF, **VERSION_FIELDS,
        )
        # Add two more (supporting) links so source_event_ids has >1 entry
        # and order matters for a meaningful round-trip check.
        links = links + [
            ObservationEvent(
                observation_id="obs_310", event_id="evt_1040", link_role="supporting",
                created_at=AS_OF, **VERSION_FIELDS,
            ),
            ObservationEvent(
                observation_id="obs_310", event_id="evt_1041", link_role="supporting",
                created_at=AS_OF, **VERSION_FIELDS,
            ),
        ]
        self.repo.insert_observation(observation, links)

        proposal = propose_evidence_from_observation_validated(
            observation, links, events, belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        original_source_event_ids = list(proposal.source_event_ids)
        evidence = authorize_evidence(
            proposal, evidence_id="bev_310", created_at=AS_OF,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)

        persisted = self.repo.get_evidence("bev_310")
        self.assertEqual(persisted.source_event_ids, original_source_event_ids)
        self.assertEqual(len(persisted.source_event_ids), 3)

        # Also survives the list_evidence() / list_active_evidence() paths.
        via_list = self.repo.list_evidence(user_id=USER_ID, belief_id=BELIEF_ID)[0]
        self.assertEqual(via_list.source_event_ids, original_source_event_ids)


class ProvenanceValidationStillFailsBeforeSaveTests(RepositoryTestCase):
    def test_observation_linking_an_unknown_event_is_rejected_and_nothing_is_written(self) -> None:
        ghost_event = _event("evt_ghost")  # deliberately never inserted
        observation, links = create_observation_from_event(
            ghost_event, observation_id="obs_bad", category="routine", observation_text="x",
            importance=0.5, confidence=0.5, created_at=AS_OF, **VERSION_FIELDS,
        )
        with self.assertRaises(ValueError):
            self.repo.insert_observation(observation, links)
        self.assertIsNone(self.repo.get_observation("obs_bad"))
        self.assertEqual(self.repo.list_observation_events("obs_bad"), [])

    def test_observation_linking_a_wrong_users_event_is_rejected_and_nothing_is_written(self) -> None:
        event = _event("evt_1042", user_id="usr_OTHER")
        self.repo.insert_event(event)
        _, links = create_observation_from_event(
            event, observation_id="obs_bad2", category="routine", observation_text="x",
            importance=0.5, confidence=0.5, created_at=AS_OF, **VERSION_FIELDS,
        )
        # observation.user_id comes from the event in create_observation_from_event,
        # so force a mismatch by building the observation for a different user directly.
        mismatched_observation = UserObservation(
            observation_id="obs_bad2", user_id=USER_ID, category="routine", observation="x",
            importance=0.5, confidence=0.5, created_at=AS_OF, **VERSION_FIELDS,
        )
        with self.assertRaises(ValueError) as ctx:
            self.repo.insert_observation(mismatched_observation, links)
        self.assertIn("crosses users", str(ctx.exception))
        self.assertIsNone(self.repo.get_observation("obs_bad2"))

    def test_evidence_referencing_an_unknown_event_is_rejected_and_nothing_is_written(self) -> None:
        event = _event("evt_1042")
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id="obs_1042", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id="bev_bad", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        # Retarget the evidence at an event_id that was never inserted.
        retargeted = evidence.model_copy(update={"source_event_ids": ["evt_never_stored"], "event_id": None})
        with self.assertRaises(ValueError) as ctx:
            self.repo.insert_evidence(retargeted)
        self.assertIn("unknown event_id", str(ctx.exception))
        self.assertIsNone(self.repo.get_evidence("bev_bad"))

    def test_evidence_referencing_a_wrong_users_event_is_rejected_and_nothing_is_written(self) -> None:
        event = _event("evt_1042")
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id="obs_1042", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id="bev_bad2", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        # Store a *different* event under the same id but a different user,
        # then evidence claiming that id will now cross users.
        other_repo_event = _event("evt_9999", user_id="usr_OTHER")
        self.repo.insert_event(other_repo_event)
        retargeted = evidence.model_copy(update={"source_event_ids": ["evt_9999"], "event_id": None})
        with self.assertRaises(ValueError) as ctx:
            self.repo.insert_evidence(retargeted)
        self.assertIn("different user", str(ctx.exception))
        self.assertIsNone(self.repo.get_evidence("bev_bad2"))


class InactiveEvidenceExcludedFromActiveListingTests(RepositoryTestCase):
    def _insert_evidence_row(self, evidence_id: str, event_id: str) -> None:
        event = _event(event_id)
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id=f"obs_{event_id}", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id=evidence_id, created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)

    def test_inactive_evidence_is_excluded_from_active_listing_but_not_from_list_evidence(self) -> None:
        self._insert_evidence_row("bev_a", "evt_a")
        self._insert_evidence_row("bev_b", "evt_b")

        self.repo.mark_evidence_inactive("bev_a", reason="reset", invalidated_at=AS_OF)

        active = self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertEqual([row.evidence_id for row in active], ["bev_b"])

        everything = self.repo.list_evidence(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertEqual(sorted(row.evidence_id for row in everything), ["bev_a", "bev_b"])

        invalidated_row = self.repo.get_evidence("bev_a")
        self.assertFalse(invalidated_row.is_active)
        self.assertEqual(invalidated_row.invalidation_reason, "reset")


class InvalidatedEvidenceLocksTheLatestBeliefTests(RepositoryTestCase):
    """The regression case: mark_evidence_inactive() must fail-close the
    belief it backs (blueprint section 6.0.2) so get_latest_belief() cannot
    keep serving a stale, usable-looking confidence in the window between
    invalidation and the next successful recompute."""

    def _save_positive_confidence_belief(self) -> UserBelief:
        event = _event("evt_1042")
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id="obs_1042", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id="bev_1042", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)
        belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
            as_of=AS_OF, first_observed=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.save_belief(belief)
        return belief

    def test_latest_belief_is_locked_after_its_only_evidence_is_invalidated(self) -> None:
        original = self._save_positive_confidence_belief()
        self.assertGreater(original.confidence, 0.0)
        self.assertFalse(original.locked_until_recompute)

        self.repo.mark_evidence_inactive("bev_1042", reason="deletion", invalidated_at=AS_OF)

        before_recompute = self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertIsNotNone(before_recompute)
        # This is the fail-closed signal itself: the belief must now be
        # locked, so no consumer can treat its (still-present) confidence
        # number as current -- it is not usable until a fresh recompute runs.
        self.assertTrue(before_recompute.locked_until_recompute)

    def test_recomputing_and_saving_after_invalidation_unlocks_with_a_new_zero_confidence_belief(self) -> None:
        self._save_positive_confidence_belief()
        self.repo.mark_evidence_inactive("bev_1042", reason="deletion", invalidated_at=AS_OF)
        self.assertTrue(self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID).locked_until_recompute)

        recomputed = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
            as_of=AS_OF, first_observed=AS_OF, recompute_reason="deletion", **VERSION_FIELDS,
        )
        self.repo.save_belief(recomputed)

        latest = self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertEqual(latest.confidence, 0.0)
        self.assertEqual(latest.status.value, "outdated")
        self.assertFalse(latest.locked_until_recompute)

    def test_locking_is_idempotent_and_does_not_spam_the_belief_history(self) -> None:
        self._save_positive_confidence_belief()
        self.repo.mark_evidence_inactive("bev_1042", reason="deletion", invalidated_at=AS_OF)
        # A second invalidation-triggering call against the same belief scope
        # (e.g. another evidence row for the same belief becoming inactive)
        # must not append a second redundant locked copy on top of the first.
        self.repo._lock_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        rows = self.repo._conn.execute(
            "SELECT COUNT(*) FROM user_beliefs WHERE user_id = ? AND belief_id = ?", (USER_ID, BELIEF_ID)
        ).fetchone()
        self.assertEqual(rows[0], 2)  # original + exactly one locked copy


class NoStaleBeliefAfterInvalidationTests(RepositoryTestCase):
    def test_no_stale_belief_is_returned_after_all_evidence_invalidated_and_recomputed(self) -> None:
        event = _event("evt_1042")
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id="obs_1042", category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=BELIEF_ID, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id="bev_1042", created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)

        first_belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
            as_of=AS_OF, first_observed=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.save_belief(first_belief)
        self.assertGreater(self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID).confidence, 0.0)

        self.repo.mark_evidence_inactive("bev_1042", reason="deletion", invalidated_at=AS_OF)
        self.assertEqual(self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID), [])

        # Before any recompute happens, the belief must already be locked --
        # see InvalidatedEvidenceLocksTheLatestBeliefTests for the dedicated
        # coverage of this specific fail-closed behavior.
        stale = self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertTrue(stale.locked_until_recompute)

        second_belief = recompute_belief(
            belief_id=BELIEF_ID, user_id=USER_ID, belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.repo.list_active_evidence(user_id=USER_ID, belief_id=BELIEF_ID),
            as_of=AS_OF, first_observed=event.timestamp, recompute_reason="deletion", **VERSION_FIELDS,
        )
        self.repo.save_belief(second_belief)

        latest = self.repo.get_latest_belief(user_id=USER_ID, belief_id=BELIEF_ID)
        self.assertEqual(latest.confidence, 0.0)
        self.assertEqual(latest.status.value, "outdated")
        # The stale, higher-confidence first_belief must not be what comes back.
        self.assertNotEqual(latest.confidence, first_belief.confidence)


class ReadOnlyListingHelpersTests(RepositoryTestCase):
    """tools/inspect_user_model.py's read-only support: list_events(),
    list_observations(), list_observation_events_for(), list_all_evidence(),
    and list_latest_beliefs() -- the general-purpose, optionally-multi-scope
    listers added for inspection, distinct from the single-scope helpers
    (get_event(), list_evidence(), get_latest_belief(), etc.) every pipeline
    step already uses because it knows exactly which scope it wants."""

    def _seed_one_full_chain(self, *, user_id: str, belief_id: str, event_id: str, observation_id: str,
                              evidence_id: str) -> None:
        event = _event(event_id, user_id=user_id)
        self.repo.insert_event(event)
        observation, links = create_observation_from_event(
            event, observation_id=observation_id, category="routine", observation_text="x",
            importance=0.6, confidence=0.6, created_at=event.timestamp, **VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        proposal = propose_evidence_from_observation_validated(
            observation, links, [event], belief_id=belief_id, direction="support",
            source_type="recorded_event", context_key="fitness", strength=0.9,
            model_version="pipeline-0.1", belief_type=BELIEF_TYPE, **VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal, evidence_id=evidence_id, created_at=event.timestamp,
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.repo.insert_evidence(evidence)

    def test_list_events_returns_all_events_and_can_be_scoped_by_user(self) -> None:
        self.repo.insert_event(_event("evt_1", user_id="usr_1"))
        self.repo.insert_event(_event("evt_2", user_id="usr_2"))

        self.assertEqual({e.event_id for e in self.repo.list_events()}, {"evt_1", "evt_2"})
        self.assertEqual([e.event_id for e in self.repo.list_events(user_id="usr_2")], ["evt_2"])

    def test_list_observations_returns_all_observations_and_can_be_scoped_by_user(self) -> None:
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_1", event_id="evt_1",
            observation_id="obs_1", evidence_id="bev_1",
        )
        self._seed_one_full_chain(
            user_id="usr_2", belief_id="bel_1", event_id="evt_2",
            observation_id="obs_2", evidence_id="bev_2",
        )

        self.assertEqual(
            {o.observation_id for o in self.repo.list_observations()}, {"obs_1", "obs_2"}
        )
        self.assertEqual(
            [o.observation_id for o in self.repo.list_observations(user_id="usr_1")], ["obs_1"]
        )

    def test_list_observation_events_for_returns_links_for_the_given_ids_only(self) -> None:
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_1", event_id="evt_1",
            observation_id="obs_1", evidence_id="bev_1",
        )
        self._seed_one_full_chain(
            user_id="usr_2", belief_id="bel_1", event_id="evt_2",
            observation_id="obs_2", evidence_id="bev_2",
        )

        links = self.repo.list_observation_events_for(["obs_1"])
        self.assertEqual([link.observation_id for link in links], ["obs_1"])
        self.assertEqual(self.repo.list_observation_events_for([]), [])

    def test_list_all_evidence_supports_optional_user_and_belief_filters(self) -> None:
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_1", event_id="evt_1",
            observation_id="obs_1", evidence_id="bev_1",
        )
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_2", event_id="evt_2",
            observation_id="obs_2", evidence_id="bev_2",
        )
        self._seed_one_full_chain(
            user_id="usr_2", belief_id="bel_1", event_id="evt_3",
            observation_id="obs_3", evidence_id="bev_3",
        )

        self.assertEqual(
            {e.evidence_id for e in self.repo.list_all_evidence()}, {"bev_1", "bev_2", "bev_3"}
        )
        self.assertEqual(
            {e.evidence_id for e in self.repo.list_all_evidence(user_id="usr_1")}, {"bev_1", "bev_2"}
        )
        self.assertEqual(
            {e.evidence_id for e in self.repo.list_all_evidence(belief_id="bel_1")}, {"bev_1", "bev_3"}
        )
        self.assertEqual(
            [e.evidence_id for e in self.repo.list_all_evidence(user_id="usr_1", belief_id="bel_1")],
            ["bev_1"],
        )

    def test_list_all_evidence_includes_inactive_rows_unlike_list_active_evidence(self) -> None:
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_1", event_id="evt_1",
            observation_id="obs_1", evidence_id="bev_1",
        )
        self.repo.mark_evidence_inactive("bev_1", reason="deletion", invalidated_at=AS_OF)

        self.assertEqual(self.repo.list_active_evidence(user_id="usr_1", belief_id="bel_1"), [])
        all_evidence = self.repo.list_all_evidence(user_id="usr_1", belief_id="bel_1")
        self.assertEqual(len(all_evidence), 1)
        self.assertFalse(all_evidence[0].is_active)

    def test_list_latest_beliefs_returns_the_newest_row_per_scope_and_supports_filters(self) -> None:
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_1", event_id="evt_1",
            observation_id="obs_1", evidence_id="bev_1",
        )
        self._seed_one_full_chain(
            user_id="usr_1", belief_id="bel_2", event_id="evt_2",
            observation_id="obs_2", evidence_id="bev_2",
        )
        for belief_id in ("bel_1", "bel_2"):
            belief = recompute_belief(
                belief_id=belief_id, user_id="usr_1", belief_type=BELIEF_TYPE,
                belief_key=BELIEF_KEY, belief_value=True,
                evidence=self.repo.list_active_evidence(user_id="usr_1", belief_id=belief_id),
                as_of=AS_OF, first_observed=AS_OF, **VERSION_FIELDS,
            )
            self.repo.save_belief(belief)
        # A second recompute on bel_1 only -- list_latest_beliefs() must
        # return this newer row for bel_1, not the first one saved.
        second_bel_1 = recompute_belief(
            belief_id="bel_1", user_id="usr_1", belief_type=BELIEF_TYPE, belief_key=BELIEF_KEY,
            belief_value=True, evidence=self.repo.list_active_evidence(user_id="usr_1", belief_id="bel_1"),
            as_of=AS_OF + timedelta(days=1), first_observed=AS_OF, **VERSION_FIELDS,
        )
        self.repo.save_belief(second_bel_1)

        all_latest = self.repo.list_latest_beliefs()
        self.assertEqual({(b.belief_id, b.last_validated) for b in all_latest},
                          {("bel_1", second_bel_1.last_validated), ("bel_2", AS_OF)})

        scoped = self.repo.list_latest_beliefs(user_id="usr_1", belief_id="bel_1")
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0].last_validated, second_bel_1.last_validated)

    def test_list_latest_beliefs_returns_empty_list_when_no_belief_saved_for_scope(self) -> None:
        self.assertEqual(self.repo.list_latest_beliefs(user_id="usr_ghost", belief_id="bel_ghost"), [])


class ReadonlyAtPathTests(unittest.TestCase):
    """Repository.readonly_at_path() -- tools/inspect_user_model.py's whole
    read-only guarantee rests on this constructor actually enforcing it at
    the SQLite layer, not merely by the caller's own discipline."""

    def test_missing_path_raises_file_not_found_and_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "does_not_exist.sqlite3"

            with self.assertRaises(FileNotFoundError):
                Repository.readonly_at_path(str(missing_path))

            self.assertFalse(missing_path.exists())

    def test_opens_an_existing_database_and_can_read_from_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            writable = Repository.at_path(db_path)
            writable.insert_event(_event("evt_1"))
            writable.close()

            readonly = Repository.readonly_at_path(db_path)
            try:
                events = readonly.list_events()
            finally:
                readonly.close()
            self.assertEqual([e.event_id for e in events], ["evt_1"])

    def test_write_attempt_is_rejected_at_the_sqlite_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            Repository.at_path(db_path).close()

            readonly = Repository.readonly_at_path(db_path)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    readonly.insert_event(_event("evt_1"))
            finally:
                readonly.close()

            # Confirm nothing was actually written despite the attempt.
            verifying = Repository.at_path(db_path)
            try:
                self.assertIsNone(verifying.get_event("evt_1"))
            finally:
                verifying.close()

    def test_directory_path_is_rejected_like_a_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                Repository.readonly_at_path(tmp)


if __name__ == "__main__":
    unittest.main()

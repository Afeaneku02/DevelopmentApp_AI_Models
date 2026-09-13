"""Tests for the user_observations -> belief_evidence pipeline step
(src/beliefs/process_observations.py + tools/process_observations_to_evidence.py).

Proves the guarantees the task requires:
1. repeated check_in observations map to a conservative routine_or_preference
   belief_evidence proposal.
2. roadmap_step_completed observations map to a conservative
   behavioral_tendency belief_evidence proposal.
3. repeated runs create no duplicate evidence.
4. provenance always links back to observation_events/source_event_ids.
5. unsupported observations (and observations whose linked events don't
   actually corroborate them) are skipped safely, never raising.
6. no raw/private text is required, read, or persisted.
7. dry-run writes nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.beliefs.process_observations import process_user_observations
from src.common.enums import BeliefType, Direction, SourceType
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.observations.process_events import process_user_events
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
_CLI = Path(__file__).resolve().parents[2] / "tools" / "process_observations_to_evidence.py"


def _check_in_event(event_id: str, *, user_id: str = "usr_1", response: str, timestamp: datetime | None = None) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="check_in_recorded",
        timestamp=timestamp or AS_OF - timedelta(days=1),
        structured_data={"goalId": "goal_1", "checkInId": f"ci_{event_id}", "response": response},
        source="better_you", **VERSION_FIELDS,
    )


def _roadmap_step_event(event_id: str, *, user_id: str = "usr_1", timestamp: datetime | None = None) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="roadmap_step_completed",
        timestamp=timestamp or AS_OF - timedelta(days=1),
        structured_data={"goalId": "goal_1", "roadmapId": "rm_1", "milestoneId": "ms_1", "actionStepId": event_id},
        source="better_you", **VERSION_FIELDS,
    )


def _seed_via_observations_pipeline(repo: Repository, events: list[UserEvent]) -> None:
    """The realistic path: real events -> real observations (via the
    already-built process_user_events step) -> this module's input."""
    for event in events:
        repo.insert_event(event)
    process_user_events(repo, user_id=events[0].user_id, as_of=AS_OF, persist=True)


class CheckInMappingTests(unittest.TestCase):
    def test_a_yes_check_in_produces_supporting_evidence_for_checkin_consistency(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["created"])
        self.assertEqual(len(evidence), 1)
        row = evidence[0]
        self.assertEqual(row.belief_id, "bel_usr_1_checkin_consistency")
        self.assertEqual(row.direction, Direction.SUPPORT)
        self.assertEqual(row.strength, 0.7)
        self.assertEqual(row.source_type, SourceType.RECORDED_EVENT)
        self.assertEqual(row.context_key, "engagement")

    def test_response_determines_direction_and_strength(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(
                repo,
                [
                    _check_in_event("evt_yes", response="yes"),
                    _check_in_event("evt_partly", response="partly"),
                    _check_in_event("evt_no", response="no"),
                ],
            )
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = {row.source_event_ids[0]: row for row in repo.list_all_evidence(user_id="usr_1")}
        finally:
            repo.close()

        self.assertEqual((evidence["evt_yes"].direction, evidence["evt_yes"].strength), (Direction.SUPPORT, 0.7))
        self.assertEqual((evidence["evt_partly"].direction, evidence["evt_partly"].strength), (Direction.SUPPORT, 0.4))
        self.assertEqual((evidence["evt_no"].direction, evidence["evt_no"].strength), (Direction.CONTRADICT, 0.5))

    def test_a_skipped_check_in_produces_no_evidence_either_direction(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="skipped")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["skipped_no_signal"])
        self.assertEqual(evidence, [])

    def test_all_check_ins_route_to_the_same_belief_id(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(
                repo,
                [_check_in_event("evt_1", response="yes"), _check_in_event("evt_2", response="yes")],
            )
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            belief_ids = {row.belief_id for row in repo.list_all_evidence(user_id="usr_1")}
        finally:
            repo.close()
        self.assertEqual(belief_ids, {"bel_usr_1_checkin_consistency"})


class RoadmapStepMappingTests(unittest.TestCase):
    def test_maps_to_a_conservative_behavioral_tendency_proposal(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_roadmap_step_event("evt_1")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["created"])
        self.assertEqual(len(evidence), 1)
        row = evidence[0]
        self.assertEqual(row.belief_id, "bel_usr_1_completes_roadmap_action_steps")
        self.assertEqual(row.direction, Direction.SUPPORT)
        self.assertEqual(row.strength, 0.6)
        self.assertEqual(row.context_key, "goal_progress")


class ProvenanceTests(unittest.TestCase):
    def test_evidence_source_event_ids_traces_back_to_the_real_event(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")[0]
        finally:
            repo.close()

        self.assertEqual(evidence.source_event_ids, ["evt_1"])
        self.assertEqual(evidence.event_id, "evt_1")
        self.assertEqual(evidence.observation_id, "obs_evt_1")

    def test_multi_event_observation_carries_every_linked_event_in_source_event_ids(self) -> None:
        # An observation manually linked to two check-in events (both
        # supporting the same underlying claim) - source_event_ids must
        # carry both, not just one.
        repo = Repository.in_memory()
        try:
            e1 = _check_in_event("evt_1", response="yes")
            e2 = _check_in_event("evt_2", response="yes")
            repo.insert_event(e1)
            repo.insert_event(e2)
            observation, links = create_observation_from_event(
                e1, observation_id="obs_multi", category="check_in",
                observation_text="User completed a scheduled check-in.",
                importance=0.3, confidence=0.9, created_at=AS_OF, **VERSION_FIELDS,
            )
            # add e2 as a supporting link too
            from src.common.enums import LinkRole
            from src.observations.models import ObservationEvent
            links = links + [
                ObservationEvent(
                    observation_id="obs_multi", event_id="evt_2", link_role=LinkRole.SUPPORTING,
                    created_at=AS_OF, **VERSION_FIELDS,
                )
            ]
            repo.insert_observation(observation, links)

            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        # Both linked events are genuinely check_in_recorded and agree on
        # "yes" - homogeneous and unambiguous, so this still works.
        self.assertEqual([o.action for o in result.outcomes], ["created"])
        self.assertEqual(sorted(evidence[0].source_event_ids), ["evt_1", "evt_2"])

    def test_check_in_observation_linked_to_one_unrelated_event_is_skipped_no_signal(self) -> None:
        # Every linked source event must be check_in_recorded - a
        # mixed-source observation is never trusted, even though the one
        # check_in_recorded event present has a perfectly good response.
        # This must never produce evidence whose source_event_ids includes
        # the unrelated event.
        repo = Repository.in_memory()
        try:
            check_in = _check_in_event("evt_1", response="yes")
            unrelated = _roadmap_step_event("evt_2")
            repo.insert_event(check_in)
            repo.insert_event(unrelated)
            observation, links = create_observation_from_event(
                check_in, observation_id="obs_mixed", category="check_in",
                observation_text="User completed a scheduled check-in.",
                importance=0.3, confidence=0.9, created_at=AS_OF, **VERSION_FIELDS,
            )
            from src.common.enums import LinkRole
            from src.observations.models import ObservationEvent
            links = links + [
                ObservationEvent(
                    observation_id="obs_mixed", event_id="evt_2", link_role=LinkRole.SUPPORTING,
                    created_at=AS_OF, **VERSION_FIELDS,
                )
            ]
            repo.insert_observation(observation, links)

            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["skipped_no_signal"])
        self.assertEqual(evidence, [])

    def test_goal_progress_observation_linked_to_one_unrelated_event_is_skipped_no_signal(self) -> None:
        # Same homogeneity requirement for the roadmap-step mapper.
        repo = Repository.in_memory()
        try:
            step = _roadmap_step_event("evt_1")
            unrelated = _check_in_event("evt_2", response="yes")
            repo.insert_event(step)
            repo.insert_event(unrelated)
            observation, links = create_observation_from_event(
                step, observation_id="obs_mixed", category="goal_progress",
                observation_text="User completed a roadmap action step.",
                importance=0.5, confidence=0.9, created_at=AS_OF, **VERSION_FIELDS,
            )
            from src.common.enums import LinkRole
            from src.observations.models import ObservationEvent
            links = links + [
                ObservationEvent(
                    observation_id="obs_mixed", event_id="evt_2", link_role=LinkRole.SUPPORTING,
                    created_at=AS_OF, **VERSION_FIELDS,
                )
            ]
            repo.insert_observation(observation, links)

            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["skipped_no_signal"])
        self.assertEqual(evidence, [])

    def test_cli_output_reports_the_provenance_bearing_belief_id_and_evidence_id(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            repo = Repository.at_path(db_path)
            try:
                _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            finally:
                repo.close()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)

            repo = Repository.at_path(db_path)
            try:
                evidence = repo.get_evidence(payload["created_evidence_ids"][0])
            finally:
                repo.close()
            self.assertEqual(evidence.source_event_ids, ["evt_1"])


class UnsupportedAndMalformedObservationsTests(unittest.TestCase):
    def test_unsupported_categories_are_skipped_safely(self) -> None:
        repo = Repository.in_memory()
        try:
            event = _check_in_event("evt_1", response="yes")
            repo.insert_event(event)
            observation, links = create_observation_from_event(
                event, observation_id="obs_1", category="something_unmapped",
                observation_text="x", importance=0.3, confidence=0.9, created_at=AS_OF, **VERSION_FIELDS,
            )
            repo.insert_observation(observation, links)

            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["skipped_unsupported_category"])
        self.assertEqual(evidence, [])

    def test_a_check_in_category_observation_linked_to_a_non_check_in_event_is_skipped_safely(self) -> None:
        # A store-level mismatch: category says check_in, but the linked
        # event is actually something else entirely. Never guessed at.
        repo = Repository.in_memory()
        try:
            event = _roadmap_step_event("evt_1")
            repo.insert_event(event)
            observation, links = create_observation_from_event(
                event, observation_id="obs_1", category="check_in",
                observation_text="x", importance=0.3, confidence=0.9, created_at=AS_OF, **VERSION_FIELDS,
            )
            repo.insert_observation(observation, links)

            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in result.outcomes], ["skipped_no_signal"])
        self.assertEqual(evidence, [])

    def test_unknown_or_other_users_observation_ids_are_reported_as_not_found(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_owned", user_id="usr_1", response="yes")])
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_other", user_id="usr_2", response="yes")])

            result = process_user_observations(
                repo, user_id="usr_1", as_of=AS_OF, persist=True,
                observation_ids=["obs_evt_owned", "obs_evt_other", "obs_does_not_exist"],
            )
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.observation_id, o.action) for o in result.outcomes),
            [
                ("obs_does_not_exist", "skipped_not_found"),
                ("obs_evt_other", "skipped_not_found"),
                ("obs_evt_owned", "created"),
            ],
        )

    def test_never_raises_on_a_mix_of_supported_unsupported_and_no_signal_observations(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(
                repo,
                [
                    _check_in_event("evt_1", response="yes"),
                    _check_in_event("evt_2", response="skipped"),
                    _roadmap_step_event("evt_3"),
                ],
            )
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.observation_id, o.action) for o in result.outcomes),
            [
                ("obs_evt_1", "created"),
                ("obs_evt_2", "skipped_no_signal"),
                ("obs_evt_3", "created"),
            ],
        )


class NoDuplicatesOnRerunTests(unittest.TestCase):
    def test_rerunning_creates_no_duplicate_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            first = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            second = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual([o.action for o in first.outcomes], ["created"])
        self.assertEqual([o.action for o in second.outcomes], ["skipped_already_processed"])
        self.assertEqual(second.created_evidence_ids, [])
        self.assertEqual(len(evidence), 1)

    def test_adding_a_new_observation_between_runs_only_creates_the_new_one(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)

            _seed_via_observations_pipeline(repo, [_check_in_event("evt_2", response="yes")])
            second = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertEqual(
            sorted((o.observation_id, o.action) for o in second.outcomes),
            [("obs_evt_1", "skipped_already_processed"), ("obs_evt_2", "created")],
        )
        self.assertEqual(len(evidence), 2)


class NoPrivateOrRawTextTests(unittest.TestCase):
    def test_no_field_on_the_proposal_or_evidence_ever_carries_the_observation_prose_text(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            observation_text = repo.get_observation("obs_evt_1").observation
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")[0]
        finally:
            repo.close()

        dumped = evidence.model_dump_json()
        self.assertNotIn(observation_text, dumped)

    def test_raw_content_on_the_source_event_never_reaches_the_evidence(self) -> None:
        secret = "PRIVATE-CHECK-IN-NOTE-should-never-appear-in-any-evidence-row"
        repo = Repository.in_memory()
        try:
            event = UserEvent(
                event_id="evt_1", user_id="usr_1", event_type="check_in_recorded",
                timestamp=AS_OF - timedelta(days=1), raw_content=secret,
                structured_data={"goalId": "goal_1", "checkInId": "ci_1", "response": "yes"},
                source="better_you", **VERSION_FIELDS,
            )
            repo.insert_event(event)
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")[0]
        finally:
            repo.close()

        self.assertNotIn(secret, evidence.model_dump_json())

    def test_only_a_closed_set_of_fields_ever_appear_no_arbitrary_structured_data_passthrough(self) -> None:
        # Guards against a regression where some future change starts
        # forwarding arbitrary structured_data verbatim into evidence.
        repo = Repository.in_memory()
        try:
            event = UserEvent(
                event_id="evt_1", user_id="usr_1", event_type="check_in_recorded",
                timestamp=AS_OF - timedelta(days=1),
                structured_data={
                    "goalId": "goal_1", "checkInId": "ci_1", "response": "yes",
                    "note": "UNEXPECTED-FIELD-should-never-leak-into-evidence",
                },
                source="better_you", **VERSION_FIELDS,
            )
            repo.insert_event(event)
            process_user_events(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            evidence = repo.list_all_evidence(user_id="usr_1")[0]
        finally:
            repo.close()

        self.assertNotIn("UNEXPECTED-FIELD", evidence.model_dump_json())


class RecomputeFlagTests(unittest.TestCase):
    def test_persist_without_recompute_locks_nothing_when_no_belief_exists_yet(self) -> None:
        # lock_belief_until_recompute() only ever writes a locked copy of an
        # *existing* belief (src.storage.repository's own docstring: "False
        # if there is no saved belief yet"). The very first evidence for a
        # belief_id has nothing to lock - a first recompute will naturally
        # see it whenever one eventually runs.
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            latest_belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
        finally:
            repo.close()

        self.assertFalse(result.recomputed)
        self.assertEqual(result.locked_belief_ids, [])
        self.assertIsNone(latest_belief)

    def test_persist_without_recompute_locks_an_already_existing_belief(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True, recompute=True)
            before = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")

            # A second round of evidence, without --recompute this time.
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_2", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True)
            after = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
        finally:
            repo.close()

        self.assertFalse(before.locked_until_recompute)
        self.assertFalse(result.recomputed)
        self.assertEqual(result.locked_belief_ids, ["bel_usr_1_checkin_consistency"])
        self.assertTrue(after.locked_until_recompute)
        # the lock is a new append-only copy, not a rewrite of the recomputed one
        self.assertEqual(after.confidence, before.confidence)

    def test_persist_with_recompute_saves_a_real_belief(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=True, recompute=True)
            latest_belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
        finally:
            repo.close()

        self.assertTrue(result.recomputed)
        self.assertEqual(len(result.recomputed_beliefs), 1)
        self.assertIsNotNone(latest_belief)
        self.assertEqual(latest_belief.belief_type, BeliefType.ROUTINE_OR_PREFERENCE)
        self.assertFalse(latest_belief.locked_until_recompute)

    def test_dry_run_never_recomputes_even_when_requested(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=False, recompute=True)
            latest_belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
        finally:
            repo.close()

        self.assertFalse(result.recomputed)
        self.assertIsNone(latest_belief)


class DryRunTests(unittest.TestCase):
    def test_dry_run_writes_nothing(self) -> None:
        repo = Repository.in_memory()
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
            result = process_user_observations(repo, user_id="usr_1", as_of=AS_OF, persist=False)
            evidence = repo.list_all_evidence(user_id="usr_1")
        finally:
            repo.close()

        self.assertFalse(result.persisted)
        self.assertEqual([o.action for o in result.outcomes], ["would_create"])
        self.assertEqual(result.created_evidence_ids, ["bev_obs_evt_1"])
        self.assertEqual(evidence, [])


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class ProcessObservationsCliTests(unittest.TestCase):
    def _seed(self, db_path: str) -> None:
        repo = Repository.at_path(db_path)
        try:
            _seed_via_observations_pipeline(repo, [_check_in_event("evt_1", response="yes")])
        finally:
            repo.close()

    def test_missing_db_exits_nonzero_without_a_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.sqlite3")
            result = _run_cli(["--db", missing, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 1)
            self.assertIn("no such database file", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_recompute_without_persist_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--recompute"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--recompute requires --persist", result.stderr)

    def test_dry_run_leaves_the_db_byte_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dry run", result.stderr)
            self.assertEqual(Path(db_path).read_bytes(), before)

    def test_persist_writes_the_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            # nothing to lock yet - this is the first-ever evidence for this
            # belief_id, so there is no existing belief to mark stale.
            self.assertIn("persisted 1 evidence row(s); 0 observation(s) skipped.", result.stderr)

            payload = json.loads(result.stdout)
            self.assertEqual(payload["created_evidence_ids"], ["bev_obs_evt_1"])

            repo = Repository.at_path(db_path)
            try:
                evidence = repo.list_all_evidence(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(evidence), 1)

    def test_persist_twice_creates_no_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            first = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("persisted 1 evidence row", first.stderr)

            second = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("persisted 0 evidence row", second.stderr)
            payload = json.loads(second.stdout)
            self.assertEqual(payload["created_evidence_ids"], [])

            repo = Repository.at_path(db_path)
            try:
                evidence = repo.list_all_evidence(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(evidence), 1)

    def test_persist_with_recompute_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            # Explicit --as-of, after the seeded evidence's observed_at:
            # recompute_belief() rejects active evidence observed after the
            # recompute's own as_of, so this must not rely on real wall-clock
            # "now" happening to be later than the fixed AS_OF used to seed.
            recompute_as_of = (AS_OF + timedelta(days=1)).isoformat()
            result = _run_cli([
                "--db", db_path, "--user-id", "usr_1", "--persist", "--recompute",
                "--as-of", recompute_as_of,
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("recomputed 1 belief", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
            finally:
                repo.close()
            self.assertIsNotNone(belief)
            self.assertFalse(belief.locked_until_recompute)

    def test_persist_without_recompute_locks_an_already_existing_belief(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            recompute_as_of = (AS_OF + timedelta(days=1)).isoformat()
            recomputed = _run_cli([
                "--db", db_path, "--user-id", "usr_1", "--persist", "--recompute", "--as-of", recompute_as_of,
            ])
            self.assertEqual(recomputed.returncode, 0, recomputed.stderr)

            # a second round of evidence via the CLI, without --recompute
            repo = Repository.at_path(db_path)
            try:
                _seed_via_observations_pipeline(repo, [_check_in_event("evt_2", response="yes")])
            finally:
                repo.close()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("locked 1 belief", result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
            finally:
                repo.close()
            self.assertTrue(belief.locked_until_recompute)


if __name__ == "__main__":
    unittest.main()

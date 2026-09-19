"""Tests for the local orchestration operation
(src/orchestration/process_user_pipeline.py) that chains the existing
pipeline steps -- unprocessed user_events -> user_observations ->
belief_evidence -> belief recomputation -- for one user in a single call.

Proves the guarantees the task requires:
1. reruns create no duplicate observations or evidence (idempotent).
2. one record's unexpected failure does not block or corrupt any other
   record's processing, and leaves no partial evidence/belief state behind
   for the failing record itself.
3. only the requested user's records are ever read or written.
4. a mix of supported/unsupported event types is handled in one call.
5. a user with zero events produces an all-zero result, no exception.
6. a dry run (persist=False) accurately simulates the *complete* chain --
   including the observation/evidence/recompute stages downstream of a
   record that only exists, so far, as an unprocessed event -- rather than
   silently previewing stage one only.
7. tools/process_user.py's exit code distinguishes a run that found no
   unexpected per-record failures from one that did.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from unittest import mock

from src.events.models import UserEvent
from src.orchestration.process_user_pipeline import (
    ProcessUserPipelineResult,
    RecordFailure,
    process_user_full_pipeline,
)
from src.storage.repository import Repository

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
_CLI = Path(__file__).resolve().parents[2] / "tools" / "process_user.py"


def _check_in_event(
    event_id: str, *, user_id: str, response: str = "yes", timestamp: datetime | None = None
) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="check_in_recorded",
        timestamp=timestamp or AS_OF - timedelta(days=1),
        structured_data={"goalId": "goal_1", "checkInId": f"ci_{event_id}", "response": response},
        source="better_you", **VERSION_FIELDS,
    )


def _roadmap_step_event(event_id: str, *, user_id: str, timestamp: datetime | None = None) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="roadmap_step_completed",
        timestamp=timestamp or AS_OF - timedelta(days=1),
        structured_data={"goalId": "goal_1", "roadmapId": "rm_1", "milestoneId": "ms_1", "actionStepId": event_id},
        source="better_you", **VERSION_FIELDS,
    )


def _unsupported_event(event_id: str, *, user_id: str) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_created",
        timestamp=AS_OF - timedelta(days=1), structured_data={"goalId": "goal_1"},
        source="better_you", **VERSION_FIELDS,
    )


def _malformed_check_in_event(event_id: str, *, user_id: str) -> UserEvent:
    # Allowlisted event_type, but missing the required checkInId -- a
    # malformed, not unsupported, record.
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="check_in_recorded",
        timestamp=AS_OF - timedelta(days=1), structured_data={"goalId": "goal_1", "response": "yes"},
        source="better_you", **VERSION_FIELDS,
    )


class _RaisingRepo:
    """Wraps a real in-memory ``Repository``, raising ``RuntimeError`` from
    one named method for records matching ``when`` -- simulates an otherwise
    unexpected failure (a corrupted row, a disk error, ...) for exactly one
    record, so a test can prove the orchestrator isolates it from every
    other record processed in the same run. Every other call is delegated
    unchanged to the wrapped repository.
    """

    def __init__(self, inner: Repository, *, raise_from: str, when: Callable[[Any], bool]) -> None:
        self._inner = inner
        self._raise_from = raise_from
        self._when = when

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def insert_observation(self, observation: Any, links: Any) -> None:
        if self._raise_from == "insert_observation" and self._when(observation):
            raise RuntimeError("simulated storage failure")
        return self._inner.insert_observation(observation, links)

    def save_belief(self, belief: Any) -> None:
        if self._raise_from == "save_belief" and self._when(belief):
            raise RuntimeError("simulated recompute failure")
        return self._inner.save_belief(belief)


class RerunIdempotencyTests(unittest.TestCase):
    def test_rerun_creates_no_duplicate_observations_or_evidence(self) -> None:
        repo = Repository.in_memory()
        try:
            user_id = "usr_rerun"
            for event in [_check_in_event("evt_1", user_id=user_id), _roadmap_step_event("evt_2", user_id=user_id)]:
                repo.insert_event(event)

            first = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)
            second = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)

            observations = repo.list_observations(user_id=user_id)
            evidence = repo.list_all_evidence(user_id=user_id)
        finally:
            repo.close()

        self.assertEqual(first.counts(), {"processed": 4, "created": 4, "skipped": 0, "failed": 0, "recomputed": 2})
        self.assertEqual(
            second.counts(), {"processed": 4, "created": 0, "skipped": 4, "failed": 0, "recomputed": 0}
        )
        self.assertEqual(len(observations), 2)  # not 4
        self.assertEqual(len(evidence), 2)  # not 4

    def test_rerun_after_recompute_leaves_belief_confidence_unchanged(self) -> None:
        repo = Repository.in_memory()
        try:
            user_id = "usr_rerun2"
            repo.insert_event(_check_in_event("evt_1", user_id=user_id))
            process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)
            belief_after_first = repo.get_latest_belief(user_id=user_id, belief_id=f"bel_{user_id}_checkin_consistency")

            process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)
            belief_after_second = repo.get_latest_belief(user_id=user_id, belief_id=f"bel_{user_id}_checkin_consistency")
        finally:
            repo.close()

        self.assertIsNotNone(belief_after_first)
        # No new evidence on the rerun, so no recompute happened -- the
        # second run's "latest belief" is the exact same saved row.
        self.assertEqual(belief_after_first.confidence, belief_after_second.confidence)
        self.assertFalse(belief_after_second.locked_until_recompute)


class DryRunSimulatesTheCompleteChainTests(unittest.TestCase):
    """A dry run (``persist=False``) must either accurately simulate the
    complete events -> observations -> belief_evidence -> recomputation
    chain, or be documented as a stage-one-only preview. This module chose
    the former (see ``process_user_full_pipeline``'s module docstring:
    ``persist=False`` clones the repo and recurses with ``persist=True``
    against the clone) -- these tests prove that choice actually holds for
    the case that used to fall through the crack: a database that, for this
    user, contains only a single unprocessed event and nothing downstream of
    it yet."""

    def test_a_lone_unprocessed_event_previews_every_downstream_stage(self) -> None:
        repo = Repository.in_memory()
        try:
            user_id = "usr_dry_full_chain"
            repo.insert_event(_check_in_event("evt_1", user_id=user_id, response="yes"))

            result = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=False)

            # Confirms this really was a dry run: nothing reached the real
            # repo at any stage, not just the events table.
            self.assertEqual(repo.list_observations(user_id=user_id), [])
            self.assertEqual(repo.list_all_evidence(user_id=user_id), [])
            self.assertIsNone(
                repo.get_latest_belief(user_id=user_id, belief_id=f"bel_{user_id}_checkin_consistency")
            )
        finally:
            repo.close()

        self.assertFalse(result.persisted)
        # Before the fix, only the event stage's would-create was visible
        # here -- the observation stage re-reads repo.list_observations()
        # after the event stage, and since persist=False never wrote
        # anything, it saw zero observations and reported an all-zero
        # preview for evidence/recompute despite one being clearly implied
        # by the event just previewed.
        self.assertEqual(
            result.counts(), {"processed": 2, "created": 2, "skipped": 0, "failed": 0, "recomputed": 1}
        )
        self.assertEqual([o.action for o in result.event_outcomes], ["would_create"])
        self.assertEqual([o.action for o in result.observation_outcomes], ["would_create"])
        self.assertEqual(result.recomputed_belief_ids, [f"bel_{user_id}_checkin_consistency"])
        self.assertEqual(len(result.recomputed_beliefs), 1)
        self.assertTrue(result.recomputed_beliefs[0]["recomputed"])
        self.assertGreater(result.recomputed_beliefs[0]["confidence"], 0.0)

    def test_dry_run_projection_matches_what_persisting_would_actually_produce(self) -> None:
        # The simulation must not drift from reality: run the same seed data
        # through persist=False and persist=True (against two separate,
        # otherwise-identical databases) and require identical counts and
        # recomputed confidence -- not merely "some nonzero preview."
        def _seed(repo: Repository, user_id: str) -> None:
            repo.insert_event(_check_in_event("evt_1", user_id=user_id, response="yes"))
            repo.insert_event(_roadmap_step_event("evt_2", user_id=user_id))

        dry_repo = Repository.in_memory()
        persisted_repo = Repository.in_memory()
        try:
            _seed(dry_repo, "usr_dry_parity")
            _seed(persisted_repo, "usr_dry_parity")

            dry_result = process_user_full_pipeline(
                dry_repo, user_id="usr_dry_parity", as_of=AS_OF, persist=False
            )
            persisted_result = process_user_full_pipeline(
                persisted_repo, user_id="usr_dry_parity", as_of=AS_OF, persist=True
            )

            self.assertEqual(dry_result.counts(), persisted_result.counts())
            self.assertEqual(dry_result.recomputed_belief_ids, persisted_result.recomputed_belief_ids)
            self.assertEqual(
                sorted(b["confidence"] for b in dry_result.recomputed_beliefs),
                sorted(b["confidence"] for b in persisted_result.recomputed_beliefs),
            )
            # And the real database behind the dry run still has nothing.
            self.assertEqual(dry_repo.list_observations(user_id="usr_dry_parity"), [])
        finally:
            dry_repo.close()
            persisted_repo.close()


class ZeroNewEventsTests(unittest.TestCase):
    def test_a_user_with_no_events_produces_an_all_zero_result(self) -> None:
        repo = Repository.in_memory()
        try:
            result = process_user_full_pipeline(repo, user_id="usr_never_seen", as_of=AS_OF, persist=True)
        finally:
            repo.close()

        self.assertEqual(result.counts(), {"processed": 0, "created": 0, "skipped": 0, "failed": 0, "recomputed": 0})
        self.assertEqual(result.failures, [])
        self.assertEqual(result.recomputed_belief_ids, [])


class MixedEventTypesTests(unittest.TestCase):
    def test_supported_types_are_processed_unsupported_and_malformed_are_skipped(self) -> None:
        repo = Repository.in_memory()
        try:
            user_id = "usr_mixed"
            for event in [
                _check_in_event("evt_checkin", user_id=user_id, response="yes"),
                _roadmap_step_event("evt_roadmap", user_id=user_id),
                _unsupported_event("evt_unsupported", user_id=user_id),
                _malformed_check_in_event("evt_malformed", user_id=user_id),
            ]:
                repo.insert_event(event)

            result = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)
            beliefs = repo.list_latest_beliefs(user_id=user_id)
        finally:
            repo.close()

        self.assertEqual(result.events_created, 2)
        self.assertEqual(result.events_skipped, 2)
        self.assertEqual(result.events_failed, 0)
        self.assertEqual(result.observations_created, 2)
        self.assertEqual(result.observations_failed, 0)
        # check-in and roadmap-step map to two distinct beliefs.
        self.assertEqual(result.beliefs_recomputed, 2)
        self.assertEqual(len(beliefs), 2)
        for belief in beliefs:
            self.assertFalse(belief.locked_until_recompute)


class UserIsolationTests(unittest.TestCase):
    def test_only_the_requested_user_is_read_or_written(self) -> None:
        repo = Repository.in_memory()
        try:
            repo.insert_event(_check_in_event("evt_a1", user_id="usr_a"))
            repo.insert_event(_check_in_event("evt_b1", user_id="usr_b"))

            result = process_user_full_pipeline(repo, user_id="usr_a", as_of=AS_OF, persist=True)

            a_observations = repo.list_observations(user_id="usr_a")
            b_observations = repo.list_observations(user_id="usr_b")
            b_evidence = repo.list_all_evidence(user_id="usr_b")
            b_beliefs = repo.list_latest_beliefs(user_id="usr_b")
        finally:
            repo.close()

        self.assertEqual(result.counts()["created"], 2)  # usr_a's event + observation
        self.assertEqual(len(a_observations), 1)
        self.assertEqual(b_observations, [])
        self.assertEqual(b_evidence, [])
        self.assertEqual(b_beliefs, [])


class PartialFailureTests(unittest.TestCase):
    def test_one_failing_event_does_not_block_or_corrupt_the_others(self) -> None:
        inner = Repository.in_memory()
        try:
            user_id = "usr_partial_events"
            good_event = _check_in_event("evt_good", user_id=user_id, response="yes")
            bad_event = _roadmap_step_event("evt_bad", user_id=user_id)
            inner.insert_event(good_event)
            inner.insert_event(bad_event)

            # The bad event's observation would be "obs_evt_bad" -- simulate
            # an unexpected storage failure only for that one write.
            repo = _RaisingRepo(
                inner, raise_from="insert_observation",
                when=lambda observation: observation.observation_id == "obs_evt_bad",
            )

            result = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)

            good_observation = inner.get_observation("obs_evt_good")
            bad_observation = inner.get_observation("obs_evt_bad")
        finally:
            inner.close()

        self.assertEqual(result.events_created, 1)
        self.assertEqual(result.events_failed, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].stage, "event")
        self.assertEqual(result.failures[0].record_id, "evt_bad")
        self.assertIsNotNone(good_observation)  # the unrelated good record still succeeded
        self.assertIsNone(bad_observation)  # the bad record left nothing behind

    def test_a_recompute_failure_rolls_back_its_own_evidence_and_spares_others(self) -> None:
        inner = Repository.in_memory()
        try:
            user_id = "usr_partial_obs"
            good_event = _check_in_event("evt_good", user_id=user_id, response="yes")
            bad_event = _roadmap_step_event("evt_bad", user_id=user_id)
            inner.insert_event(good_event)
            inner.insert_event(bad_event)

            bad_belief_id = f"bel_{user_id}_completes_roadmap_action_steps"
            good_belief_id = f"bel_{user_id}_checkin_consistency"

            # Simulate the mandatory recompute itself failing only for the
            # roadmap-step belief, after its evidence has already been
            # inserted in the same transaction as the pipeline's own call.
            repo = _RaisingRepo(
                inner, raise_from="save_belief",
                when=lambda belief: belief.belief_id == bad_belief_id,
            )

            result = process_user_full_pipeline(repo, user_id=user_id, as_of=AS_OF, persist=True)

            bad_evidence = inner.list_all_evidence(user_id=user_id, belief_id=bad_belief_id)
            bad_belief = inner.get_latest_belief(user_id=user_id, belief_id=bad_belief_id)
            good_evidence = inner.list_all_evidence(user_id=user_id, belief_id=good_belief_id)
            good_belief = inner.get_latest_belief(user_id=user_id, belief_id=good_belief_id)
        finally:
            inner.close()

        self.assertEqual(result.observations_failed, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].stage, "observation")
        self.assertEqual(result.failures[0].record_id, "obs_evt_bad")

        # The failing record's evidence insert was rolled back together with
        # its failed recompute -- no orphaned evidence, no belief at all.
        self.assertEqual(bad_evidence, [])
        self.assertIsNone(bad_belief)

        # The unrelated good record was unaffected: its evidence exists and
        # its belief was recomputed and left unlocked.
        self.assertEqual(len(good_evidence), 1)
        self.assertIsNotNone(good_belief)
        self.assertFalse(good_belief.locked_until_recompute)
        self.assertEqual(result.recomputed_belief_ids, [good_belief_id])


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_CLI), *args], capture_output=True, text=True)


class ProcessUserCliTests(unittest.TestCase):
    def _seed(self, db_path: str) -> None:
        repo = Repository.at_path(db_path)
        try:
            repo.insert_event(_check_in_event("evt_1", user_id="usr_1", response="yes"))
        finally:
            repo.close()

    def test_missing_db_exits_nonzero_without_a_traceback(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.sqlite3")
            result = _run_cli(["--db", missing, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 1)
            self.assertIn("no such database file", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_dry_run_leaves_the_db_byte_identical(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dry run", result.stderr)
            self.assertEqual(Path(db_path).read_bytes(), before)

    def test_dry_run_previews_the_full_chain_including_recompute(self) -> None:
        # The seeded db has exactly one unprocessed event and nothing
        # downstream of it yet -- the case that used to make a dry run
        # report the event's would-create and then silently zero out every
        # later stage, because nothing from stage one was ever actually
        # written for stage two to pick back up.
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)
            before = Path(db_path).read_bytes()

            result = _run_cli(["--db", db_path, "--user-id", "usr_1"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "dry run: nothing was written",
                result.stderr,
            )
            self.assertIn("would recompute 1 belief(s)", result.stderr)
            self.assertEqual(Path(db_path).read_bytes(), before)

            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["counts"], {"processed": 2, "created": 2, "skipped": 0, "failed": 0, "recomputed": 1}
            )
            self.assertEqual(len(payload["recomputed_beliefs"]), 1)
            self.assertGreater(payload["recomputed_beliefs"][0]["confidence"], 0.0)

    def test_persist_builds_the_whole_chain(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            result = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("persisted 2 record(s); 0 skipped; 0 failed; recomputed 1 belief(s).", result.stderr)

            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["counts"], {"processed": 2, "created": 2, "skipped": 0, "failed": 0, "recomputed": 1}
            )

            repo = Repository.at_path(db_path)
            try:
                evidence = repo.list_all_evidence(user_id="usr_1")
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_usr_1_checkin_consistency")
            finally:
                repo.close()
            self.assertEqual(len(evidence), 1)
            self.assertIsNotNone(belief)
            self.assertFalse(belief.locked_until_recompute)

    def test_persist_twice_creates_no_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            first = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("persisted 2 record(s)", first.stderr)

            second = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("persisted 0 record(s); 2 skipped; 0 failed; recomputed 0 belief(s).", second.stderr)

            repo = Repository.at_path(db_path)
            try:
                observations = repo.list_observations(user_id="usr_1")
                evidence = repo.list_all_evidence(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(observations), 1)
            self.assertEqual(len(evidence), 1)


class ProcessUserCliExitCodeTests(unittest.TestCase):
    """result.failures is how process_user_full_pipeline reports a record
    that hit an unexpected error while every other record still succeeded
    or was safely skipped (see PartialFailureTests above for how that
    happens) -- but exit 0 is what a scripted caller actually branches on,
    and 0 historically meant "nothing went wrong." These tests hold
    tools/process_user.py to the fixed contract: a run is only exit 0 when
    result.failures is empty, whether or not anything else in the run
    succeeded."""

    def _seed(self, db_path: str) -> None:
        repo = Repository.at_path(db_path)
        try:
            repo.insert_event(_check_in_event("evt_1", user_id="usr_1", response="yes"))
        finally:
            repo.close()

    def test_nonempty_failures_exits_nonzero_and_distinct_from_the_missing_db_exit_code(self) -> None:
        import tools.process_user as cli_module

        fake_result = ProcessUserPipelineResult(
            user_id="usr_1", persisted=True,
            events_created=0, events_skipped=0, events_failed=1,
            observations_created=0, observations_skipped=0, observations_failed=0,
            beliefs_recomputed=0, recomputed_belief_ids=[],
            failures=[RecordFailure("event", "evt_1", "RuntimeError: simulated storage failure")],
            event_outcomes=[], observation_outcomes=[], recomputed_beliefs=[],
        )

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            with mock.patch.object(cli_module, "process_user_full_pipeline", return_value=fake_result):
                exit_code = cli_module.main(["--db", db_path, "--user-id", "usr_1", "--persist"])

        # Nonzero, and not the same code a caller-error (e.g. a missing
        # --db, exit 1) already uses -- a scripted caller must be able to
        # tell "this run itself found a problem" apart from "this run
        # couldn't even start."
        self.assertNotEqual(exit_code, 0)
        self.assertNotEqual(exit_code, 1)

    def test_empty_failures_still_exits_zero_even_with_skips(self) -> None:
        # A run with zero unexpected failures is exit 0 no matter how many
        # records were skipped for an ordinary, already-graceful reason --
        # only result.failures (not result.counts()["skipped"]) drives the
        # exit code.
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "canonical.sqlite3")
            self._seed(db_path)

            first = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(first.returncode, 0, first.stderr)

            second = _run_cli(["--db", db_path, "--user-id", "usr_1", "--persist"])
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("0 failed", second.stderr)


if __name__ == "__main__":
    unittest.main()

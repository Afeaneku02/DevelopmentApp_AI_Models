"""Tests for tools/resolve_belief_key.py -- the manual belief-key
canonicalization CLI.

Runs the CLI as a real subprocess, matching tests/event_intake's own CLI test
style for the other intake/recompute/invalidation CLIs. Every persistence
claim is double-checked by reading the database back through
src.storage.repository.Repository directly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_RESOLVE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "resolve_belief_key.py"
_RECOMPUTE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "recompute_belief.py"


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def _run_resolve(args: list[str]) -> subprocess.CompletedProcess:
    return _run(_RESOLVE_SCRIPT, args)


class DuplicateCanonicalizationIdRejectionTests(unittest.TestCase):
    def test_duplicate_canonicalization_id_is_rejected_and_original_row_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            first = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "key_a",
            ])
            self.assertEqual(first.returncode, 0, first.stderr)

            second = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "key_b",
            ])

            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Duplicate canonicalization_id", second.stderr)
            self.assertIn("can_1", second.stderr)

            repo = Repository.at_path(db_path)
            try:
                all_decisions = repo.list_belief_key_canonicalizations(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual(len(all_decisions), 1)
            self.assertEqual(all_decisions[0].proposed_key, "key_a")

    def test_re_resolving_the_same_proposed_key_with_a_new_id_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            first = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "key_a",
            ])
            self.assertEqual(first.returncode, 0, first.stderr)

            second = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_2", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "key_a",
            ])
            self.assertEqual(second.returncode, 0, second.stderr)

            repo = Repository.at_path(db_path)
            try:
                all_decisions = repo.list_belief_key_canonicalizations(user_id="usr_1")
            finally:
                repo.close()
            self.assertEqual([d.canonicalization_id for d in all_decisions], ["can_1", "can_2"])


class KnownAliasTests(unittest.TestCase):
    def test_known_alias_resolves_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "prefers_evening_exercise_sessions",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["canonical_key"], "higher_adherence_after_work")
            self.assertEqual(printed["decision"], "alias")

            repo = Repository.at_path(db_path)
            try:
                stored = repo.get_latest_belief_key_canonicalization(
                    user_id="usr_1", belief_type="behavioral_tendency",
                    proposed_key="prefers_evening_exercise_sessions",
                )
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.canonical_key, "higher_adherence_after_work")


class UnknownKeyStaysSeparateTests(unittest.TestCase):
    def test_unknown_key_stays_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "a_totally_novel_key",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["canonical_key"], "a_totally_novel_key")
            self.assertEqual(printed["decision"], "keep_separate")


class RiskyMergeGoesToManualReviewTests(unittest.TestCase):
    def test_merge_request_is_downgraded_to_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "some_key",
                "--proposed-decision", "merge", "--reason", "seems related",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["decision"], "manual_review")
            self.assertNotEqual(printed["decision"], "merge")


class CannotOverrideBackendPolicyTests(unittest.TestCase):
    def test_alias_request_to_an_unverified_target_is_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "another_key",
                "--proposed-decision", "alias", "--proposed-canonical-key", "higher_adherence_after_work",
            ])

            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed["decision"], "manual_review")

    def test_invalid_proposed_decision_choice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "some_key",
                "--proposed-decision", "not_a_real_decision",
            ])

            self.assertEqual(result.returncode, 2)
            self.assertIn("--proposed-decision", result.stderr)


class RecomputeUsesCanonicalKeyTests(unittest.TestCase):
    def test_canonical_key_from_resolution_is_what_recompute_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")

            resolve_result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "prefers_evening_exercise_sessions",
            ])
            self.assertEqual(resolve_result.returncode, 0, resolve_result.stderr)
            canonical_key = json.loads(resolve_result.stdout)["canonical_key"]
            self.assertNotEqual(canonical_key, "prefers_evening_exercise_sessions")

            recompute_result = _run(_RECOMPUTE_SCRIPT, [
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", canonical_key,
                "--belief-value-json", "true", "--allow-no-evidence",
            ])
            self.assertEqual(recompute_result.returncode, 0, recompute_result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            self.assertEqual(belief.belief_key, "higher_adherence_after_work")


class DoesNotRewriteExistingRowsTests(unittest.TestCase):
    def test_resolving_a_key_does_not_touch_any_belief_or_evidence_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            recompute_result = _run(_RECOMPUTE_SCRIPT, [
                "--db", db_path, "--belief-id", "bel_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--belief-key", "prefers_evening_exercise_sessions",
                "--belief-value-json", "true", "--allow-no-evidence",
            ])
            self.assertEqual(recompute_result.returncode, 0, recompute_result.stderr)

            resolve_result = _run_resolve([
                "--db", db_path, "--canonicalization-id", "can_1", "--user-id", "usr_1",
                "--belief-type", "behavioral_tendency", "--proposed-key", "prefers_evening_exercise_sessions",
            ])
            self.assertEqual(resolve_result.returncode, 0, resolve_result.stderr)

            repo = Repository.at_path(db_path)
            try:
                belief = repo.get_latest_belief(user_id="usr_1", belief_id="bel_1")
            finally:
                repo.close()
            # The existing belief row, saved under the raw (pre-canonical)
            # key, must remain exactly as it was -- canonicalization never
            # rewrites or deletes it.
            self.assertEqual(belief.belief_key, "prefers_evening_exercise_sessions")


if __name__ == "__main__":
    unittest.main()

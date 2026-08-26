"""Tests for tools/demo_user_model.py -- the deterministic demo runner.

Asserts on the printed summary's content and ordering, not just that the
script runs without raising: the whole point of the demo is to visibly show
the confidence/status/lock transitions across a recompute -> invalidate ->
recompute cycle, so those transitions are exactly what these tests check.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tools.demo_user_model import run

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "demo_user_model.py"


class DemoOutputContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = run()

    def test_all_three_events_are_shown(self) -> None:
        for event_id in ("evt_1041", "evt_1042", "evt_1043"):
            self.assertIn(f"UserEvent {event_id}", self.output)

    def test_all_three_evidence_rows_show_their_own_source_event_id(self) -> None:
        for event_id in ("evt_1041", "evt_1042", "evt_1043"):
            self.assertIn(f"source_event_ids=['{event_id}']", self.output)

    def test_evidence_uses_canonical_source_type_and_reliability_default(self) -> None:
        self.assertIn("source_type=recorded_event", self.output)
        self.assertIn("source_reliability=0.95", self.output)

    def test_evidence_stays_leaf_default_since_no_backend_approval_was_ever_given(self) -> None:
        self.assertIn("authorized_aggregation_mode=leaf_default", self.output)

    def test_first_recompute_shows_positive_unlocked_confidence(self) -> None:
        section = self.output.split("Recompute #1")[1].split("--")[1]
        confidence = float(section.split("confidence:")[1].splitlines()[0].strip())
        status = section.split("status:")[1].splitlines()[0].strip()
        locked = section.split("locked_until_recompute:")[1].splitlines()[0].strip()
        self.assertGreater(confidence, 0.0)
        self.assertEqual(status, "provisional")
        self.assertEqual(locked, "False")

    def test_belief_is_locked_before_the_second_recompute(self) -> None:
        section = self.output.split("BEFORE recompute")[1].split("--")[1]
        confidence = float(section.split("confidence:")[1].splitlines()[0].strip())
        locked = section.split("locked_until_recompute:")[1].splitlines()[0].strip()
        # Locked, but still showing the same (now non-authoritative) prior
        # confidence -- the lock flag is what a caller must check, not a
        # zeroed-out number at lock time (see src/storage/repository.py's
        # _lock_latest_belief()).
        self.assertGreater(confidence, 0.0)
        self.assertEqual(locked, "True")

    def test_second_recompute_resets_confidence_to_zero_and_status_to_outdated(self) -> None:
        section = self.output.split("Recompute #2")[1]
        confidence = float(section.split("confidence:")[1].splitlines()[0].strip())
        status = section.split("status:")[1].splitlines()[0].strip()
        locked = section.split("locked_until_recompute:")[1].splitlines()[0].strip()
        self.assertEqual(confidence, 0.0)
        self.assertEqual(status, "outdated")
        self.assertEqual(locked, "False")

    def test_transitions_appear_in_the_correct_order(self) -> None:
        # unlocked-positive -> locked-stale -> unlocked-zero, not any other order.
        first = self.output.index("Recompute #1")
        invalidating = self.output.index("Invalidating all evidence")
        before_second = self.output.index("BEFORE recompute")
        second = self.output.index("Recompute #2")
        self.assertLess(first, invalidating)
        self.assertLess(invalidating, before_second)
        self.assertLess(before_second, second)

    def test_active_evidence_count_drops_to_zero_after_invalidation(self) -> None:
        self.assertIn("active evidence remaining: 0", self.output)


class DemoCliTests(unittest.TestCase):
    def test_running_as_a_script_exits_zero_and_prints_the_summary(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH)], capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Better You adaptive user model", result.stdout)
        self.assertIn("status: outdated", result.stdout)


if __name__ == "__main__":
    unittest.main()

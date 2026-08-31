"""Tests for the adaptive-user-model evaluation harness
(src/evals/harness.py + tools/evaluate_user_model.py).

Covers:
- every bundled example scenario passes, and the eight required scenarios
  are present;
- a scenario whose expectation is wrong is reported failed and the CLI
  exits nonzero;
- a structurally broken scenario (unknown op, dangling reference, no
  expectations) is reported failed, not crashed;
- CLI exit codes: 0 all pass, 1 any fail, 2 usage error;
- ``--format json`` emits a machine-readable scorecard.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evals.harness import (
    discover_manifests,
    format_report,
    run_manifests,
    run_scenario,
)

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples" / "evals"
_CLI = _ROOT / "tools" / "evaluate_user_model.py"

_REQUIRED_SCENARIOS = {
    "clean_support_raises_confidence",
    "mild_contradiction_lowers_confidence",
    "strong_contradiction_contests_belief",
    "invalidation_locks_stale_belief",
    "recommendation_uses_only_allowed_beliefs",
    "outcome_learning_creates_weak_proposals",
    "review_approval_promotes_evidence",
    "review_rejection_promotes_nothing",
}


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CLI), *args], capture_output=True, text=True, cwd=str(_ROOT)
    )


class BundledExamplesTests(unittest.TestCase):
    def test_all_eight_required_scenarios_are_present(self) -> None:
        names = {p.stem for p in _EXAMPLES.glob("*.json")}
        self.assertEqual(_REQUIRED_SCENARIOS, names & _REQUIRED_SCENARIOS)

    def test_every_bundled_scenario_passes(self) -> None:
        report = run_manifests([str(_EXAMPLES)])
        self.assertTrue(report.scenarios, "no scenarios were discovered")
        failed = [
            (s.name, s.error or [c for c in s.checks if not c.passed])
            for s in report.scenarios
            if not s.passed
        ]
        self.assertEqual(failed, [], f"scenarios failed: {failed}")
        self.assertTrue(report.passed)

    def test_cli_exits_zero_on_the_bundled_examples(self) -> None:
        result = _run_cli(["--manifest", str(_EXAMPLES)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("scenarios passed", result.stderr)

    def test_cli_json_format_is_valid_and_passes(self) -> None:
        result = _run_cli(["--manifest", str(_EXAMPLES), "--format", "json"])
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["summary"]["failed"], 0)
        self.assertEqual(payload["summary"]["scenarios"], len(payload["scenarios"]))


class FailedExpectationTests(unittest.TestCase):
    """A wrong expectation must be reported as a failure, not silently pass."""

    def _inverted_manifest(self) -> dict:
        return {
            "name": "inverted_expectation",
            "description": "clean support raises confidence, but we (wrongly) assert it falls",
            "steps": [
                {"op": "evidence", "id": "s1", "belief": "b", "direction": "support",
                 "strength": 0.9, "source_type": "recorded_event", "days_before_as_of": 12},
                {"op": "recompute", "belief": "b", "belief_key": "higher_adherence_after_work",
                 "belief_value": True, "checkpoint": "one"},
                {"op": "evidence", "id": "s2", "belief": "b", "direction": "support",
                 "strength": 0.9, "source_type": "explicit_user_statement", "days_before_as_of": 8},
                {"op": "recompute", "belief": "b", "belief_key": "higher_adherence_after_work",
                 "belief_value": True},
            ],
            "expect": [
                {"check": "belief_confidence", "belief": "b", "lt_checkpoint": "one"},
            ],
        }

    def test_run_scenario_marks_a_wrong_expectation_failed(self) -> None:
        result = run_scenario(self._inverted_manifest())
        self.assertFalse(result.passed)
        self.assertEqual(len(result.checks), 1)
        self.assertFalse(result.checks[0].passed)
        self.assertIsNone(result.error)

    def test_cli_exits_nonzero_when_an_expectation_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inverted.json"
            path.write_text(json.dumps(self._inverted_manifest()), encoding="utf-8")
            result = _run_cli(["--manifest", str(path)])
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("FAIL", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_json_reports_the_failure_and_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inverted.json"
            path.write_text(json.dumps(self._inverted_manifest()), encoding="utf-8")
            result = _run_cli(["--manifest", str(path), "--format", "json"])
            self.assertEqual(result.returncode, 1)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["summary"]["failed"], 1)


class BrokenScenarioTests(unittest.TestCase):
    def test_unknown_op_is_reported_not_raised(self) -> None:
        result = run_scenario(
            {"name": "bad_op", "steps": [{"op": "teleport"}], "expect": [{"check": "signal_exists", "context": "x"}]}
        )
        self.assertFalse(result.passed)
        self.assertIn("unknown op", result.error or "")

    def test_dangling_reference_is_reported_not_raised(self) -> None:
        result = run_scenario(
            {
                "name": "bad_ref",
                "steps": [{"op": "observation", "id": "o1", "event": "nope"}],
                "expect": [{"check": "signal_exists", "context": "x"}],
            }
        )
        self.assertFalse(result.passed)
        self.assertIn("unknown event", result.error or "")

    def test_scenario_with_no_expectations_fails(self) -> None:
        result = run_scenario({"name": "empty", "steps": [], "expect": []})
        self.assertFalse(result.passed)
        self.assertIn("no expectations", result.error or "")

    def test_cli_exits_nonzero_for_a_broken_scenario(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "broken",
                        "steps": [{"op": "evidence", "id": "e1", "belief": "b", "observation": "missing"}],
                        "expect": [{"check": "belief_confidence", "belief": "b", "min": 0}],
                    }
                ),
                encoding="utf-8",
            )
            result = _run_cli(["--manifest", str(path)])
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("ERROR", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


class UsageErrorTests(unittest.TestCase):
    def test_missing_manifest_file_exits_two(self) -> None:
        result = _run_cli(["--manifest", str(_ROOT / "does_not_exist.json")])
        self.assertEqual(result.returncode, 2)
        self.assertIn("not found", result.stderr)

    def test_empty_directory_exits_two(self) -> None:
        with TemporaryDirectory() as tmp:
            result = _run_cli(["--manifest", tmp])
            self.assertEqual(result.returncode, 2)
            self.assertIn("No scenario manifests", result.stderr)


class HarnessUnitTests(unittest.TestCase):
    def test_discover_manifests_sorts_json_files_in_a_directory(self) -> None:
        found = discover_manifests([str(_EXAMPLES)])
        self.assertEqual(found, sorted(found))
        self.assertTrue(all(p.suffix == ".json" for p in found))
        self.assertGreaterEqual(len(found), 8)

    def test_format_report_names_failing_checks(self) -> None:
        report = run_manifests([str(_EXAMPLES)])
        text = format_report(report)
        self.assertIn("scenarios passed", text)
        for name in _REQUIRED_SCENARIOS:
            self.assertIn(name, text)

    def test_passing_scenario_records_all_checks(self) -> None:
        manifest = json.loads(
            (_EXAMPLES / "clean_support_raises_confidence.json").read_text(encoding="utf-8")
        )
        result = run_scenario(manifest, source="clean_support_raises_confidence.json")
        self.assertTrue(result.passed)
        self.assertTrue(result.checks)
        self.assertTrue(all(c.passed for c in result.checks))
        self.assertEqual(result.passed_checks, result.total_checks)


if __name__ == "__main__":
    unittest.main()

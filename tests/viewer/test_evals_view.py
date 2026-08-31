"""Tests for src/viewer/evals_view.py -- the pure HTML rendering of the
evaluation scorecard shown at the server's ``/evals`` route.

Covers: the shipped manifests render as an all-pass scorecard; a missing
directory renders an empty-state page; a broken/unparseable manifest renders
as a failed scenario with its error; running the report uses fresh in-memory
repos and touches no on-disk database.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.viewer.evals_view import (
    DEFAULT_MANIFEST_DIR,
    collect_eval_report,
    render_evals_html,
)

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


class ShippedManifestsScorecardTests(unittest.TestCase):
    def test_default_manifests_render_as_an_all_pass_scorecard(self) -> None:
        report, resolved = collect_eval_report()
        self.assertEqual(resolved, DEFAULT_MANIFEST_DIR)
        self.assertTrue(report.passed, [s.name for s in report.scenarios if not s.passed])

        html = render_evals_html(report, manifest_dir=resolved)
        self.assertIn("<!doctype html>", html.lstrip().lower())
        self.assertIn("evaluation scorecard", html.lower())
        self.assertIn("READ-ONLY", html)
        self.assertIn("ALL PASS", html)
        for name in _REQUIRED_SCENARIOS:
            self.assertIn(name, html)
        # summary counts are present
        self.assertIn("scenarios passed", html)
        self.assertIn("checks passed", html)
        # a link back to the normal viewer
        self.assertIn('href="/"', html)

    def test_scorecard_renders_nav_links_to_both_viewer_pages(self) -> None:
        report, resolved = collect_eval_report()
        html = render_evals_html(report, manifest_dir=resolved)
        self.assertIn('<nav class="nav">', html)
        self.assertIn("User Model", html)
        self.assertIn("Eval Scorecard", html)
        self.assertIn('href="/"', html)
        self.assertIn('href="/evals"', html)
        # the scorecard marks its own nav link active
        self.assertIn('<a href="/evals" class="active">Eval Scorecard</a>', html)

    def test_every_check_row_is_shown(self) -> None:
        report, resolved = collect_eval_report()
        html = render_evals_html(report, manifest_dir=resolved)
        total_checks = sum(s.total_checks for s in report.scenarios)
        self.assertGreater(total_checks, 20)
        # each scenario contributes a checks table
        self.assertEqual(html.count("<table>"), len(report.scenarios))


class MissingManifestDirTests(unittest.TestCase):
    def test_missing_directory_renders_empty_state_not_crash(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not_here"
            report, resolved = collect_eval_report(missing)
            self.assertEqual(report.scenarios, [])
            self.assertFalse(report.passed)

            html = render_evals_html(report, manifest_dir=resolved)
            self.assertIn("No evaluation manifests were found", html)
            self.assertIn(str(missing), html)


class BrokenManifestTests(unittest.TestCase):
    def test_unparseable_manifest_renders_as_a_failed_scenario(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.json").write_text("{ this is not json", encoding="utf-8")
            report, resolved = collect_eval_report(tmp)
            self.assertFalse(report.passed)
            self.assertEqual(len(report.scenarios), 1)
            self.assertFalse(report.scenarios[0].passed)

            html = render_evals_html(report, manifest_dir=resolved)
            self.assertIn("FAIL", html)
            self.assertIn("could not load manifest", html)

    def test_scenario_with_a_bad_step_renders_its_error(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "bad_step.json").write_text(
                json.dumps(
                    {
                        "name": "bad_step",
                        "description": "uses an op that does not exist",
                        "steps": [{"op": "teleport"}],
                        "expect": [{"check": "signal_exists", "context": "x"}],
                    }
                ),
                encoding="utf-8",
            )
            report, resolved = collect_eval_report(tmp)
            html = render_evals_html(report, manifest_dir=resolved)
            self.assertIn("could not run this scenario", html)
            self.assertIn("unknown op", html)
            self.assertIn("FAILURES", html)

    def test_a_wrong_expectation_shows_a_failing_check_row(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "wrong.json").write_text(
                json.dumps(
                    {
                        "name": "wrong_expectation",
                        "steps": [
                            {"op": "evidence", "id": "s1", "belief": "b", "direction": "support",
                             "strength": 0.9, "source_type": "recorded_event", "days_before_as_of": 10},
                            {"op": "recompute", "belief": "b", "belief_key": "higher_adherence_after_work",
                             "belief_value": True},
                        ],
                        "expect": [{"check": "belief_confidence", "belief": "b", "min": 0.99}],
                    }
                ),
                encoding="utf-8",
            )
            report, resolved = collect_eval_report(tmp)
            self.assertFalse(report.passed)
            html = render_evals_html(report, manifest_dir=resolved)
            self.assertIn("eval_user__belief__b.confidence", html)
            self.assertIn("0.99", html)
            self.assertIn('<span class="tag fail">fail</span>', html)


class HtmlEscapingTests(unittest.TestCase):
    def test_manifest_supplied_text_is_escaped(self) -> None:
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "xss.json").write_text(
                json.dumps(
                    {
                        "name": "xss_<script>alert(1)</script>",
                        "description": "<img src=x onerror=alert(1)>",
                        "steps": [],
                        "expect": [{"check": "signal_exists", "context": "x"}],
                    }
                ),
                encoding="utf-8",
            )
            report, resolved = collect_eval_report(tmp)
            html = render_evals_html(report, manifest_dir=resolved)
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertNotIn("<img src=x onerror=alert(1)>", html)
            self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()

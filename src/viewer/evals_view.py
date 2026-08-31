"""Pure HTML rendering for the read-only evaluation scorecard
(``tools/serve_user_model.py``'s ``/evals`` route).

``collect_eval_report()`` runs the existing evaluation harness
(``src.evals.harness``) over a directory of scenario manifests -- every
scenario in its own fresh in-memory ``Repository``, so this never touches
the database being served -- and ``render_evals_html()`` turns the resulting
``EvalReport`` into one self-contained page (inline CSS reused from
``user_model_view``, no JavaScript, every value ``html.escape``-d).

No scoring, recommendation, or outcome-learning logic lives here: this
module only calls ``src.evals.harness.run_manifests`` and projects its
result onto HTML.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from src.evals.harness import EvalReport, run_manifests
from src.viewer.user_model_view import _CSS, _esc, _nav, _table, _tag

_EVALS_CSS = """
.scenario { margin: 1.5rem 0; }
.scenario h2 { margin-bottom: .25rem; }
.scenario .desc { color: #666; font-size: .9rem; margin: 0 0 .5rem; }
.tag.pass { background: #0a7d33; color: #fff; }
.tag.fail { background: #b02a37; color: #fff; }
.callout { border-left: 3px solid #b02a37; background: #fdf0f1; padding: .5rem .75rem;
           margin: .5rem 0; font-size: .9rem; }
@media (prefers-color-scheme: dark) { .callout { background: #2a1e20; } }
"""

# The default manifest directory shipped with the repo.
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "examples" / "evals"


def collect_eval_report(manifest_dir: str | Path | None = None) -> tuple[EvalReport, Path]:
    """Run every ``*.json`` scenario manifest under ``manifest_dir`` (default:
    the repo's ``examples/evals/``) through the harness and return
    ``(report, resolved_dir)``.

    Never raises for a missing directory or a broken manifest: a missing
    directory yields an empty report; a manifest that cannot be parsed or
    executed is recorded on the report as a failed scenario with an
    ``error`` message (the harness does this itself)."""
    resolved = Path(manifest_dir) if manifest_dir is not None else DEFAULT_MANIFEST_DIR
    if not resolved.is_dir():
        return EvalReport(scenarios=[]), resolved
    return run_manifests([str(resolved)]), resolved


def _summary_cards(report: EvalReport) -> str:
    data = report.to_dict()["summary"]
    total_checks = sum(s.total_checks for s in report.scenarios)
    passed_checks = sum(s.passed_checks for s in report.scenarios)
    cards = [
        ("scenarios", data["scenarios"]),
        ("scenarios passed", data["passed"]),
        ("scenarios failed", data["failed"]),
        ("checks", total_checks),
        ("checks passed", passed_checks),
    ]
    parts = [
        f'<div class="card"><div class="n">{int(n)}</div><div class="l">{html.escape(label)}</div></div>'
        for label, n in cards
    ]
    return f'<div class="cards">{"".join(parts)}</div>'


def _scenario_section(scenario) -> str:
    tag = _tag("PASS", "pass") if scenario.passed else _tag("FAIL", "fail")
    parts = [
        '<div class="scenario">',
        f"<h2>{tag} {html.escape(scenario.name)}</h2>",
    ]
    if scenario.description:
        parts.append(f'<p class="desc">{html.escape(scenario.description)}</p>')
    if scenario.source:
        parts.append(f'<p class="desc">manifest: <code>{html.escape(scenario.source)}</code></p>')
    if scenario.error:
        parts.append(f'<div class="callout">could not run this scenario: {html.escape(scenario.error)}</div>')

    rows = [
        [
            _tag("pass", "pass") if check.passed else _tag("fail", "fail"),
            _esc(check.check),
            _esc(check.target),
            _esc(check.expected),
            _esc(check.actual),
            _esc(check.message),
        ]
        for check in scenario.checks
    ]
    parts.append(
        _table(
            ["result", "check", "target", "expected", "actual", "message"],
            rows,
            empty="This scenario declared no checks.",
            wrap_columns={2, 3, 4, 5},
        )
    )
    parts.append("</div>")
    return "".join(parts)


def render_evals_html(
    report: EvalReport,
    *,
    manifest_dir: str | Path,
    generated_at: datetime | None = None,
) -> str:
    """Render an ``EvalReport`` to one self-contained HTML page. Pure -- no
    I/O, no scripts, no external resources."""
    generated_at = generated_at or datetime.now(timezone.utc)
    overall = _tag("ALL PASS", "pass") if report.passed else _tag("FAILURES", "fail")

    if report.scenarios:
        body_sections = "\n".join(_scenario_section(s) for s in report.scenarios)
    else:
        body_sections = (
            '<p class="empty">No evaluation manifests were found under '
            f"<code>{html.escape(str(manifest_dir))}</code>.</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>User model evaluation scorecard</title>
<style>{_CSS}{_EVALS_CSS}</style>
</head>
<body>
{_nav("evals")}
<h1>Adaptive user model &mdash; evaluation scorecard</h1>
<div class="meta">
  <span class="readonly">READ-ONLY</span> {overall}
  &middot; manifests: <code>{html.escape(str(manifest_dir))}</code><br>
  generated {html.escape(generated_at.isoformat())}
</div>
{_summary_cards(report)}
{body_sections}
</body>
</html>
"""

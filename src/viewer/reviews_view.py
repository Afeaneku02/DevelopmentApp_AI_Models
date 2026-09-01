"""Pure data-shaping and HTML rendering for the read-only manual review
queue (``tools/serve_user_model.py``'s ``/reviews`` route).

The review queue surfaces outcome-learning signals that a human must decide
on before their proposed evidence can be promoted (blueprint section 6.4's
manual-review gate, applied to the section 12.6 learning loop):

- **Pending** -- signals with no review yet, shown with their proposed
  evidence so a reviewer can see exactly what would be written.
- **Reviewed** -- signals whose latest review is ``approved`` / ``rejected``,
  shown with a status badge and, separately, the full append-only review
  trail.

Everything here is read-only and side-effect free. ``collect_review_queue()``
reads a ``Repository`` only through its own list helpers
(``list_outcome_learning_signals``,
``list_outcome_learning_signal_reviews``, ``list_all_evidence``) and reuses
the projection helpers from ``user_model_view`` -- it recomputes nothing,
promotes nothing, and re-implements no promotion logic. ``render_reviews_html``
is a pure function of its input (inline CSS reused from ``user_model_view``,
no JavaScript, no approve/reject controls, every value ``html.escape``-d).
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.viewer.user_model_view import (
    _CSS,
    _esc,
    _list_or_empty,
    _nav,
    _table,
    _tag,
    outcome_learning_signal_review_row,
    outcome_learning_signal_row,
)

# ``.tag.pending`` / ``.tag.approved`` / ``.tag.rejected`` / ``.tag.support``
# and the rest are already defined in ``user_model_view._CSS``; this page
# adds no styles of its own.

_PENDING = "pending"


@dataclass
class ReviewQueue:
    """Everything the ``/reviews`` page shows, already shaped into plain
    dict rows."""

    db_path: str
    generated_at: datetime
    user_id: str | None = None
    pending_signals: list[dict[str, Any]] = field(default_factory=list)
    reviewed_signals: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        by_status: dict[str, int] = {}
        for row in self.reviewed_signals:
            by_status[row["review_status"]] = by_status.get(row["review_status"], 0) + 1
        return {
            "signals": len(self.pending_signals) + len(self.reviewed_signals),
            "pending": len(self.pending_signals),
            "approved": by_status.get("approved", 0),
            "rejected": by_status.get("rejected", 0),
            "reviews": len(self.reviews),
        }


def collect_review_queue(
    repo: Any,
    *,
    db_path: str,
    user_id: str | None = None,
    generated_at: datetime | None = None,
) -> ReviewQueue:
    """Read the outcome-learning signals and their manual-review trail out of
    ``repo`` (read-only) and split them into pending vs. reviewed.

    ``repo`` is typed loosely so a ``Repository.readonly_at_path(...)`` view
    or an in-memory test repo both work; it is only ever *read* from here.
    Signals are user-scoped by the repository; reviews carry no user_id of
    their own, so they are scoped to the signals actually shown."""
    generated_at = generated_at or datetime.now(timezone.utc)

    signals = _list_or_empty(repo.list_outcome_learning_signals, user_id=user_id)

    # A signal counts as "promoted" once a repeated_pattern_summary
    # belief_evidence row carrying its independence_group exists -- the same
    # match user_model_view uses. Read-only: this only looks at the ledger.
    promoted_by_group: dict[str, list[str]] = {}
    for row in _list_or_empty(repo.list_all_evidence, user_id=user_id):
        if row.source_type.value == "repeated_pattern_summary":
            promoted_by_group.setdefault(row.independence_group, []).append(row.evidence_id)

    shown_signal_ids = {s.signal_id for s in signals}
    reviews_by_signal: dict[str, list[Any]] = {}
    shown_reviews: list[Any] = []
    for review in _list_or_empty(repo.list_outcome_learning_signal_reviews):
        if review.signal_id not in shown_signal_ids:
            continue
        reviews_by_signal.setdefault(review.signal_id, []).append(review)
        shown_reviews.append(review)

    rows = [
        outcome_learning_signal_row(
            signal,
            promoted_evidence_ids=promoted_by_group.get(signal.independence_group, []),
            reviews=reviews_by_signal.get(signal.signal_id, []),
        )
        for signal in signals
    ]

    return ReviewQueue(
        db_path=db_path,
        generated_at=generated_at,
        user_id=user_id,
        pending_signals=[r for r in rows if r["review_status"] == _PENDING],
        reviewed_signals=[r for r in rows if r["review_status"] != _PENDING],
        reviews=[outcome_learning_signal_review_row(r) for r in shown_reviews],
    )


# --------------------------------------------------------------- rendering --


def _summary_cards(queue: ReviewQueue) -> str:
    summary = queue.summary()
    cards = [
        ("pending review", summary["pending"]),
        ("approved", summary["approved"]),
        ("rejected", summary["rejected"]),
        ("signals total", summary["signals"]),
        ("review decisions", summary["reviews"]),
    ]
    parts = [
        f'<div class="card"><div class="n">{int(n)}</div><div class="l">{html.escape(label)}</div></div>'
        for label, n in cards
    ]
    return f'<div class="cards">{"".join(parts)}</div>'


def _proposed(row: dict[str, Any]) -> str:
    if not row["proposed_evidence"]:
        return _esc(row["proposed_evidence_count"])
    compact = ", ".join(
        f'{p["belief_id"]}:{p["direction"]}@{p["strength"]}' for p in row["proposed_evidence"]
    )
    return f'{row["proposed_evidence_count"]} ({html.escape(compact)})'


def _pending_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["signal_id", "user_id", "context", "kind", "direction", "trials", "supp/adv/neut",
         "belief_ids", "causal_claim", "proposed_evidence", "created_at", "rationale"],
        [
            [
                _esc(r["signal_id"]), _esc(r["user_id"]), _esc(r["recommendation_context"]),
                _tag(r["kind"], r["kind"]),
                _tag(r["direction"], r["direction"]) if r["direction"] else _esc(""),
                _esc(r["trial_count"]),
                f'{r["supportive_count"]}/{r["adverse_count"]}/{r["neutral_count"]}',
                _esc(", ".join(r["belief_ids"])),
                _tag("causal!", "high") if r["causal_claim"] else _tag("none", "not_required"),
                _proposed(r),
                _esc(r["created_at"]),
                _esc(r["rationale"]),
            ]
            for r in rows
        ],
        empty="No outcome-learning signals are waiting for review.",
        wrap_columns={7, 9, 11},
    )
    return f"<h2>Pending review</h2>{table}"


def _proposed_evidence_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["signal_id", "belief_id", "direction", "strength", "context"],
        [
            [
                _esc(r["signal_id"]), _esc(p["belief_id"]),
                _tag(p["direction"], p["direction"]), _esc(p["strength"]),
                _esc(r["recommendation_context"]),
            ]
            for r in rows
            for p in r["proposed_evidence"]
        ],
        empty="No proposed evidence on the pending signals.",
    )
    return f"<h2>Proposed evidence (pending signals)</h2>{table}"


def _reviewed_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["status", "signal_id", "user_id", "context", "kind", "reviewers", "promoted",
         "created_at"],
        [
            [
                _tag(r["review_status"], r["review_status"]),
                _esc(r["signal_id"]), _esc(r["user_id"]), _esc(r["recommendation_context"]),
                _tag(r["kind"], r["kind"]),
                _esc(", ".join(sorted({d["reviewer_id"] for d in r["review_decisions"]}))),
                _tag("promoted", "support") if r["promoted"] else _tag("no", "not_required"),
                _esc(r["created_at"]),
            ]
            for r in rows
        ],
        empty="No signals have been reviewed yet.",
    )
    return f"<h2>Reviewed signals</h2>{table}"


def _review_trail_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["review_id", "signal_id", "reviewer_id", "decision", "promotion", "recompute",
         "promoted_evidence_ids", "recomputed_belief_ids", "created_at", "notes"],
        [
            [
                _esc(r["review_id"]), _esc(r["signal_id"]), _esc(r["reviewer_id"]),
                _tag(r["decision"], r["decision"]),
                _tag("yes", "support") if r["promotion_requested"] else _tag("no", "not_required"),
                _tag("yes", "support") if r["recompute_requested"] else _tag("no", "not_required"),
                _esc(", ".join(r["promoted_evidence_ids"])),
                _esc(", ".join(r["recomputed_belief_ids"])),
                _esc(r["created_at"]),
                _esc(r["notes"]),
            ]
            for r in rows
        ],
        empty="No review decisions recorded.",
        wrap_columns={6, 7, 9},
    )
    return f"<h2>Review decisions</h2>{table}"


def render_reviews_html(queue: ReviewQueue) -> str:
    """Render the review queue to one self-contained HTML page. Pure -- no
    I/O, no scripts, no external resources, no approve/reject controls."""
    scope = ""
    if queue.user_id is not None:
        scope = f" &middot; scope: user_id={html.escape(queue.user_id)}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>User model manual review queue</title>
<style>{_CSS}</style>
</head>
<body>
{_nav("reviews")}
<h1>Adaptive user model &mdash; manual review queue</h1>
<div class="meta">
  <span class="readonly">READ-ONLY</span>
  database: <code>{html.escape(queue.db_path)}</code>{scope}<br>
  generated {html.escape(queue.generated_at.isoformat())}
  &middot; promotion decisions are made from the CLI
  (<code>tools/review_outcome_learning_signal.py</code>), not this page
</div>
{_summary_cards(queue)}
{_pending_section(queue.pending_signals)}
{_proposed_evidence_section(queue.pending_signals)}
{_reviewed_section(queue.reviewed_signals)}
{_review_trail_section(queue.reviews)}
</body>
</html>
"""

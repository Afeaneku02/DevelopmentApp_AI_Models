"""Pure data-shaping and HTML rendering for the read-only adaptive-user-model
viewer (``tools/view_user_model.py``).

Everything here is read-only and side-effect free:

- ``collect_view_model()`` reads a ``Repository`` through its own read/list
  helpers only (``list_events``, ``list_observations``,
  ``list_observation_events_for``, ``list_all_evidence``,
  ``list_latest_beliefs``, ``list_belief_key_canonicalizations``) -- never
  direct SQL, never a write path. It recomputes nothing, suppresses
  nothing, invalidates nothing, canonicalizes nothing.
- Every ``*_row()`` function is a plain projection of one already-persisted
  record onto the handful of fields the viewer shows.
- ``render_html()`` turns the collected view model into a single
  self-contained HTML string (inline CSS, no JavaScript, no external
  requests). It is a pure function of its input.

The viewer deliberately shows exactly what is stored: a belief's
``confidence``/``status`` are displayed as-is, alongside its
``locked_until_recompute`` flag, without any judgement about whether the
cached numbers are still authoritative -- that is the reader's call, and the
lock flag is the signal for it.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.beliefs.canonicalization import BeliefKeyCanonicalization
from src.beliefs.models import BeliefEvidence, UserBelief
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation

# The three mutually exclusive states the viewer buckets evidence into, in
# precedence order: an invalidated row is "inactive" even if it was also
# once flagged a duplicate; a still-live row that is duplicate-suppressed is
# "duplicate_suppressed"; anything else is "active". This matches
# ``src.beliefs.models.active_evidence`` (active == is_active and not
# is_duplicate_suppressed) while also separating out the inactive case.
EVIDENCE_STATE_ACTIVE = "active"
EVIDENCE_STATE_INACTIVE = "inactive"
EVIDENCE_STATE_DUPLICATE_SUPPRESSED = "duplicate_suppressed"
EVIDENCE_STATES = (
    EVIDENCE_STATE_ACTIVE,
    EVIDENCE_STATE_INACTIVE,
    EVIDENCE_STATE_DUPLICATE_SUPPRESSED,
)


def evidence_state(evidence: BeliefEvidence) -> str:
    """Bucket one evidence row into exactly one of ``EVIDENCE_STATES``."""
    if not evidence.is_active:
        return EVIDENCE_STATE_INACTIVE
    if evidence.is_duplicate_suppressed:
        return EVIDENCE_STATE_DUPLICATE_SUPPRESSED
    return EVIDENCE_STATE_ACTIVE


def _fmt(value: Any) -> str:
    """Stable, human-readable string for a scalar field value."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def event_row(event: UserEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "event_type": event.event_type,
        "timestamp": _fmt(event.timestamp),
        "source": event.source,
        "goal_id": event.goal_id,
        "session_id": event.session_id,
        "has_structured_data": event.structured_data is not None,
    }


def observation_row(
    observation: UserObservation, links: list[ObservationEvent]
) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "user_id": observation.user_id,
        "category": observation.category,
        "observation": observation.observation,
        "importance": observation.importance,
        "confidence": observation.confidence,
        "created_at": _fmt(observation.created_at),
        "linked_event_ids": [link.event_id for link in links],
    }


def observation_event_row(link: ObservationEvent) -> dict[str, Any]:
    return {
        "observation_id": link.observation_id,
        "event_id": link.event_id,
        "link_role": link.link_role.value,
        "created_at": _fmt(link.created_at),
    }


def evidence_row(evidence: BeliefEvidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "belief_id": evidence.belief_id,
        "user_id": evidence.user_id,
        "direction": evidence.direction.value,
        "state": evidence_state(evidence),
        "is_active": evidence.is_active,
        "is_duplicate_suppressed": evidence.is_duplicate_suppressed,
        "source_type": evidence.source_type.value,
        "context_key": evidence.context_key,
        "strength": evidence.strength,
        "source_reliability": evidence.source_reliability,
        "source_event_ids": list(evidence.source_event_ids),
        "observed_at": _fmt(evidence.observed_at),
        "suppression_reason": evidence.suppression_reason,
        "invalidated_at": _fmt(evidence.invalidated_at),
        "invalidation_reason": evidence.invalidation_reason,
    }


def belief_row(belief: UserBelief) -> dict[str, Any]:
    """The belief fields the viewer highlights: belief_key, confidence,
    status, locked_until_recompute, and the evidence counts."""
    return {
        "belief_id": belief.belief_id,
        "user_id": belief.user_id,
        "belief_type": belief.belief_type.value,
        "belief_key": belief.belief_key,
        "belief_value": _fmt(belief.belief_value),
        "confidence": belief.confidence,
        "status": belief.status.value,
        "locked_until_recompute": belief.locked_until_recompute,
        "supporting_evidence_count": belief.supporting_evidence_count,
        "contradicting_evidence_count": belief.contradicting_evidence_count,
        "total_evidence_count": belief.total_evidence_count,
        "effective_evidence_count": belief.effective_evidence_count,
        "first_observed": _fmt(belief.first_observed),
        "last_validated": _fmt(belief.last_validated),
        "last_successful_recompute_at": _fmt(belief.last_successful_recompute_at),
    }


def canonicalization_row(decision: BeliefKeyCanonicalization) -> dict[str, Any]:
    return {
        "canonicalization_id": decision.canonicalization_id,
        "user_id": decision.user_id,
        "belief_type": decision.belief_type.value,
        "proposed_key": decision.proposed_key,
        "canonical_key": decision.canonical_key,
        "decision": decision.decision.value,
        "authorized_by": decision.authorized_by,
        "authorized_at": _fmt(decision.authorized_at),
        "registry_version": decision.canonical_key_registry_version,
        "decision_reason": decision.decision_reason,
    }


@dataclass
class ViewModel:
    """Everything the viewer shows, already shaped into plain dict rows."""

    db_path: str
    generated_at: datetime
    user_id: str | None = None
    belief_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    observation_events: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    canonicalizations: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return summarize(self)


def summarize(view_model: ViewModel) -> dict[str, Any]:
    """Small headline counts for the top of the page."""
    evidence_by_state = {state: 0 for state in EVIDENCE_STATES}
    for row in view_model.evidence:
        evidence_by_state[row["state"]] = evidence_by_state.get(row["state"], 0) + 1

    beliefs_by_status: dict[str, int] = {}
    locked_beliefs = 0
    for row in view_model.beliefs:
        beliefs_by_status[row["status"]] = beliefs_by_status.get(row["status"], 0) + 1
        if row["locked_until_recompute"]:
            locked_beliefs += 1

    canonicalizations_by_decision: dict[str, int] = {}
    for row in view_model.canonicalizations:
        key = row["decision"]
        canonicalizations_by_decision[key] = canonicalizations_by_decision.get(key, 0) + 1

    return {
        "events": len(view_model.events),
        "observations": len(view_model.observations),
        "observation_events": len(view_model.observation_events),
        "evidence": len(view_model.evidence),
        "evidence_by_state": evidence_by_state,
        "beliefs": len(view_model.beliefs),
        "beliefs_by_status": beliefs_by_status,
        "locked_beliefs": locked_beliefs,
        "canonicalizations": len(view_model.canonicalizations),
        "canonicalizations_by_decision": canonicalizations_by_decision,
    }


def collect_view_model(
    repo: Any,
    *,
    db_path: str,
    user_id: str | None = None,
    belief_id: str | None = None,
    generated_at: datetime | None = None,
) -> ViewModel:
    """Read the whole (optionally user/belief-scoped) picture out of a
    ``Repository`` using only its read/list helpers. ``repo`` is typed
    loosely on purpose so a ``Repository.readonly_at_path(...)`` or an
    in-memory test repo both work; it is only ever *read* from here."""
    generated_at = generated_at or datetime.now(timezone.utc)

    events = repo.list_events(user_id=user_id)
    observations = repo.list_observations(user_id=user_id)
    links = repo.list_observation_events_for([o.observation_id for o in observations])
    links_by_observation: dict[str, list[ObservationEvent]] = {}
    for link in links:
        links_by_observation.setdefault(link.observation_id, []).append(link)
    ordered_links = sorted(links, key=lambda link: (link.observation_id, link.event_id))

    evidence = repo.list_all_evidence(user_id=user_id, belief_id=belief_id)
    beliefs = repo.list_latest_beliefs(user_id=user_id, belief_id=belief_id)
    canonicalizations = repo.list_belief_key_canonicalizations(user_id=user_id)

    return ViewModel(
        db_path=db_path,
        generated_at=generated_at,
        user_id=user_id,
        belief_id=belief_id,
        events=[event_row(e) for e in events],
        observations=[
            observation_row(o, links_by_observation.get(o.observation_id, []))
            for o in observations
        ],
        observation_events=[observation_event_row(link) for link in ordered_links],
        evidence=[evidence_row(e) for e in evidence],
        beliefs=[belief_row(b) for b in beliefs],
        canonicalizations=[canonicalization_row(c) for c in canonicalizations],
    )


# --------------------------------------------------------------- rendering --

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
       background: #f6f7f9; color: #1b1f24; }
@media (prefers-color-scheme: dark) { body { background: #14171a; color: #e6e6e6; } }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; }
.meta { color: #666; font-size: .85rem; margin-bottom: 1rem; }
.readonly { display: inline-block; padding: .1rem .5rem; border-radius: 4px;
            background: #0a7d33; color: #fff; font-size: .75rem; letter-spacing: .03em; }
.cards { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0; }
.card { background: #fff; border: 1px solid #d9dde1; border-radius: 6px; padding: .5rem .75rem; min-width: 8rem; }
@media (prefers-color-scheme: dark) { .card { background: #1e2226; border-color: #333; } }
.card .n { font-size: 1.3rem; font-weight: 600; }
.card .l { color: #777; font-size: .8rem; }
table { border-collapse: collapse; width: 100%; background: #fff; margin-bottom: .5rem;
        display: block; overflow-x: auto; }
@media (prefers-color-scheme: dark) { table { background: #1e2226; } }
th, td { border: 1px solid #d9dde1; padding: .35rem .5rem; text-align: left; vertical-align: top;
         white-space: nowrap; }
@media (prefers-color-scheme: dark) { th, td { border-color: #333; } }
th { background: #eef1f4; font-weight: 600; }
@media (prefers-color-scheme: dark) { th { background: #262b30; } }
td.wrap { white-space: normal; min-width: 18rem; }
.tag { display: inline-block; padding: .05rem .4rem; border-radius: 3px; font-size: .78rem; font-weight: 600; }
.tag.active { background: #0a7d33; color: #fff; }
.tag.inactive { background: #8a8f95; color: #fff; }
.tag.duplicate_suppressed { background: #b5860b; color: #fff; }
.tag.locked { background: #b02a37; color: #fff; }
.tag.unlocked { background: #e7ebf0; color: #444; border: 1px solid #ccc; }
.tag.status { background: #2f6fb0; color: #fff; }
.tag.decision { background: #5a4b8a; color: #fff; }
.empty { color: #888; font-style: italic; margin-bottom: .5rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(_fmt(value))


def _tag(text: str, css_class: str) -> str:
    return f'<span class="tag {html.escape(css_class)}">{html.escape(text)}</span>'


def _table(
    headers: list[str], rows: list[list[str]], *, empty: str, wrap_columns: set[int] | None = None
) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    wrap_columns = wrap_columns or set()
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td class="wrap">{cell}</td>' if i in wrap_columns else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _summary_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("events", summary["events"]),
        ("observations", summary["observations"]),
        ("obs-event links", summary["observation_events"]),
        ("evidence", summary["evidence"]),
        ("beliefs", summary["beliefs"]),
        ("locked beliefs", summary["locked_beliefs"]),
        ("canonicalizations", summary["canonicalizations"]),
    ]
    parts = [
        f'<div class="card"><div class="n">{int(n)}</div><div class="l">{html.escape(label)}</div></div>'
        for label, n in cards
    ]
    es = summary["evidence_by_state"]
    parts.append(
        '<div class="card"><div class="n">'
        f'{int(es.get("active", 0))}/{int(es.get("inactive", 0))}/{int(es.get("duplicate_suppressed", 0))}'
        '</div><div class="l">evidence active/inactive/suppressed</div></div>'
    )
    return f'<div class="cards">{"".join(parts)}</div>'


def _events_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["event_id", "user_id", "event_type", "timestamp", "source", "goal_id", "session_id", "structured_data"],
        [
            [
                _esc(r["event_id"]), _esc(r["user_id"]), _esc(r["event_type"]), _esc(r["timestamp"]),
                _esc(r["source"]), _esc(r["goal_id"]), _esc(r["session_id"]),
                "yes" if r["has_structured_data"] else "",
            ]
            for r in rows
        ],
        empty="No events stored.",
    )
    return f"<h2>Events</h2>{table}"


def _observations_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["observation_id", "user_id", "category", "importance", "confidence", "created_at",
         "linked_event_ids", "observation"],
        [
            [
                _esc(r["observation_id"]), _esc(r["user_id"]), _esc(r["category"]),
                _esc(r["importance"]), _esc(r["confidence"]), _esc(r["created_at"]),
                _esc(", ".join(r["linked_event_ids"])),
                _esc(r["observation"]),
            ]
            for r in rows
        ],
        empty="No observations stored.",
        wrap_columns={7},
    )
    return f"<h2>Observations</h2>{table}"


def _observation_events_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["observation_id", "event_id", "link_role", "created_at"],
        [
            [_esc(r["observation_id"]), _esc(r["event_id"]), _esc(r["link_role"]), _esc(r["created_at"])]
            for r in rows
        ],
        empty="No observation-event links stored.",
    )
    return f"<h2>Observation-event links</h2>{table}"


def _evidence_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["evidence_id", "belief_id", "state", "direction", "source_type", "context_key",
         "strength", "reliability", "source_event_ids", "reason"],
        [
            [
                _esc(r["evidence_id"]), _esc(r["belief_id"]),
                _tag(r["state"], r["state"]), _esc(r["direction"]), _esc(r["source_type"]),
                _esc(r["context_key"]), _esc(r["strength"]), _esc(r["source_reliability"]),
                _esc(", ".join(r["source_event_ids"])),
                _esc(r["invalidation_reason"] or r["suppression_reason"]),
            ]
            for r in rows
        ],
        empty="No evidence stored.",
        wrap_columns={9},
    )
    return f"<h2>Evidence</h2>{table}"


def _beliefs_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["belief_id", "belief_key", "confidence", "status", "lock", "value",
         "support", "contradict", "total", "effective", "last_validated"],
        [
            [
                _esc(r["belief_id"]), _esc(r["belief_key"]),
                f'{r["confidence"]:.4f}',
                _tag(r["status"], "status"),
                _tag("LOCKED", "locked") if r["locked_until_recompute"] else _tag("unlocked", "unlocked"),
                _esc(r["belief_value"]),
                _esc(r["supporting_evidence_count"]), _esc(r["contradicting_evidence_count"]),
                _esc(r["total_evidence_count"]), f'{r["effective_evidence_count"]:.4f}',
                _esc(r["last_validated"]),
            ]
            for r in rows
        ],
        empty="No beliefs stored.",
    )
    return f"<h2>Beliefs</h2>{table}"


def _canonicalizations_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["canonicalization_id", "belief_type", "proposed_key", "canonical_key", "decision",
         "authorized_at", "reason"],
        [
            [
                _esc(r["canonicalization_id"]), _esc(r["belief_type"]),
                _esc(r["proposed_key"]), _esc(r["canonical_key"]),
                _tag(r["decision"], "decision"), _esc(r["authorized_at"]),
                _esc(r["decision_reason"]),
            ]
            for r in rows
        ],
        empty="No belief-key canonicalization decisions stored.",
        wrap_columns={6},
    )
    return f"<h2>Belief-key canonicalization decisions</h2>{table}"


def render_html(view_model: ViewModel) -> str:
    """Render the whole view model to one self-contained HTML page. Pure --
    no I/O, no external resources, no scripts."""
    summary = summarize(view_model)
    scope_bits = []
    if view_model.user_id is not None:
        scope_bits.append(f"user_id={view_model.user_id}")
    if view_model.belief_id is not None:
        scope_bits.append(f"belief_id={view_model.belief_id}")
    scope = f" &middot; scope: {html.escape(', '.join(scope_bits))}" if scope_bits else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>User model viewer &mdash; {html.escape(view_model.db_path)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Adaptive user model &mdash; read-only viewer</h1>
<div class="meta">
  <span class="readonly">READ-ONLY</span>
  database: <code>{html.escape(view_model.db_path)}</code>{scope}<br>
  generated {html.escape(view_model.generated_at.isoformat())}
</div>
{_summary_cards(summary)}
{_events_section(view_model.events)}
{_observations_section(view_model.observations)}
{_observation_events_section(view_model.observation_events)}
{_evidence_section(view_model.evidence)}
{_beliefs_section(view_model.beliefs)}
{_canonicalizations_section(view_model.canonicalizations)}
</body>
</html>
"""

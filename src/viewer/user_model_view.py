"""Pure data-shaping and HTML rendering for the read-only adaptive-user-model
viewer (``tools/view_user_model.py``).

Everything here is read-only and side-effect free:

- ``collect_view_model()`` reads a ``Repository`` through its own read/list
  helpers only (``list_events``, ``list_observations``,
  ``list_observation_events_for``, ``list_all_evidence``,
  ``list_latest_beliefs``, ``list_belief_key_canonicalizations``,
  ``list_recommendations``, ``list_recommendation_outcomes``,
  ``list_outcome_learning_signals``) -- never direct SQL, never a write
  path. It recomputes nothing, suppresses nothing, invalidates nothing,
  canonicalizes nothing, issues no recommendation, and promotes no
  outcome-learning proposal into evidence. A database opened before the
  recommendation tables existed simply shows those sections empty (see
  ``_list_or_empty``).
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
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.beliefs.canonicalization import BeliefKeyCanonicalization
from src.beliefs.models import BeliefEvidence, UserBelief
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation
from src.recommendations.models import (
    OutcomeLearningSignal,
    RecommendationOutcome,
    UserRecommendation,
)

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


def recommendation_row(recommendation: UserRecommendation) -> dict[str, Any]:
    """The recommendation fields the viewer highlights: the resolved risk
    context, the manual-review gate, the ranking numbers, the beliefs used,
    and the recommendation text itself."""
    return {
        "recommendation_id": recommendation.recommendation_id,
        "user_id": recommendation.user_id,
        "recommendation_context": recommendation.recommendation_context,
        "proposed_context_key": recommendation.proposed_context_key,
        "risk_tier": recommendation.risk_tier.value,
        "risk_resolution_path": recommendation.risk_resolution_path.value,
        "review_required": recommendation.review_required,
        "review_status": recommendation.review_status.value,
        "required_resolution_mode": (
            recommendation.required_resolution_mode.value
            if recommendation.required_resolution_mode is not None
            else None
        ),
        "ranking_score": recommendation.ranking_score,
        "confidence": recommendation.confidence,
        "belief_ids_used": list(recommendation.belief_ids_used),
        "blocked_belief_count": len(recommendation.blocked_beliefs),
        "recommendation": recommendation.recommendation,
        "goal": recommendation.goal,
        "created_at": _fmt(recommendation.created_at),
    }


def recommendation_outcome_row(outcome: RecommendationOutcome) -> dict[str, Any]:
    return {
        "outcome_id": outcome.outcome_id,
        "recommendation_id": outcome.recommendation_id,
        "followed": outcome.followed.value,
        "result": outcome.result.value,
        "source": outcome.source,
        "observed_at": _fmt(outcome.observed_at),
        "created_at": _fmt(outcome.created_at),
        "user_feedback": outcome.user_feedback,
        "measured_result": outcome.measured_result,
    }


def outcome_learning_signal_row(
    signal: OutcomeLearningSignal, *, promoted_evidence_ids: list[str] | None = None
) -> dict[str, Any]:
    """One outcome-learning signal, projected onto the fields the viewer
    shows. ``proposed_evidence`` is summarised, not expanded: its count plus
    a compact ``belief_id:direction@strength`` list, so the section stays
    readable while still showing what evidence was proposed.

    ``promoted_evidence_ids`` (the ``belief_evidence`` rows that were
    authorized from this signal -- matched by its ``independence_group``)
    is supplied by ``collect_view_model``; ``promoted`` is ``True`` when any
    exist, so the viewer shows which signals have actually become evidence.
    """
    promoted_evidence_ids = sorted(promoted_evidence_ids or [])
    proposed = [
        {
            "belief_id": p.belief_id,
            "direction": p.direction.value,
            "strength": p.strength,
        }
        for p in signal.proposed_evidence
    ]
    return {
        "signal_id": signal.signal_id,
        "user_id": signal.user_id,
        "recommendation_context": signal.recommendation_context,
        "kind": signal.kind.value,
        "direction": signal.direction.value if signal.direction is not None else None,
        "trial_count": signal.trial_count,
        "supportive_count": signal.supportive_count,
        "adverse_count": signal.adverse_count,
        "neutral_count": signal.neutral_count,
        "belief_ids": list(signal.belief_ids),
        "recommendation_ids": list(signal.recommendation_ids),
        "outcome_ids": list(signal.outcome_ids),
        "independence_group": signal.independence_group,
        "causal_claim": signal.causal_claim,
        "created_at": _fmt(signal.created_at),
        "rationale": signal.rationale,
        "proposed_evidence_count": len(proposed),
        "proposed_evidence": proposed,
        "promoted": bool(promoted_evidence_ids),
        "promoted_evidence_ids": promoted_evidence_ids,
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
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    recommendation_outcomes: list[dict[str, Any]] = field(default_factory=list)
    outcome_learning_signals: list[dict[str, Any]] = field(default_factory=list)

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

    recommendations_by_risk_tier: dict[str, int] = {}
    review_required_recommendations = 0
    for row in view_model.recommendations:
        recommendations_by_risk_tier[row["risk_tier"]] = (
            recommendations_by_risk_tier.get(row["risk_tier"], 0) + 1
        )
        if row["review_required"]:
            review_required_recommendations += 1

    outcomes_by_followed: dict[str, int] = {}
    for row in view_model.recommendation_outcomes:
        outcomes_by_followed[row["followed"]] = outcomes_by_followed.get(row["followed"], 0) + 1

    signals_by_kind: dict[str, int] = {}
    proposed_evidence_rows = 0
    promoted_signals = 0
    for row in view_model.outcome_learning_signals:
        signals_by_kind[row["kind"]] = signals_by_kind.get(row["kind"], 0) + 1
        proposed_evidence_rows += row["proposed_evidence_count"]
        if row["promoted"]:
            promoted_signals += 1

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
        "recommendations": len(view_model.recommendations),
        "recommendations_by_risk_tier": recommendations_by_risk_tier,
        "review_required_recommendations": review_required_recommendations,
        "recommendation_outcomes": len(view_model.recommendation_outcomes),
        "outcomes_by_followed": outcomes_by_followed,
        "outcome_learning_signals": len(view_model.outcome_learning_signals),
        "outcome_learning_signals_by_kind": signals_by_kind,
        "outcome_learning_proposed_evidence": proposed_evidence_rows,
        "outcome_learning_signals_promoted": promoted_signals,
    }


def _list_or_empty(list_fn: Callable[..., list[Any]], **kwargs: Any) -> list[Any]:
    """Call a ``Repository.list_*`` helper, but treat a missing table as an
    empty result rather than an error. ``readonly_at_path`` never runs the
    schema, so a database created before the recommendation tables existed
    would otherwise raise ``sqlite3.OperationalError`` here -- the viewer's
    job is to show what is there, and "nothing" is a valid answer."""
    try:
        return list_fn(**kwargs)
    except sqlite3.OperationalError:
        return []


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

    recommendations = _list_or_empty(repo.list_recommendations, user_id=user_id)
    # Outcomes carry no user_id; scope them to the recommendations shown so a
    # user-scoped page never leaks another user's outcome rows.
    shown_recommendation_ids = {r.recommendation_id for r in recommendations}
    recommendation_outcomes = [
        o
        for o in _list_or_empty(repo.list_recommendation_outcomes)
        if o.recommendation_id in shown_recommendation_ids
    ]
    # Learning signals do carry user_id, so the repository filters them directly.
    outcome_learning_signals = _list_or_empty(
        repo.list_outcome_learning_signals, user_id=user_id
    )
    # A signal is "promoted" once a repeated_pattern_summary belief_evidence
    # row carrying its independence_group exists in the ledger (that is what
    # promote_outcome_learning_signal writes). Match against the user's whole
    # ledger, not the possibly belief-scoped ``evidence`` list above.
    promoted_by_group: dict[str, list[str]] = {}
    for row in _list_or_empty(repo.list_all_evidence, user_id=user_id):
        if row.source_type.value == "repeated_pattern_summary":
            promoted_by_group.setdefault(row.independence_group, []).append(row.evidence_id)

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
        recommendations=[recommendation_row(r) for r in recommendations],
        recommendation_outcomes=[
            recommendation_outcome_row(o) for o in recommendation_outcomes
        ],
        outcome_learning_signals=[
            outcome_learning_signal_row(
                s, promoted_evidence_ids=promoted_by_group.get(s.independence_group, [])
            )
            for s in outcome_learning_signals
        ],
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
.tag.low, .tag.followed, .tag.successful { background: #0a7d33; color: #fff; }
.tag.medium, .tag.partially_followed, .tag.mixed, .tag.pending { background: #b5860b; color: #fff; }
.tag.high, .tag.unsuccessful { background: #b02a37; color: #fff; }
.tag.not_followed, .tag.ignored { background: #8a8f95; color: #fff; }
.tag.support { background: #0a7d33; color: #fff; }
.tag.weak_contradiction { background: #b02a37; color: #fff; }
.tag.no_signal { background: #e7ebf0; color: #444; border: 1px solid #ccc; }
.tag.contradict { background: #b02a37; color: #fff; }
.tag.not_required, .tag.unknown, .tag.not_yet_known {
    background: #e7ebf0; color: #444; border: 1px solid #ccc; }
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
        ("recommendations", summary["recommendations"]),
        ("recs awaiting review", summary["review_required_recommendations"]),
        ("outcomes", summary["recommendation_outcomes"]),
        ("learning signals", summary["outcome_learning_signals"]),
        ("proposed evidence", summary["outcome_learning_proposed_evidence"]),
        ("promoted signals", summary["outcome_learning_signals_promoted"]),
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


def _recommendations_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["recommendation_id", "user_id", "context", "risk_tier", "review", "review_status",
         "resolution_mode", "ranking_score", "confidence", "belief_ids_used", "blocked",
         "created_at", "recommendation"],
        [
            [
                _esc(r["recommendation_id"]), _esc(r["user_id"]),
                _esc(r["recommendation_context"]),
                _tag(r["risk_tier"], r["risk_tier"]),
                _tag("REVIEW", "high") if r["review_required"] else _tag("auto", "not_required"),
                _tag(r["review_status"], r["review_status"]),
                _esc(r["required_resolution_mode"]),
                f'{r["ranking_score"]:.4f}', f'{r["confidence"]:.4f}',
                _esc(", ".join(r["belief_ids_used"])),
                _esc(r["blocked_belief_count"]),
                _esc(r["created_at"]),
                _esc(r["recommendation"]),
            ]
            for r in rows
        ],
        empty="No recommendations stored.",
        wrap_columns={12},
    )
    return f"<h2>Recommendations</h2>{table}"


def _recommendation_outcomes_section(rows: list[dict[str, Any]]) -> str:
    table = _table(
        ["outcome_id", "recommendation_id", "followed", "result", "source",
         "observed_at", "created_at", "user_feedback", "measured_result"],
        [
            [
                _esc(r["outcome_id"]), _esc(r["recommendation_id"]),
                _tag(r["followed"], r["followed"]), _tag(r["result"], r["result"]),
                _esc(r["source"]), _esc(r["observed_at"]), _esc(r["created_at"]),
                _esc(r["user_feedback"]), _esc(r["measured_result"]),
            ]
            for r in rows
        ],
        empty="No recommendation outcomes stored.",
        wrap_columns={7, 8},
    )
    return f"<h2>Recommendation outcomes</h2>{table}"


def _outcome_learning_signals_section(rows: list[dict[str, Any]]) -> str:
    def _proposed(row: dict[str, Any]) -> str:
        if not row["proposed_evidence"]:
            return _esc(row["proposed_evidence_count"])
        compact = ", ".join(
            f'{p["belief_id"]}:{p["direction"]}@{p["strength"]}' for p in row["proposed_evidence"]
        )
        return f'{row["proposed_evidence_count"]} ({html.escape(compact)})'

    def _promoted(row: dict[str, Any]) -> str:
        if not row["promoted"]:
            return _tag("no", "not_required")
        return f'{_tag("promoted", "support")} {html.escape(", ".join(row["promoted_evidence_ids"]))}'

    table = _table(
        ["signal_id", "user_id", "context", "kind", "direction", "trials",
         "supp/adv/neut", "belief_ids", "recommendation_ids", "outcome_ids",
         "causal_claim", "proposed_evidence", "promoted", "created_at", "rationale"],
        [
            [
                _esc(r["signal_id"]), _esc(r["user_id"]), _esc(r["recommendation_context"]),
                _tag(r["kind"], r["kind"]),
                _tag(r["direction"], r["direction"]) if r["direction"] else _esc(""),
                _esc(r["trial_count"]),
                f'{r["supportive_count"]}/{r["adverse_count"]}/{r["neutral_count"]}',
                _esc(", ".join(r["belief_ids"])),
                _esc(", ".join(r["recommendation_ids"])),
                _esc(", ".join(r["outcome_ids"])),
                _tag("causal!", "high") if r["causal_claim"] else _tag("none", "not_required"),
                _proposed(r),
                _promoted(r),
                _esc(r["created_at"]),
                _esc(r["rationale"]),
            ]
            for r in rows
        ],
        empty="No outcome-learning signals stored.",
        wrap_columns={8, 9, 14},
    )
    return f"<h2>Outcome learning signals</h2>{table}"


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
{_recommendations_section(view_model.recommendations)}
{_recommendation_outcomes_section(view_model.recommendation_outcomes)}
{_outcome_learning_signals_section(view_model.outcome_learning_signals)}
</body>
</html>
"""

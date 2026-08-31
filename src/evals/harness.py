"""Scenario-driven evaluation harness for the adaptive user model.

A *scenario manifest* is a JSON object with three parts::

    {
      "name": "clean_support_raises_confidence",
      "description": "clean supporting evidence raises confidence",
      "steps":  [ {"op": "...", ...}, ... ],
      "expect": [ {"check": "...", ...}, ... ]
    }

``run_scenario()`` builds a fresh in-memory ``Repository``, replays every
step through the *same* sanctioned functions the real CLIs use
(``recompute_belief``, ``generate_recommendation``,
``analyze_recommendation_outcomes``, ``review_outcome_learning_signal``,
``promote_outcome_learning_signal``, ``Repository.mark_evidence_inactive``,
...), then evaluates every ``expect`` check against the final database. No
model behaviour is added or re-implemented here -- this is pure validation
around the system already built.

Steps (``op``):

- ``event``           -- insert a ``UserEvent`` (``id``, ``event_type``,
  ``days_before_as_of``).
- ``observation``     -- ``create_observation_from_event`` (``id``,
  ``event``, ``category``, ``text``, ``importance``, ``confidence``).
- ``evidence``        -- propose + authorize + insert one ``belief_evidence``
  leaf for ``belief``. Uses an explicit ``observation`` ref, or auto-creates
  a hidden event+observation when none is given. Fields: ``id``,
  ``direction``, ``source_type``, ``strength``, ``belief_type``,
  ``context_key``, ``days_before_as_of``.
- ``recompute``       -- ``recompute_belief`` for ``belief`` from its ledger
  (``belief_key``, ``belief_value``, ``belief_type``, optional
  ``reason``/``no_active_evidence_status`` for the invalidation branch,
  optional ``checkpoint`` name to snapshot the result).
- ``checkpoint``      -- snapshot a belief's current confidence/status under
  ``name`` for later ``gt_checkpoint`` / ``lt_checkpoint`` comparisons.
- ``lock``            -- ``Repository.lock_belief_until_recompute`` for
  ``belief``.
- ``invalidate_evidence`` -- ``Repository.mark_evidence_inactive`` for
  ``evidence`` (one id or a list); also fail-closes the belief.
- ``recommendation``  -- ``generate_recommendation`` from every latest belief
  (``id``, ``context_key``, ``goal``) and persist it.
- ``outcome``         -- append one ``RecommendationOutcome`` for
  ``recommendation`` (``followed``, ``result``, ``source``, ``day_offset``).
- ``learn``           -- ``analyze_recommendation_outcomes`` over the stored
  recommendations/outcomes; insert every signal produced. ``id`` binds the
  signal for this context (``context``) to a name.
- ``review``          -- ``review_outcome_learning_signal`` for ``signal``
  (``review_id``, ``reviewer_id``, ``decision``, ``promote``, ``recompute``,
  ``allow_duplicate``, ``notes``).
- ``promote``         -- ``promote_outcome_learning_signal`` for ``signal``
  directly (``persist``, ``recompute``).

Checks (``check``) -- every check carries one or more comparison keys
(``equals`` / ``in`` / ``min`` / ``max`` / ``gt`` / ``lt`` /
``eq_checkpoint`` / ``gt_checkpoint`` / ``lt_checkpoint`` / ``approx`` +
``tol``); all present must hold:

- ``checkpoint``          -- ``name`` + ``field`` (default ``confidence``;
  also ``status``): compares a value snapshotted by a ``checkpoint`` step or
  a ``recompute`` step's ``checkpoint``.
- ``belief_field``        -- ``belief`` + ``field`` (``confidence``,
  ``status``, ``locked_until_recompute``, ``supporting_evidence_count``,
  ``contradicting_evidence_count``, ``total_evidence_count``,
  ``effective_support_count``, ``effective_evidence_count``,
  ``belief_value``, ``belief_key``).
- ``belief_confidence`` / ``belief_status`` / ``belief_locked`` -- sugar for
  the three most common ``belief_field`` cases.
- ``belief_evidence_count`` -- ``belief`` + optional ``active`` (default
  true).
- ``recommendation_uses_only`` -- ``recommendation`` + ``allowed`` (belief
  refs): ``belief_ids_used`` must be a subset.
- ``recommendation_excludes`` -- ``recommendation`` + ``belief``; optional
  ``require_blocked`` to also require it in ``blocked_beliefs``.
- ``recommendation_field`` -- ``recommendation`` + ``field`` (``risk_tier``,
  ``review_required``, ``review_status``, ``recommendation_context``,
  ``ranking_score``, ``confidence``).
- ``signal_exists``       -- ``signal`` ref or ``context``; compares a bool.
- ``signal_field``        -- ``signal`` + ``field`` (``kind``,
  ``causal_claim``, ``direction``, ``trial_count``, ``supportive_count``,
  ``adverse_count``, ``neutral_count``, ``proposed_evidence_count``).
- ``signal_proposed_strength`` -- ``signal``; compares the max proposed
  evidence strength (0.0 when there are none).
- ``signal_proposed_source_types`` -- ``signal``; the set of proposed
  evidence source types must equal ``equals`` (string or list).
- ``review_status``       -- ``signal``; the latest review decision, or
  ``"pending"``.
- ``promoted_evidence_count`` -- ``signal``; active
  ``repeated_pattern_summary`` rows carrying the signal's
  ``independence_group``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.beliefs.models import authorize_evidence
from src.beliefs.propose_evidence import propose_evidence_from_observation_validated
from src.beliefs.recompute import recompute_belief
from src.common.enums import BeliefStatus, SourceType
from src.events.models import UserEvent
from src.observations.create_observation import create_observation_from_event
from src.recommendations.engine import generate_recommendation
from src.recommendations.models import RecommendationOutcome
from src.recommendations.outcome_learning import analyze_recommendation_outcomes
from src.recommendations.promotion import promote_outcome_learning_signal
from src.recommendations.review import review_outcome_learning_signal
from src.storage.repository import Repository

_VERSION_FIELDS = dict(
    schema_version="6",
    scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6",
    policy_version="policy-0.6",
)
_DEFAULT_AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
_PROMOTION_DELAY = timedelta(days=30)
_AGGREGATION_POLICY_VERSION = "evidence-aggregation-0.6"


class ScenarioError(RuntimeError):
    """A manifest could not be executed as written (unknown op, missing
    reference, a sanctioned function rejected the step). The scenario is
    reported as failed."""


# --------------------------------------------------------------- results --


@dataclass
class CheckResult:
    check: str
    target: str
    expected: str
    actual: str
    passed: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "target": self.target,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class ScenarioResult:
    name: str
    description: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    source: str | None = None

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "error": self.error,
            "source": self.source,
            "checks": [c.to_dict() for c in self.checks],
            "summary": {"total": self.total_checks, "passed": self.passed_checks},
        }


@dataclass
class EvalReport:
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.scenarios) and all(s.passed for s in self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        passed = sum(1 for s in self.scenarios if s.passed)
        return {
            "passed": self.passed,
            "summary": {
                "scenarios": len(self.scenarios),
                "passed": passed,
                "failed": len(self.scenarios) - passed,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ------------------------------------------------------------ execution --


def _parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _require(step: dict[str, Any], key: str, *, op: str) -> Any:
    if key not in step:
        raise ScenarioError(f"{op} step is missing required field {key!r}")
    return step[key]


class _ScenarioRunner:
    def __init__(self, manifest: dict[str, Any], *, source: str | None = None) -> None:
        if not isinstance(manifest, dict):
            raise ScenarioError("scenario manifest must be a JSON object")
        self.source = source
        self.name = str(manifest.get("name") or (Path(source).stem if source else "scenario"))
        self.description = str(manifest.get("description", ""))
        self.user_id = str(manifest.get("user_id", "eval_user"))
        as_of_raw = manifest.get("as_of")
        self.as_of = _parse_timestamp(as_of_raw) if as_of_raw else _DEFAULT_AS_OF
        self.promote_as_of = self.as_of + _PROMOTION_DELAY
        self.steps = manifest.get("steps", [])
        self.expect = manifest.get("expect", [])
        if not isinstance(self.steps, list) or not isinstance(self.expect, list):
            raise ScenarioError('"steps" and "expect" must both be arrays')

        self.repo = Repository.in_memory()
        self.clock = self.as_of
        self.events: dict[str, UserEvent] = {}
        self.observations: dict[str, tuple[Any, list[Any]]] = {}
        self.evidence: dict[str, Any] = {}
        self.beliefs: set[str] = set()
        self.recs: dict[str, str] = {}
        self.signals: dict[str, str] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self._outcome_seq = 0

    # -- helpers ---------------------------------------------------------

    def belief_ref(self, name: Any) -> str:
        return f"{self.user_id}__belief__{name}"

    def _event(self, ref: str) -> UserEvent:
        if ref not in self.events:
            raise ScenarioError(f"reference to unknown event {ref!r}")
        return self.events[ref]

    def _signal_id(self, ref: Any) -> str:
        return self.signals.get(str(ref), str(ref))

    def _require_signal(self, ref: Any) -> Any:
        signal = self.repo.get_outcome_learning_signal(self._signal_id(ref))
        if signal is None:
            raise ScenarioError(f"no outcome-learning signal bound to / stored for {ref!r}")
        return signal

    def _require_belief(self, belief_id: str) -> Any:
        belief = self.repo.get_latest_belief(user_id=self.user_id, belief_id=belief_id)
        if belief is None:
            raise ScenarioError(f"no belief has been recomputed for {belief_id!r}")
        return belief

    def _require_rec(self, ref: Any) -> Any:
        rec_id = self.recs.get(str(ref), str(ref))
        rec = self.repo.get_recommendation(rec_id)
        if rec is None:
            raise ScenarioError(f"no recommendation bound to / stored for {ref!r}")
        return rec

    def _checkpoint_confidence(self, name: str) -> float:
        if name not in self.checkpoints:
            raise ScenarioError(f"reference to unknown checkpoint {name!r}")
        return float(self.checkpoints[name]["confidence"])

    # -- run -----------------------------------------------------------

    def run(self) -> ScenarioResult:
        error: str | None = None
        checks: list[CheckResult] = []
        try:
            for index, step in enumerate(self.steps, start=1):
                self._run_step(index, step)
            if not self.expect:
                raise ScenarioError("scenario has no expectations to check")
            checks = [self._run_check(spec) for spec in self.expect]
        except ScenarioError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - any unexpected failure fails the scenario
            error = f"unexpected {type(exc).__name__}: {exc}"
        finally:
            self.repo.close()

        passed = error is None and bool(checks) and all(c.passed for c in checks)
        return ScenarioResult(
            name=self.name,
            description=self.description,
            passed=passed,
            checks=checks,
            error=error,
            source=self.source,
        )

    def _run_step(self, index: int, step: Any) -> None:
        if not isinstance(step, dict):
            raise ScenarioError(f"step {index} must be a JSON object")
        op = step.get("op")
        handler = _STEP_OPS.get(op)
        if handler is None:
            raise ScenarioError(f"step {index}: unknown op {op!r} (known: {sorted(_STEP_OPS)})")
        try:
            handler(self, step)
        except ScenarioError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ScenarioError(f"step {index} ({op}): {type(exc).__name__}: {exc}") from exc

    def _run_check(self, spec: Any) -> CheckResult:
        if not isinstance(spec, dict):
            return CheckResult("?", "", "", "", False, f"check must be a JSON object, got {spec!r}")
        name = spec.get("check")
        handler = _CHECKS.get(name)
        if handler is None:
            return CheckResult(
                str(name), "", "", "", False, f"unknown check {name!r} (known: {sorted(_CHECKS)})"
            )
        try:
            return handler(self, spec)
        except ScenarioError as exc:
            return CheckResult(str(name), _target_hint(spec), "", "", False, str(exc))
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                str(name), _target_hint(spec), "", "", False, f"{type(exc).__name__}: {exc}"
            )

    # -- step ops -----------------------------------------------------

    def op_event(self, step: dict[str, Any]) -> None:
        step_id = str(_require(step, "id", op="event"))
        days = float(step.get("days_before_as_of", 20))
        event = UserEvent(
            event_id=f"{self.user_id}__evt__{step_id}",
            user_id=self.user_id,
            event_type=str(step.get("event_type", "goal_completed")),
            timestamp=self.as_of - timedelta(days=days),
            source=str(step.get("source", "app")),
            **_VERSION_FIELDS,
        )
        self.repo.insert_event(event)
        self.events[step_id] = event

    def op_observation(self, step: dict[str, Any]) -> None:
        step_id = str(_require(step, "id", op="observation"))
        event = self._event(str(_require(step, "event", op="observation")))
        observation, links = create_observation_from_event(
            event,
            observation_id=f"{self.user_id}__obs__{step_id}",
            category=str(step.get("category", "routine")),
            observation_text=str(step.get("text", "eval observation")),
            importance=float(step.get("importance", 0.6)),
            confidence=float(step.get("confidence", 0.6)),
            created_at=event.timestamp,
            **_VERSION_FIELDS,
        )
        self.repo.insert_observation(observation, links)
        self.observations[step_id] = (observation, links)

    def op_evidence(self, step: dict[str, Any]) -> None:
        step_id = str(_require(step, "id", op="evidence"))
        belief_id = self.belief_ref(_require(step, "belief", op="evidence"))
        belief_type = str(step.get("belief_type", "behavioral_tendency"))

        if "observation" in step:
            key = str(step["observation"])
            if key not in self.observations:
                raise ScenarioError(f"evidence step references unknown observation {key!r}")
            observation, links = self.observations[key]
            link_event_ids = {link.event_id for link in links}
            source_events = [e for e in self.events.values() if e.event_id in link_event_ids]
            created_at = min(e.timestamp for e in source_events)
        else:
            days = float(step.get("days_before_as_of", 20))
            event = UserEvent(
                event_id=f"{self.user_id}__evt__{step_id}",
                user_id=self.user_id,
                event_type="goal_completed",
                timestamp=self.as_of - timedelta(days=days),
                source="app",
                **_VERSION_FIELDS,
            )
            self.repo.insert_event(event)
            observation, links = create_observation_from_event(
                event,
                observation_id=f"{self.user_id}__obs__{step_id}",
                category="routine",
                observation_text="eval evidence observation",
                importance=0.6,
                confidence=0.6,
                created_at=event.timestamp,
                **_VERSION_FIELDS,
            )
            self.repo.insert_observation(observation, links)
            source_events = [event]
            created_at = event.timestamp

        proposal = propose_evidence_from_observation_validated(
            observation,
            links,
            source_events,
            belief_id=belief_id,
            direction=str(step.get("direction", "support")),
            source_type=str(step.get("source_type", "recorded_event")),
            context_key=str(step.get("context_key", "fitness")),
            strength=float(step.get("strength", 0.9)),
            model_version="eval-harness-0.6",
            belief_type=belief_type,
            **_VERSION_FIELDS,
        )
        evidence = authorize_evidence(
            proposal,
            evidence_id=f"{self.user_id}__bev__{step_id}",
            created_at=created_at,
            aggregation_policy_version=_AGGREGATION_POLICY_VERSION,
        )
        self.repo.insert_evidence(evidence)
        self.evidence[step_id] = evidence
        self.beliefs.add(belief_id)

    def op_recompute(self, step: dict[str, Any]) -> None:
        belief_id = self.belief_ref(_require(step, "belief", op="recompute"))
        active = self.repo.list_active_evidence(user_id=self.user_id, belief_id=belief_id)
        first_observed = min(
            (row.observed_at for row in active), default=self.as_of - timedelta(days=365)
        )
        extra: dict[str, Any] = {}
        if step.get("no_active_evidence_status"):
            extra["no_active_evidence_status"] = BeliefStatus(step["no_active_evidence_status"])
        belief = recompute_belief(
            belief_id=belief_id,
            user_id=self.user_id,
            belief_type=str(step.get("belief_type", "behavioral_tendency")),
            belief_key=str(step.get("belief_key", "eval_belief_key")),
            belief_value=step.get("belief_value", True),
            evidence=self.repo.list_evidence(user_id=self.user_id, belief_id=belief_id),
            as_of=self.clock,
            first_observed=first_observed,
            recompute_reason=step.get("reason"),
            **_VERSION_FIELDS,
            **extra,
        )
        self.repo.save_belief(belief)
        self.beliefs.add(belief_id)
        if step.get("checkpoint"):
            self.checkpoints[str(step["checkpoint"])] = {
                "belief": belief_id,
                "confidence": belief.confidence,
                "status": belief.status.value,
            }

    def op_checkpoint(self, step: dict[str, Any]) -> None:
        name = str(_require(step, "name", op="checkpoint"))
        belief_id = self.belief_ref(_require(step, "belief", op="checkpoint"))
        belief = self._require_belief(belief_id)
        self.checkpoints[name] = {
            "belief": belief_id,
            "confidence": belief.confidence,
            "status": belief.status.value,
        }

    def op_lock(self, step: dict[str, Any]) -> None:
        belief_id = self.belief_ref(_require(step, "belief", op="lock"))
        self.repo.lock_belief_until_recompute(user_id=self.user_id, belief_id=belief_id)

    def op_invalidate_evidence(self, step: dict[str, Any]) -> None:
        target = _require(step, "evidence", op="invalidate_evidence")
        refs = target if isinstance(target, list) else [target]
        reason = str(step.get("reason", "eval invalidation"))
        for ref in refs:
            key = str(ref)
            if key not in self.evidence:
                raise ScenarioError(f"invalidate_evidence references unknown evidence {key!r}")
            self.repo.mark_evidence_inactive(
                self.evidence[key].evidence_id, reason=reason, invalidated_at=self.clock
            )

    def op_recommendation(self, step: dict[str, Any]) -> None:
        step_id = str(_require(step, "id", op="recommendation"))
        beliefs = self.repo.list_latest_beliefs(user_id=self.user_id)
        recommendation = generate_recommendation(
            recommendation_id=f"{self.user_id}__rec__{step_id}",
            user_id=self.user_id,
            context_key=str(step.get("context_key", "fitness_scheduling")),
            beliefs=beliefs,
            created_at=self.as_of,
            goal=step.get("goal"),
        )
        self.repo.insert_recommendation(recommendation)
        self.recs[step_id] = recommendation.recommendation_id

    def op_outcome(self, step: dict[str, Any]) -> None:
        rec_ref = str(_require(step, "recommendation", op="outcome"))
        if rec_ref not in self.recs:
            raise ScenarioError(f"outcome step references unknown recommendation {rec_ref!r}")
        rec_id = self.recs[rec_ref]
        self._outcome_seq += 1
        offset = float(step.get("day_offset", self._outcome_seq))
        outcome_id = str(step.get("id") or f"{rec_id}__out_{self._outcome_seq}")
        outcome = RecommendationOutcome(
            outcome_id=outcome_id,
            recommendation_id=rec_id,
            followed=str(step.get("followed", "followed")),
            result=str(step.get("result", "successful")),
            source=str(step.get("source", "app_event")),
            created_at=self.as_of + timedelta(days=offset),
            user_feedback=step.get("user_feedback"),
            **_VERSION_FIELDS,
        )
        self.repo.insert_recommendation_outcome(outcome)

    def op_learn(self, step: dict[str, Any]) -> None:
        recommendations = self.repo.list_recommendations(user_id=self.user_id)
        shown = {r.recommendation_id for r in recommendations}
        outcomes = [
            o for o in self.repo.list_recommendation_outcomes() if o.recommendation_id in shown
        ]
        belief_source_events: dict[str, list[str]] = {}
        for belief_id in sorted(self.beliefs):
            events = sorted(
                {
                    source
                    for row in self.repo.list_active_evidence(
                        user_id=self.user_id, belief_id=belief_id
                    )
                    for source in row.source_event_ids
                }
            )
            if events:
                belief_source_events[belief_id] = events
        signals = analyze_recommendation_outcomes(
            recommendations, outcomes, as_of=self.as_of, belief_source_events=belief_source_events
        )
        for signal in signals:
            self.repo.insert_outcome_learning_signal(signal)

        step_id = step.get("id")
        if step_id is not None:
            context = step.get("context")
            chosen = next(
                (s for s in signals if context is None or s.recommendation_context == context),
                None,
            )
            if chosen is None:
                raise ScenarioError(
                    f"learn step id {step_id!r}: no signal was produced"
                    + (f" for context {context!r}" if context else "")
                )
            self.signals[str(step_id)] = chosen.signal_id

    def op_review(self, step: dict[str, Any]) -> None:
        signal_id = self._signal_id(_require(step, "signal", op="review"))
        self.clock = max(self.clock, self.promote_as_of)
        review_outcome_learning_signal(
            self.repo,
            review_id=str(step.get("review_id", "eval_rev_1")),
            signal_id=signal_id,
            reviewer_id=str(step.get("reviewer_id", "eval_reviewer")),
            decision=str(_require(step, "decision", op="review")),
            as_of=self.promote_as_of,
            notes=step.get("notes"),
            promote=bool(step.get("promote", False)),
            recompute=bool(step.get("recompute", False)),
            allow_duplicate=bool(step.get("allow_duplicate", False)),
        )

    def op_promote(self, step: dict[str, Any]) -> None:
        signal_id = self._signal_id(_require(step, "signal", op="promote"))
        self.clock = max(self.clock, self.promote_as_of)
        promote_outcome_learning_signal(
            self.repo,
            signal_id=signal_id,
            as_of=self.promote_as_of,
            persist=bool(step.get("persist", True)),
            recompute=bool(step.get("recompute", False)),
        )

    # -- checks ------------------------------------------------------

    def _compare(self, actual: Any, spec: dict[str, Any]) -> tuple[bool, str]:
        exprs: list[str] = []
        ok = True
        if "equals" in spec:
            exprs.append(f"== {spec['equals']!r}")
            ok = ok and actual == spec["equals"]
        if "in" in spec:
            exprs.append(f"in {spec['in']}")
            ok = ok and actual in spec["in"]
        if "min" in spec:
            exprs.append(f">= {spec['min']}")
            ok = ok and actual >= spec["min"]
        if "max" in spec:
            exprs.append(f"<= {spec['max']}")
            ok = ok and actual <= spec["max"]
        if "gt" in spec:
            exprs.append(f"> {spec['gt']}")
            ok = ok and actual > spec["gt"]
        if "lt" in spec:
            exprs.append(f"< {spec['lt']}")
            ok = ok and actual < spec["lt"]
        if "eq_checkpoint" in spec:
            base = self._checkpoint_confidence(str(spec["eq_checkpoint"]))
            tol = float(spec.get("tol", 1e-9))
            exprs.append(f"== checkpoint {spec['eq_checkpoint']!r} ({base:.6f}, tol {tol})")
            ok = ok and abs(actual - base) <= tol
        if "gt_checkpoint" in spec:
            base = self._checkpoint_confidence(str(spec["gt_checkpoint"]))
            exprs.append(f"> checkpoint {spec['gt_checkpoint']!r} ({base:.6f})")
            ok = ok and actual > base
        if "lt_checkpoint" in spec:
            base = self._checkpoint_confidence(str(spec["lt_checkpoint"]))
            exprs.append(f"< checkpoint {spec['lt_checkpoint']!r} ({base:.6f})")
            ok = ok and actual < base
        if "approx" in spec:
            tol = float(spec.get("tol", 1e-6))
            exprs.append(f"~= {spec['approx']} (tol {tol})")
            ok = ok and abs(actual - spec["approx"]) <= tol
        if not exprs:
            return False, "(no comparison specified)"
        return ok, " and ".join(exprs)

    def _belief_field_value(self, belief: Any, fieldname: str) -> Any:
        if fieldname == "status":
            return belief.status.value
        if not hasattr(belief, fieldname):
            raise ScenarioError(f"belief has no field {fieldname!r}")
        return getattr(belief, fieldname)

    def chk_checkpoint(self, spec: dict[str, Any]) -> CheckResult:
        name = str(_require(spec, "name", op="checkpoint check"))
        if name not in self.checkpoints:
            raise ScenarioError(f"reference to unknown checkpoint {name!r}")
        fieldname = str(spec.get("field", "confidence"))
        snapshot = self.checkpoints[name]
        if fieldname not in snapshot:
            raise ScenarioError(f"checkpoint {name!r} has no field {fieldname!r}")
        actual = snapshot[fieldname]
        ok, expected = self._compare(actual, spec)
        return CheckResult("checkpoint", f"{name}.{fieldname}", expected, _fmt(actual), ok)

    def chk_belief_field(self, spec: dict[str, Any]) -> CheckResult:
        belief_id = self.belief_ref(_require(spec, "belief", op="belief_field check"))
        fieldname = str(_require(spec, "field", op="belief_field check"))
        belief = self._require_belief(belief_id)
        actual = self._belief_field_value(belief, fieldname)
        ok, expected = self._compare(actual, spec)
        return CheckResult("belief_field", f"{belief_id}.{fieldname}", expected, _fmt(actual), ok)

    def chk_belief_confidence(self, spec: dict[str, Any]) -> CheckResult:
        return self.chk_belief_field({**spec, "field": "confidence"})

    def chk_belief_status(self, spec: dict[str, Any]) -> CheckResult:
        return self.chk_belief_field({**spec, "field": "status"})

    def chk_belief_locked(self, spec: dict[str, Any]) -> CheckResult:
        return self.chk_belief_field({**spec, "field": "locked_until_recompute"})

    def chk_belief_evidence_count(self, spec: dict[str, Any]) -> CheckResult:
        belief_id = self.belief_ref(_require(spec, "belief", op="belief_evidence_count check"))
        active_only = bool(spec.get("active", True))
        rows = (
            self.repo.list_active_evidence(user_id=self.user_id, belief_id=belief_id)
            if active_only
            else self.repo.list_evidence(user_id=self.user_id, belief_id=belief_id)
        )
        ok, expected = self._compare(len(rows), spec)
        scope = "active" if active_only else "all"
        return CheckResult(
            "belief_evidence_count", f"{belief_id} ({scope})", expected, str(len(rows)), ok
        )

    def chk_recommendation_uses_only(self, spec: dict[str, Any]) -> CheckResult:
        rec = self._require_rec(_require(spec, "recommendation", op="recommendation_uses_only check"))
        allowed = {self.belief_ref(b) for b in _require(spec, "allowed", op="recommendation_uses_only check")}
        used = set(rec.belief_ids_used)
        ok = used <= allowed
        return CheckResult(
            "recommendation_uses_only",
            rec.recommendation_id,
            f"belief_ids_used subset of {sorted(allowed)}",
            str(sorted(used)),
            ok,
            "" if ok else f"disallowed belief(s) used: {sorted(used - allowed)}",
        )

    def chk_recommendation_excludes(self, spec: dict[str, Any]) -> CheckResult:
        rec = self._require_rec(_require(spec, "recommendation", op="recommendation_excludes check"))
        belief_id = self.belief_ref(_require(spec, "belief", op="recommendation_excludes check"))
        used = set(rec.belief_ids_used)
        blocked = {b.belief_id for b in rec.blocked_beliefs}
        ok = belief_id not in used
        detail = "" if ok else f"{belief_id} is in belief_ids_used"
        if spec.get("require_blocked"):
            if belief_id not in blocked:
                ok = False
                detail = f"{belief_id} is not in blocked_beliefs"
        return CheckResult(
            "recommendation_excludes",
            rec.recommendation_id,
            f"{belief_id} not used" + (" and blocked" if spec.get("require_blocked") else ""),
            f"used={sorted(used)} blocked={sorted(blocked)}",
            ok,
            detail,
        )

    def chk_recommendation_field(self, spec: dict[str, Any]) -> CheckResult:
        rec = self._require_rec(_require(spec, "recommendation", op="recommendation_field check"))
        fieldname = str(_require(spec, "field", op="recommendation_field check"))
        raw = getattr(rec, fieldname, None)
        if fieldname == "risk_tier":
            actual: Any = rec.risk_tier.value
        elif fieldname == "review_status":
            actual = rec.review_status.value
        elif raw is None and not hasattr(rec, fieldname):
            raise ScenarioError(f"recommendation has no field {fieldname!r}")
        else:
            actual = raw
        ok, expected = self._compare(actual, spec)
        return CheckResult("recommendation_field", f"{rec.recommendation_id}.{fieldname}", expected, _fmt(actual), ok)

    def chk_signal_exists(self, spec: dict[str, Any]) -> CheckResult:
        if "signal" in spec:
            exists = self.repo.get_outcome_learning_signal(self._signal_id(spec["signal"])) is not None
            target = str(spec["signal"])
        elif "context" in spec:
            exists = any(
                s.recommendation_context == spec["context"]
                for s in self.repo.list_outcome_learning_signals(user_id=self.user_id)
            )
            target = f"context={spec['context']}"
        else:
            raise ScenarioError("signal_exists check needs a 'signal' or 'context' field")
        want = spec.get("equals", True)
        return CheckResult("signal_exists", target, f"== {want!r}", _fmt(exists), exists == want)

    def _signal_field_value(self, signal: Any, fieldname: str) -> Any:
        if fieldname == "kind":
            return signal.kind.value
        if fieldname == "direction":
            return signal.direction.value if signal.direction is not None else None
        if fieldname == "proposed_evidence_count":
            return len(signal.proposed_evidence)
        if not hasattr(signal, fieldname):
            raise ScenarioError(f"signal has no field {fieldname!r}")
        return getattr(signal, fieldname)

    def chk_signal_field(self, spec: dict[str, Any]) -> CheckResult:
        signal = self._require_signal(_require(spec, "signal", op="signal_field check"))
        fieldname = str(_require(spec, "field", op="signal_field check"))
        actual = self._signal_field_value(signal, fieldname)
        ok, expected = self._compare(actual, spec)
        return CheckResult("signal_field", f"{signal.signal_id}.{fieldname}", expected, _fmt(actual), ok)

    def chk_signal_proposed_strength(self, spec: dict[str, Any]) -> CheckResult:
        signal = self._require_signal(_require(spec, "signal", op="signal_proposed_strength check"))
        strengths = [p.strength for p in signal.proposed_evidence]
        actual = max(strengths) if strengths else 0.0
        ok, expected = self._compare(actual, spec)
        return CheckResult(
            "signal_proposed_strength", signal.signal_id, expected, f"{actual:.6f}", ok
        )

    def chk_signal_proposed_source_types(self, spec: dict[str, Any]) -> CheckResult:
        signal = self._require_signal(_require(spec, "signal", op="signal_proposed_source_types check"))
        want = spec.get("equals", "repeated_pattern_summary")
        want_set = {want} if isinstance(want, str) else set(want)
        actual = {p.source_type.value for p in signal.proposed_evidence}
        ok = actual == want_set
        return CheckResult(
            "signal_proposed_source_types", signal.signal_id, f"== {sorted(want_set)}", str(sorted(actual)), ok
        )

    def chk_review_status(self, spec: dict[str, Any]) -> CheckResult:
        signal_id = self._signal_id(_require(spec, "signal", op="review_status check"))
        reviews = self.repo.list_outcome_learning_signal_reviews(signal_id=signal_id)
        status = reviews[-1].decision.value if reviews else "pending"
        ok, expected = self._compare(status, spec)
        return CheckResult("review_status", signal_id, expected, status, ok)

    def chk_promoted_evidence_count(self, spec: dict[str, Any]) -> CheckResult:
        signal = self._require_signal(_require(spec, "signal", op="promoted_evidence_count check"))
        rows = [
            row
            for row in self.repo.list_all_evidence(user_id=self.user_id)
            if row.source_type is SourceType.REPEATED_PATTERN_SUMMARY
            and row.independence_group == signal.independence_group
            and row.is_active
        ]
        ok, expected = self._compare(len(rows), spec)
        return CheckResult(
            "promoted_evidence_count", signal.signal_id, expected, str(len(rows)), ok
        )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _target_hint(spec: dict[str, Any]) -> str:
    for key in ("belief", "recommendation", "signal", "context"):
        if key in spec:
            return f"{key}={spec[key]}"
    return ""


_STEP_OPS: dict[str, Callable[[_ScenarioRunner, dict[str, Any]], None]] = {
    "event": _ScenarioRunner.op_event,
    "observation": _ScenarioRunner.op_observation,
    "evidence": _ScenarioRunner.op_evidence,
    "recompute": _ScenarioRunner.op_recompute,
    "checkpoint": _ScenarioRunner.op_checkpoint,
    "lock": _ScenarioRunner.op_lock,
    "invalidate_evidence": _ScenarioRunner.op_invalidate_evidence,
    "recommendation": _ScenarioRunner.op_recommendation,
    "outcome": _ScenarioRunner.op_outcome,
    "learn": _ScenarioRunner.op_learn,
    "review": _ScenarioRunner.op_review,
    "promote": _ScenarioRunner.op_promote,
}

_CHECKS: dict[str, Callable[[_ScenarioRunner, dict[str, Any]], CheckResult]] = {
    "checkpoint": _ScenarioRunner.chk_checkpoint,
    "belief_field": _ScenarioRunner.chk_belief_field,
    "belief_confidence": _ScenarioRunner.chk_belief_confidence,
    "belief_status": _ScenarioRunner.chk_belief_status,
    "belief_locked": _ScenarioRunner.chk_belief_locked,
    "belief_evidence_count": _ScenarioRunner.chk_belief_evidence_count,
    "recommendation_uses_only": _ScenarioRunner.chk_recommendation_uses_only,
    "recommendation_excludes": _ScenarioRunner.chk_recommendation_excludes,
    "recommendation_field": _ScenarioRunner.chk_recommendation_field,
    "signal_exists": _ScenarioRunner.chk_signal_exists,
    "signal_field": _ScenarioRunner.chk_signal_field,
    "signal_proposed_strength": _ScenarioRunner.chk_signal_proposed_strength,
    "signal_proposed_source_types": _ScenarioRunner.chk_signal_proposed_source_types,
    "review_status": _ScenarioRunner.chk_review_status,
    "promoted_evidence_count": _ScenarioRunner.chk_promoted_evidence_count,
}


# --------------------------------------------------------------- public --


def run_scenario(manifest: dict[str, Any], *, source: str | None = None) -> ScenarioResult:
    """Execute one scenario manifest and score it. Never raises for a bad
    manifest or a failing check -- the failure is recorded on the returned
    ``ScenarioResult`` (``passed=False``, ``error`` set for a structural
    problem)."""
    try:
        runner = _ScenarioRunner(manifest, source=source)
    except ScenarioError as exc:
        name = str(manifest.get("name") if isinstance(manifest, dict) else None) or (
            Path(source).stem if source else "scenario"
        )
        return ScenarioResult(name=name, description="", passed=False, error=str(exc), source=source)
    return runner.run()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_manifests(paths: list[str]) -> list[Path]:
    """Expand a list of file/dir paths into a sorted list of ``*.json``
    manifest files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.glob("*.json")))
        else:
            found.append(p)
    return found


def run_manifests(paths: list[str]) -> EvalReport:
    report = EvalReport()
    for manifest_path in discover_manifests(paths):
        if not manifest_path.is_file():
            report.scenarios.append(
                ScenarioResult(
                    name=manifest_path.stem,
                    description="",
                    passed=False,
                    error=f"manifest file not found: {manifest_path}",
                    source=str(manifest_path),
                )
            )
            continue
        try:
            manifest = load_manifest(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            report.scenarios.append(
                ScenarioResult(
                    name=manifest_path.stem,
                    description="",
                    passed=False,
                    error=f"could not load manifest: {exc}",
                    source=str(manifest_path),
                )
            )
            continue
        report.scenarios.append(run_scenario(manifest, source=str(manifest_path)))
    return report


def format_report(report: EvalReport, *, verbose: bool = False) -> str:
    """A human-readable scorecard."""
    lines: list[str] = ["Adaptive user model evaluation", "=" * 30, ""]
    for scenario in report.scenarios:
        tag = "PASS" if scenario.passed else "FAIL"
        header = f"{tag}  {scenario.name}"
        if scenario.total_checks:
            header += f"  ({scenario.passed_checks}/{scenario.total_checks} checks)"
        lines.append(header)
        if scenario.description:
            lines.append(f"      {scenario.description}")
        if scenario.error:
            lines.append(f"      ERROR: {scenario.error}")
        for check in scenario.checks:
            if check.passed and not verbose:
                continue
            mark = "PASS" if check.passed else "FAIL"
            target = f"[{check.target}]" if check.target else ""
            detail = f"  {check.message}" if check.message else ""
            lines.append(
                f"    {mark} {check.check}{target}  expected {check.expected}; got {check.actual}{detail}"
            )
        lines.append("")
    summary = report.to_dict()["summary"]
    lines.append("-" * 48)
    lines.append(
        f"{summary['passed']}/{summary['scenarios']} scenarios passed"
        + ("" if report.passed else f"  ({summary['failed']} FAILED)")
    )
    return "\n".join(lines)

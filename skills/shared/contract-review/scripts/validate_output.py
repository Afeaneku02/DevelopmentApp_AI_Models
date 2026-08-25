#!/usr/bin/env python3
"""Deterministic part of contract-review: closed-enum membership and
required-version-field checks against a JSON record, for record types that
do not depend on LLM judgment to verify (Blueprint section 10.4: "Use scripts for
validations that should not depend on LLM judgment: enum membership,
required version fields, schema validation, duplicate evidence checks").

This does NOT check the model-proposes/backend-authorizes boundary or test
coverage -- those require reading the diff/code, which is contract-review's
own job, not a script's.

Usage:
    python validate_output.py <record_type> <record.json>

<record_type> is one of: belief, evidence, evidence_record, event, observation,
recommendation. <record.json> may contain a single object or a JSON array of
objects.

``evidence`` vs ``evidence_record`` (these check different, non-overlapping
things -- picking the wrong one either misses real violations or false-flags
legitimate output):

- ``evidence``: validates a *pre-authorization proposal* -- the shape a
  skill/extractor/LLM is allowed to produce. It FAILS if the record carries
  any backend-owned field (``authorized_aggregation_mode``,
  ``aggregation_authorized_by``, ``aggregation_authorized_at``, etc.) at all,
  because a proposal must never carry them.
- ``evidence_record``: validates an *already-authorized, persisted*
  belief_evidence row (e.g. the output of ``authorize_evidence()`` in
  ``src/beliefs/models.py``). Such a row is SUPPOSED to carry those fields;
  this instead checks that the authorization metadata is internally
  consistent (a recognized ``authorized_aggregation_mode``, an
  ``aggregation_authorized_by`` on every record, and ``replaces_evidence_ids``
  /timestamps present whenever the mode is ``aggregate_replacement``).

Running ``evidence`` against a backend-authorized record will always fail
(that is the record doing exactly what only backend code may do); that is a
misuse of the check, not a contract violation in the record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BELIEF_TYPES = {
    "behavioral_tendency", "routine_or_preference", "communication_or_learning_preference",
    "current_state_related", "goal_or_intention", "constraint_or_aversion",
    "cross_context_tendency", "recommendation_response_pattern", "sensitive_or_high_impact_inference",
}
SOURCE_TYPES = {
    "explicit_user_correction", "recorded_event", "explicit_user_statement",
    "repeated_pattern_summary", "model_observation", "llm_inference", "unverified_hypothesis",
}
BELIEF_STATUSES = {"candidate", "provisional", "validated", "contested", "outdated", "rejected"}
DIRECTIONS = {"support", "contradict"}
RISK_TIERS = {"low", "medium", "high"}
AGGREGATION_MODES = {"leaf_default", "aggregate_replacement"}

BASE_VERSION_FIELDS = ("schema_version", "scoring_version", "canonicalizer_version", "policy_version")
RECOMMENDATION_VERSION_FIELDS = BASE_VERSION_FIELDS + ("risk_policy_version", "risk_domain_policy_version")

MODEL_AUTHORITY_VIOLATION_FIELDS = {
    # A record produced directly by model/extractor output must never carry
    # these -- they are backend-authorized-only fields (Blueprint section 6.1.2, section 6.4).
    "authorized_aggregation_mode", "aggregation_authorized_by", "aggregation_authorized_at",
}


def _check_version_fields(record: dict, required: tuple[str, ...]) -> list[str]:
    return [f"missing required version field: {field}" for field in required if not record.get(field)]


def _is_real_number(value: object) -> bool:
    """True only for an actual int/float -- never a bool (bool is a subclass
    of int in Python, so a naive ``isinstance(value, (int, float))`` check
    silently accepts True/False as 1/0) and never a numeric string (a JSON
    string like "0.68" must never be treated as satisfying a numeric
    contract field, even though it looks like a number)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_unit_interval_field(record: dict, field: str) -> list[str]:
    """Field must be a real number in [0, 1] (strength, source_reliability,
    observation importance/confidence all share this contract)."""
    value = record.get(field)
    if not _is_real_number(value) or not (0.0 <= value <= 1.0):
        return [f"{field} must be a real number in [0, 1], got {value!r}."]
    return []


def validate_belief(record: dict) -> list[str]:
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    if record.get("belief_type") not in BELIEF_TYPES:
        errors.append(f"belief_type {record.get('belief_type')!r} is not one of the nine canonical values.")
    if record.get("status") not in BELIEF_STATUSES:
        errors.append(f"status {record.get('status')!r} is not one of the six canonical lifecycle values.")
    confidence = record.get("confidence")
    if not _is_real_number(confidence):
        errors.append(f"confidence must be a real number, got {confidence!r}.")
    elif not (confidence == 0.0 or 0.02 <= confidence <= 0.98):
        errors.append(f"confidence {confidence!r} is outside the [0.02, 0.98] band (0.0 only allowed for no-evidence).")
    return errors


def _validate_evidence_shape(record: dict) -> list[str]:
    """Checks shared by both ``evidence`` (proposal) and ``evidence_record``
    (authorized) shapes: canonical enums, provenance, and the strength/
    source_reliability numeric contract (blueprint section 5.3 declares both
    as real numbers in [0, 1])."""
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    if record.get("source_type") not in SOURCE_TYPES:
        errors.append(f"source_type {record.get('source_type')!r} is not one of the seven canonical values.")
    if record.get("direction") not in DIRECTIONS:
        errors.append(f"direction {record.get('direction')!r} must be 'support' or 'contradict'.")
    source_event_ids = record.get("source_event_ids")
    if not source_event_ids:
        errors.append("source_event_ids is required and must be non-empty.")
    event_id = record.get("event_id")
    if event_id and source_event_ids and event_id not in source_event_ids:
        errors.append(
            f"event_id {event_id!r} is not included in source_event_ids {source_event_ids!r}; "
            "source_event_ids is the sole authoritative provenance field and evidence must "
            "never rely on event_id alone."
        )
    errors.extend(_validate_unit_interval_field(record, "strength"))
    errors.extend(_validate_unit_interval_field(record, "source_reliability"))
    return errors


def validate_evidence(record: dict) -> list[str]:
    """Validates a PRE-AUTHORIZATION PROPOSAL shape. See the module docstring
    for why this is a distinct check from ``validate_evidence_record``."""
    errors = _validate_evidence_shape(record)
    present_authority_violations = MODEL_AUTHORITY_VIOLATION_FIELDS.intersection(record.keys())
    if present_authority_violations:
        errors.append(
            f"record sets backend-only field(s) directly, a model-proposes/backend-authorizes boundary "
            f"violation: {sorted(present_authority_violations)}"
        )
    return errors


def validate_evidence_record(record: dict) -> list[str]:
    """Validates an ALREADY-AUTHORIZED, PERSISTED belief_evidence row -- the
    output of backend authorization (e.g. ``authorize_evidence()`` in
    ``src/beliefs/models.py``), not raw model/extractor output. See the
    module docstring for why this is a distinct check from
    ``validate_evidence``."""
    errors = _validate_evidence_shape(record)

    mode = record.get("authorized_aggregation_mode")
    if mode not in AGGREGATION_MODES:
        errors.append(f"authorized_aggregation_mode {mode!r} must be one of {sorted(AGGREGATION_MODES)}.")
    if not record.get("aggregation_authorized_by"):
        errors.append(
            "aggregation_authorized_by is missing; every authorized record, including "
            "leaf_default, must record who/what authorized it."
        )
    if mode == "aggregate_replacement":
        if not record.get("replaces_evidence_ids"):
            errors.append("authorized_aggregation_mode is aggregate_replacement but replaces_evidence_ids is empty.")
        if not record.get("aggregation_authorized_at"):
            errors.append("authorized_aggregation_mode is aggregate_replacement but aggregation_authorized_at is missing.")
    return errors


def validate_event(record: dict) -> list[str]:
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    for field in ("event_id", "user_id", "event_type", "timestamp"):
        if not record.get(field):
            errors.append(f"missing required field: {field}")
    return errors


def validate_observation(record: dict) -> list[str]:
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    for field in ("importance", "confidence"):
        errors.extend(_validate_unit_interval_field(record, field))
    return errors


def validate_recommendation(record: dict) -> list[str]:
    errors = _check_version_fields(record, RECOMMENDATION_VERSION_FIELDS)
    if record.get("risk_tier") not in RISK_TIERS:
        errors.append(f"risk_tier {record.get('risk_tier')!r} must be one of {sorted(RISK_TIERS)}.")
    if record.get("review_required") and record.get("exploration_applied"):
        errors.append("exploration_applied must be false whenever review_required is true.")
    if not record.get("profile_snapshot_id") and not record.get("frozen_belief_state"):
        errors.append("recommendation must carry profile_snapshot_id or frozen_belief_state for auditability.")
    return errors


VALIDATORS = {
    "belief": validate_belief,
    "evidence": validate_evidence,
    "evidence_record": validate_evidence_record,
    "event": validate_event,
    "observation": validate_observation,
    "recommendation": validate_recommendation,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in VALIDATORS:
        print(f"Usage: validate_output.py <{'|'.join(VALIDATORS)}> <record.json>", file=sys.stderr)
        return 2
    record_type, path = argv
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]

    # "evidence_record" validates the same belief_evidence shape as
    # "evidence" (evidence_id), not a record with its own "evidence_record_id".
    id_field = "evidence_id" if record_type == "evidence_record" else record_type + "_id"

    validator = VALIDATORS[record_type]
    all_errors: list[str] = []
    for i, record in enumerate(records):
        for error in validator(record):
            all_errors.append(f"record[{i}] ({record.get(id_field, '?')}): {error}")

    if all_errors:
        for error in all_errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(records)} {record_type} record(s) passed contract validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

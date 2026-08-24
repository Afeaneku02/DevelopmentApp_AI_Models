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

<record_type> is one of: belief, evidence, event, observation, recommendation
<record.json> may contain a single object or a JSON array of objects.
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

BASE_VERSION_FIELDS = ("schema_version", "scoring_version", "canonicalizer_version", "policy_version")
RECOMMENDATION_VERSION_FIELDS = BASE_VERSION_FIELDS + ("risk_policy_version", "risk_domain_policy_version")

MODEL_AUTHORITY_VIOLATION_FIELDS = {
    # A record produced directly by model/extractor output must never carry
    # these -- they are backend-authorized-only fields (Blueprint section 6.1.2, section 6.4).
    "authorized_aggregation_mode", "aggregation_authorized_by", "aggregation_authorized_at",
}


def _check_version_fields(record: dict, required: tuple[str, ...]) -> list[str]:
    return [f"missing required version field: {field}" for field in required if not record.get(field)]


def validate_belief(record: dict) -> list[str]:
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    if record.get("belief_type") not in BELIEF_TYPES:
        errors.append(f"belief_type {record.get('belief_type')!r} is not one of the nine canonical values.")
    if record.get("status") not in BELIEF_STATUSES:
        errors.append(f"status {record.get('status')!r} is not one of the six canonical lifecycle values.")
    confidence = record.get("confidence")
    if isinstance(confidence, (int, float)) and not (0.02 <= confidence <= 0.98 or confidence == 0.0):
        errors.append(f"confidence {confidence!r} is outside the [0.02, 0.98] band (0.0 only allowed for no-evidence).")
    return errors


def validate_evidence(record: dict) -> list[str]:
    errors = _check_version_fields(record, BASE_VERSION_FIELDS)
    if record.get("source_type") not in SOURCE_TYPES:
        errors.append(f"source_type {record.get('source_type')!r} is not one of the seven canonical values.")
    if record.get("direction") not in DIRECTIONS:
        errors.append(f"direction {record.get('direction')!r} must be 'support' or 'contradict'.")
    if not record.get("source_event_ids"):
        errors.append("source_event_ids is required and must be non-empty.")
    present_authority_violations = MODEL_AUTHORITY_VIOLATION_FIELDS.intersection(record.keys())
    if present_authority_violations:
        errors.append(
            f"record sets backend-only field(s) directly, a model-proposes/backend-authorizes boundary "
            f"violation: {sorted(present_authority_violations)}"
        )
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
        value = record.get(field)
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            errors.append(f"{field} must be a number in [0, 1], got {value!r}")
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

    validator = VALIDATORS[record_type]
    all_errors: list[str] = []
    for i, record in enumerate(records):
        for error in validator(record):
            all_errors.append(f"record[{i}] ({record.get(record_type + '_id', '?')}): {error}")

    if all_errors:
        for error in all_errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(records)} {record_type} record(s) passed contract validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Deterministic completion check for create-belief-evidence.

Validates that proposed belief_evidence rows use the canonical seven-value
source_type enum with the matching default reliability, never self-authorize
aggregation, and never carry a confidence field (confidence is computed
downstream by update-user-beliefs, never proposed here).

Usage:
    python validate_evidence.py <evidence_batch.json>

<evidence_batch.json> shape: a JSON array of belief_evidence proposal objects.

Exit code 0 means the batch passes; non-zero means it does not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Canonical seven-value source_type enum -> default reliability (Blueprint section 6.2.1).
# Kept here only as a deterministic guard value, not as a restatement of the
# full policy -- see references/policy.md for the authoritative table.
DEFAULT_RELIABILITY = {
    "explicit_user_correction": 1.00,
    "recorded_event": 0.95,
    "explicit_user_statement": 0.85,
    "repeated_pattern_summary": 0.80,
    "model_observation": 0.70,
    "llm_inference": 0.55,
    "unverified_hypothesis": 0.35,
}

FORBIDDEN_FIELDS = {
    "authorized_aggregation_mode",
    "aggregation_authorized_by",
    "aggregation_authorized_at",
    "is_duplicate_suppressed",
    "suppression_reason",
    "is_active",
    "invalidated_at",
    "invalidation_reason",
    "confidence",
    "confidence_hint",
}

RELIABILITY_TOLERANCE = 0.05


def validate_row(row: dict) -> list[str]:
    errors = []
    evidence_id = row.get("evidence_id", "<unknown>")

    source_type = row.get("source_type")
    if source_type not in DEFAULT_RELIABILITY:
        errors.append(f"{evidence_id}: source_type {source_type!r} is not one of the seven canonical values.")
        source_type = None

    if row.get("direction") not in ("support", "contradict"):
        errors.append(f"{evidence_id}: direction must be 'support' or 'contradict', got {row.get('direction')!r}.")

    if not row.get("source_event_ids"):
        errors.append(f"{evidence_id}: source_event_ids is required and must be non-empty.")

    if source_type is not None:
        expected = DEFAULT_RELIABILITY[source_type]
        actual = row.get("source_reliability")
        if not isinstance(actual, (int, float)) or abs(actual - expected) > RELIABILITY_TOLERANCE:
            errors.append(
                f"{evidence_id}: source_reliability {actual!r} does not match the canonical default "
                f"{expected!r} for source_type {source_type!r} (tolerance {RELIABILITY_TOLERANCE})."
            )

    proposed_mode = row.get("proposed_aggregation_mode", "leaf_default")
    if proposed_mode == "aggregate_replacement" and not row.get("replaces_evidence_ids"):
        errors.append(f"{evidence_id}: aggregate_replacement proposed without explicit replaces_evidence_ids.")

    present_forbidden = FORBIDDEN_FIELDS.intersection(row.keys())
    if present_forbidden:
        errors.append(
            f"{evidence_id}: sets backend-owned or out-of-contract field(s) it must never write: "
            f"{sorted(present_forbidden)}."
        )

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: validate_evidence.py <evidence_batch.json>", file=sys.stderr)
        return 2
    rows = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    errors: list[str] = []
    for row in rows:
        errors.extend(validate_row(row))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(rows)} evidence proposal(s) passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

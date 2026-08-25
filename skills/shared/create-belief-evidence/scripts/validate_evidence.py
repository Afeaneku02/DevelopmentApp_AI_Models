#!/usr/bin/env python3
"""Deterministic completion check for create-belief-evidence.

Validates that proposed belief_evidence rows use the canonical seven-value
source_type enum with the matching default reliability (or an explicitly
justified deviation from it -- see ``reliability_deviation_reason`` below),
that ``strength`` is a real number in ``[0, 1]`` (the blueprint schema's
declared range for it), never self-authorize aggregation, and never carry a
confidence field (confidence is computed downstream by update-user-beliefs,
never proposed here).

This check must stay aligned with the deviation-justification contract also
enforced by ``BeliefEvidenceProposal`` in ``src/beliefs/models.py``: a
``source_reliability`` more than ``RELIABILITY_TOLERANCE`` away from the
canonical default is accepted only when ``reliability_deviation_reason`` is
present and non-blank. This is a portable, dependency-free script and does
not import that Pydantic model, so the rule is deliberately re-implemented
here rather than shared -- if one changes, check the other.

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

    strength = row.get("strength")
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        errors.append(f"{evidence_id}: strength must be numeric, got {strength!r}.")
    elif not (0.0 <= strength <= 1.0):
        errors.append(f"{evidence_id}: strength {strength!r} is outside the [0, 1] range.")

    if source_type is not None:
        expected = DEFAULT_RELIABILITY[source_type]
        actual = row.get("source_reliability")
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            errors.append(f"{evidence_id}: source_reliability must be numeric, got {actual!r}.")
        elif not (0.0 <= actual <= 1.0):
            errors.append(f"{evidence_id}: source_reliability {actual!r} is outside the [0, 1] range.")
        elif abs(actual - expected) > RELIABILITY_TOLERANCE + 1e-9:
            # The 1e-9 epsilon absorbs float-precision noise at the tolerance
            # boundary (e.g. 1.0 - 0.95 == 0.050000000000000044 in binary
            # floating point, not exactly 0.05); matches src/beliefs/models.py.
            reason = row.get("reliability_deviation_reason")
            if not (isinstance(reason, str) and reason.strip()):
                errors.append(
                    f"{evidence_id}: source_reliability {actual!r} deviates from the canonical default "
                    f"{expected!r} for source_type {source_type!r} by more than {RELIABILITY_TOLERANCE} "
                    "tolerance; provide a non-blank reliability_deviation_reason to justify the override, "
                    "or use the default."
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

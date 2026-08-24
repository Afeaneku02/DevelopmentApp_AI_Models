#!/usr/bin/env python3
"""Reference implementation of the Better You belief-confidence algorithm
(Blueprint section 3.4.1 "Concrete Belief-Update Algorithm").

This is the executable specification for update-user-beliefs: read this
before re-deriving the formula from memory. It is deliberately dependency-free
so it can be imported directly by the real scoring module once one exists, or
run standalone as a self-test.

Branch order matches section 3.4.1 / section 6.0.2 exactly:
    1. invalidation / no-active-evidence branch (checked first)
    2. generic no-evidence branch (support_mass + contradiction_mass == 0)
    3. normal D / BaseSignal / confidence computation

Run this file directly to execute the self-test, including a reproduction of
the blueprint's own worked small-sample example (two recent, reliable
supporting items -> effective_support_count ~= 1.8 -> support_factor ~= 0.36).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INVALIDATION_REASONS = {"deletion", "reset", "duplicate_suppression", "policy_invalidation"}


@dataclass
class EvidenceItem:
    direction: str  # "support" | "contradict"
    strength: float  # [0, 1]
    source_reliability: float  # [0, 1]
    decay_lambda: float
    age_days: float
    source_type: str

    def decay(self) -> float:
        return math.exp(-self.decay_lambda * self.age_days)

    def quality(self) -> float:
        return self.strength * self.source_reliability

    def mass(self) -> float:
        return self.quality() * self.decay()

    def count_weight(self) -> float:
        return self.source_reliability * self.decay()


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def compute_confidence(
    evidence: list[EvidenceItem],
    weights: dict[str, float],
    expected_source_types: int,
    recompute_reason: str | None = None,
    active_evidence_count: int | None = None,
    no_active_evidence_status: str = "outdated",
) -> dict:
    """Compute confidence/status inputs for one belief from its active,
    deduplicated evidence ledger. `evidence` must already be deduplicated
    (independence-grouped, duplicate-suppressed) -- this function does not
    perform deduplication itself.
    """
    # Branch 1: invalidation / no-active-evidence (must run before branch 2).
    if recompute_reason in INVALIDATION_REASONS and active_evidence_count == 0:
        return {
            "confidence": 0.0,
            "D": None,
            "status": no_active_evidence_status,
            "reason": "no_active_evidence_after_deletion_or_suppression",
        }

    support = [e for e in evidence if e.direction == "support"]
    contradict = [e for e in evidence if e.direction == "contradict"]

    support_mass = sum(e.mass() for e in support)
    contradiction_mass = sum(e.mass() for e in contradict)

    # Branch 2: generic no-evidence.
    if support_mass + contradiction_mass == 0:
        return {"confidence": 0.0, "D": None, "status": "candidate", "reason": "no_evidence"}

    # Branch 3: normal computation.
    d = support_mass / (support_mass + contradiction_mass)

    all_decays = [e.decay() for e in evidence]
    all_qualities = [e.quality() for e in evidence]
    all_reliabilities = [e.source_reliability for e in evidence]

    q = _weighted_mean(all_qualities, all_decays)
    r = _weighted_mean(all_decays, all_reliabilities)

    unique_independent_source_types = len({e.source_type for e in evidence})
    v = min(1.0, unique_independent_source_types / expected_source_types) if expected_source_types else 0.0

    base_signal = weights["wD"] * d + weights["wQ"] * q + weights["wR"] * r + weights["wV"] * v

    effective_support_count = sum(e.count_weight() for e in support)
    effective_evidence_count = sum(e.count_weight() for e in evidence)

    support_factor = 1 - math.exp(-effective_support_count / 4.0)
    confidence = max(0.02, min(0.98, base_signal * support_factor))

    return {
        "confidence": confidence,
        "D": d,
        "Q": q,
        "R": r,
        "V": v,
        "base_signal": base_signal,
        "support_factor": support_factor,
        "effective_support_count": effective_support_count,
        "effective_evidence_count": effective_evidence_count,
        "supporting_evidence_count": len(support),
        "contradicting_evidence_count": len(contradict),
        "total_evidence_count": len(evidence),
        "status": None,  # lifecycle status: see determine_status()
        "reason": None,
    }


def determine_status(confidence: float, d: float | None, effective_support_count: float, thresholds: dict) -> str:
    """Determine candidate/provisional/validated per section 3.4.2's numeric
    thresholds. `contested`, `outdated`, and `rejected` require lifecycle
    context (prior status, retention windows, explicit correction) that is
    not derivable from a single scoring pass -- those transitions belong to
    the caller (update-user-beliefs' lifecycle logic), not to this function.
    """
    if d is None:
        return "candidate"
    if (
        confidence >= thresholds["validated_min_confidence"]
        and effective_support_count >= thresholds["validated_min_effective_support"]
        and d >= thresholds["validated_min_d"]
    ):
        return "validated"
    if confidence >= thresholds["provisional_min_confidence"] and effective_support_count >= 1.5:
        return "provisional"
    return "candidate"


DEFAULT_WEIGHTS = {"wD": 0.45, "wQ": 0.25, "wR": 0.15, "wV": 0.15}
DEFAULT_THRESHOLDS = {
    "provisional_min_confidence": 0.35,
    "validated_min_confidence": 0.65,
    "validated_min_effective_support": 3.5,
    "validated_min_d": 0.75,
}


def _self_test() -> None:
    # Blueprint section 3.4.1 worked example: two recent, reliable supporting items
    # should yield effective_support_count ~= 1.8 and support_factor ~= 0.36.
    two_recent_reliable = [
        EvidenceItem("support", strength=1.0, source_reliability=0.9, decay_lambda=0.015, age_days=0, source_type="recorded_event"),
        EvidenceItem("support", strength=1.0, source_reliability=0.9, decay_lambda=0.015, age_days=0, source_type="explicit_user_statement"),
    ]
    result = compute_confidence(two_recent_reliable, DEFAULT_WEIGHTS, expected_source_types=2)
    assert abs(result["effective_support_count"] - 1.8) < 0.01, result
    assert abs(result["support_factor"] - 0.36) < 0.01, result
    assert 0.02 <= result["confidence"] <= 0.98

    # No evidence at all -> D is null, confidence is 0, status is candidate.
    empty_result = compute_confidence([], DEFAULT_WEIGHTS, expected_source_types=2)
    assert empty_result == {"confidence": 0.0, "D": None, "status": "candidate", "reason": "no_evidence"}

    # Invalidation branch must win even when it would otherwise look like a
    # normal no-evidence case, and must be checked before the generic branch.
    invalidated_result = compute_confidence(
        [], DEFAULT_WEIGHTS, expected_source_types=2,
        recompute_reason="deletion", active_evidence_count=0, no_active_evidence_status="rejected",
    )
    assert invalidated_result["reason"] == "no_active_evidence_after_deletion_or_suppression"
    assert invalidated_result["status"] == "rejected"
    assert invalidated_result["D"] is None

    # Confidence must never reach the 0.98 ceiling even with overwhelming,
    # maximally reliable, zero-decay support.
    overwhelming = [
        EvidenceItem("support", strength=1.0, source_reliability=1.0, decay_lambda=0.0, age_days=0, source_type=f"type_{i}")
        for i in range(50)
    ]
    ceiling_result = compute_confidence(overwhelming, DEFAULT_WEIGHTS, expected_source_types=2)
    assert ceiling_result["confidence"] <= 0.98

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()

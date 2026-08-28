"""Versioned registry defaults: belief_type_registry, source_type reliability,
and the canonical belief_key alias registry.

Blueprint section 3.4.4 "Canonical belief_type Registry", section 6.2.1
"Default Source Reliability Values", and section 5.2 "Belief-Key
Canonicalization & Deduplication". These are frozen, versioned lookup
tables, not scattered constants (section 3.4.3) -- code that needs a decay
lambda, expected source-type count, sensitivity default, persistence
default, source reliability, or a known belief_key alias must resolve it
from here by the recorded registry/scoring version, never hard-code it
inline.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.common.enums import BeliefType, PersistencePolicy, SensitivityClass, SourceType

BELIEF_TYPE_REGISTRY_VERSION = "belief-types-0.6"


@dataclass(frozen=True)
class BeliefTypeDefaults:
    default_decay_lambda: float
    expected_source_types: int
    default_sensitivity_class: SensitivityClass
    default_persistence_policy: PersistencePolicy


BELIEF_TYPE_REGISTRY: dict[BeliefType, BeliefTypeDefaults] = {
    BeliefType.BEHAVIORAL_TENDENCY: BeliefTypeDefaults(
        0.015, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.ROUTINE_OR_PREFERENCE: BeliefTypeDefaults(
        0.015, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.COMMUNICATION_OR_LEARNING_PREFERENCE: BeliefTypeDefaults(
        0.005, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CURRENT_STATE_RELATED: BeliefTypeDefaults(
        0.080, 1, SensitivityClass.NORMAL, PersistencePolicy.SHORT_TERM
    ),
    BeliefType.GOAL_OR_INTENTION: BeliefTypeDefaults(
        0.020, 1, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CONSTRAINT_OR_AVERSION: BeliefTypeDefaults(
        0.020, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.CROSS_CONTEXT_TENDENCY: BeliefTypeDefaults(
        0.010, 3, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.RECOMMENDATION_RESPONSE_PATTERN: BeliefTypeDefaults(
        0.020, 2, SensitivityClass.NORMAL, PersistencePolicy.RETAINED
    ),
    BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE: BeliefTypeDefaults(
        0.030, 4, SensitivityClass.RESTRICTED, PersistencePolicy.DO_NOT_PERSIST
    ),
}

# Matches the scoring_config example's scoring_version in blueprint section 5.3;
# source reliability defaults are versioned as part of scoring_config (section 3.4.3).
SOURCE_RELIABILITY_SCORING_VERSION = "belief-score-0.6"

SOURCE_TYPE_RELIABILITY: dict[SourceType, float] = {
    SourceType.EXPLICIT_USER_CORRECTION: 1.00,
    SourceType.RECORDED_EVENT: 0.95,
    SourceType.EXPLICIT_USER_STATEMENT: 0.85,
    SourceType.REPEATED_PATTERN_SUMMARY: 0.80,
    SourceType.MODEL_OBSERVATION: 0.70,
    SourceType.LLM_INFERENCE: 0.55,
    SourceType.UNVERIFIED_HYPOTHESIS: 0.35,
}

# create-belief-evidence's completion check (skills/shared/create-belief-evidence/SKILL.md):
# "source_reliability matching (or explicitly justifying a deviation from) the
# versioned default for that type." This tolerance matches the one already
# enforced independently by that skill's scripts/validate_evidence.py, so a
# proposal that passes src/beliefs/models.py's validation also passes that
# skill's deterministic check, and vice versa.
SOURCE_RELIABILITY_TOLERANCE = 0.05

CANONICAL_BELIEF_KEY_REGISTRY_VERSION = "belief-key-canon-0.6"

# Known, pre-vetted (belief_type, proposed_key) -> canonical_key aliases
# (blueprint section 5.2). This is deliberately a small, hand-curated
# exact-match table, not a similarity/embedding index: the blueprint allows
# optional lightweight embeddings for this narrow deduplication task, but
# does not require them, and a deterministic exact-match registry is the
# conservative default this phase implements -- see
# src/beliefs/canonicalization.py's own module docstring for why an
# unmatched key is never guessed at. Only genuine aliases belong here (never
# a key mapped to itself); adding an entry is itself the "backend policy"
# decision that a pair of belief_key spellings mean the same thing.
CANONICAL_BELIEF_KEY_ALIASES: dict[tuple[BeliefType, str], str] = {
    (BeliefType.BEHAVIORAL_TENDENCY, "more_consistent_after_work_workouts"): "higher_adherence_after_work",
    (BeliefType.BEHAVIORAL_TENDENCY, "prefers_evening_exercise_sessions"): "higher_adherence_after_work",
}

# Schema Checklist - update-user-beliefs

Canonical source: Blueprint section 5 "Data Model" (`user_beliefs`, `scoring_config`) and section 3.4.3.

## `user_beliefs` fields this skill computes/writes

`confidence, supporting_evidence_count, contradicting_evidence_count, total_evidence_count, effective_support_count, effective_evidence_count, evidence_for, evidence_against, status, reasoning_summary, locked_until_recompute, last_recompute_attempt_id, last_successful_recompute_at`

All of `supporting_evidence_count` / `contradicting_evidence_count` / `total_evidence_count` / `effective_support_count` / `effective_evidence_count` / `evidence_for` / `evidence_against` are **derived/cache fields only** - `belief_evidence` is the source of truth, and these must always be reproducible from it under the recorded `scoring_version`.

## `scoring_config` fields this skill must load, never hard-code

`scoring_version, belief_type_registry_version, weights (wD/wQ/wR/wV), decay_lambdas_by_belief_type, source_reliability_by_source_type, expected_source_types_by_belief_type, aggregation_mode_default, lifecycle_thresholds, risk_policy_version_reference, risk_domain_policy_version_reference`

## Lifecycle status values (section 3.4.2)

`candidate`, `provisional`, `validated`, `contested`, `outdated`, `rejected` - exactly these six, transitioning per the documented thresholds, never invented ad hoc.

## Fields this skill must never set directly

Anything on `belief_evidence` (that is `create-belief-evidence`'s job) and anything on `recommendations` (that is `apply-risk-policy`'s job, once that skill is built - see `skills/shared/apply-risk-policy/SKILL.md`).

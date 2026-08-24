# Schema Checklist - create-belief-evidence

Canonical source: Blueprint section 5 "Data Model" (`belief_evidence`) and section 5.3 "Concrete JSON Contracts".

## `belief_evidence` fields this skill is responsible for proposing

`evidence_id, belief_id, user_id, direction, event_id, observation_id, source_event_ids, source_type, context_key, strength, source_reliability, observed_at, decay_lambda, created_at, model_version, prompt_version, schema_version, scoring_version, canonicalizer_version, policy_version`

- `direction`: `support` | `contradict`.
- `source_event_ids`: array; the sole authoritative provenance field. One ID for leaf evidence, multiple only for a genuinely authorized aggregate.
- `strength`: `[0, 1]`, how strongly this item supports/contradicts, independent of source trust.
- `source_reliability`: `[0, 1]`, from the canonical table in `policy.md` - do not invent ad hoc values without justification.

## Fields this skill must NOT set

`authorized_aggregation_mode`, `aggregation_authorized_by`, `aggregation_authorized_at`, `is_duplicate_suppressed`, `suppression_reason`, `is_active`, `invalidated_at`, `invalidation_reason` - all backend-owned. This skill may only ever write `proposed_aggregation_mode`.

## Also never write directly

`user_beliefs.confidence`, `user_beliefs.status`, `user_beliefs.evidence_for` / `evidence_against` - those are derived/cache fields computed by `update-user-beliefs` from the evidence ledger, never set directly by this skill.

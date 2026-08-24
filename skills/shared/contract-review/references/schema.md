# Closed Contracts Checklist - contract-review

Canonical source: Blueprint section 3.4.4, section 6.2.1, section 3.4.2, section 6.4, section 18. This is a recall aid; re-read the actual section when a violation is suspected rather than trusting this summary alone.

## Closed enums (never silently extended)

| Enum | Values |
|---|---|
| `belief_type` (registry-versioned) | `behavioral_tendency`, `routine_or_preference`, `communication_or_learning_preference`, `current_state_related`, `goal_or_intention`, `constraint_or_aversion`, `cross_context_tendency`, `recommendation_response_pattern`, `sensitive_or_high_impact_inference` |
| `source_type` | `explicit_user_correction`, `recorded_event`, `explicit_user_statement`, `repeated_pattern_summary`, `model_observation`, `llm_inference`, `unverified_hypothesis` |
| belief `status` | `candidate`, `provisional`, `validated`, `contested`, `outdated`, `rejected` |
| `direction` | `support`, `contradict` |
| `risk_tier` | `low`, `medium`, `high` |
| `authorized_aggregation_mode` | `leaf_default`, `aggregate_replacement` |

## Required version fields per record type

- Any `user_events`/`user_observations`/`belief_evidence`/`user_beliefs` row: `schema_version, scoring_version, canonicalizer_version, policy_version`.
- `recommendations`: the above plus `risk_policy_version, risk_domain_policy_version`.
- `belief_type_registry`-dependent code: `belief_type_registry_version`.

## Model-proposes / backend-authorizes boundary (never crossable by model output)

`authorized_aggregation_mode`, `aggregation_authorized_by`, `aggregation_authorized_at`, `is_duplicate_suppressed`, `review_status: approved`, `risk_tier` (final value), `required_resolution_mode`, persistence of a `sensitive_or_high_impact_inference` belief.

## Confidence vs. status vs. eligibility (section 6.5)

These are three separate gates. A change is suspect if it lets a high `confidence` value alone unlock something that `sensitivity_class`, `disallowed_contexts`, `persistence_policy`, staleness, an `explicit_user_correction`, or risk-policy would otherwise block.

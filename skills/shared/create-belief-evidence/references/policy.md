# Policy Checklist - create-belief-evidence

Canonical source: Blueprint section 6 "Evidence & Update Rules", section 6.1.1-6.1.2, section 6.2.1.

## Canonical `source_type` enum and default reliability (section 6.2.1)

Exactly these seven strings - no others, no legacy aliases:

| source_type | Default reliability | Notes |
|---|---|---|
| `explicit_user_correction` | 1.00 | Override authority beyond its numeric reliability. |
| `recorded_event` | 0.95 | Reliable for *what* happened, not *why*. |
| `explicit_user_statement` | 0.85 | Self-report; may differ from observed behavior. |
| `repeated_pattern_summary` | 0.80 | Derived; must not double-count its own leaf events. |
| `model_observation` | 0.70 | Structured, evidence-tied, still model-generated. |
| `llm_inference` | 0.55 | Semantic inference; provisional until corroborated. |
| `unverified_hypothesis` | 0.35 | Exploration-only; never drives persistent high confidence. |

## Aggregation authorization boundary (section 6.1.2)

- Default is always `leaf_default`. This skill proposes `proposed_aggregation_mode`; it never writes `authorized_aggregation_mode`.
- `aggregate_replacement` is valid only when `replaces_evidence_ids` is explicit, the referenced rows exist, belong to the same belief/user scope, and have valid `source_event_ids` provenance - and even then, only backend policy code may authorize it.
- If a `repeated_pattern_summary`'s `source_event_ids` overlap `recorded_event` rows already counted for the same belief, do not treat that as automatically fine - flag the overlap; backend deduplication decides whether it is suppressed.

## Independence & deduplication (section 6.1.1)

- Two evidence items count independently only if they represent genuinely different underlying events/observations, not a retry, rephrasing, or repeated extraction of the same source event.
- `source_event_ids` plus `independence_group` are what downstream dedup keys on - populate both honestly; do not pad `independence_group` to make retries look independent.

## Sensitivity / persistence defaults (section 3.4.4 registry)

Before proposing evidence tied to a `sensitive_or_high_impact_inference` belief type, confirm the registry's `restricted` sensitivity and `do_not_persist` default - this skill must flag such proposals for stricter review rather than proposing normal persistence.

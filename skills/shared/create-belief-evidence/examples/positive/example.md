# Positive Example

## Input observation

```json
{
  "observation_id": "obs_310",
  "observation": "User has repeatedly completed after-work workouts (3 of the last 3 workout events).",
  "confidence": 0.7
}
```

## Correct proposed evidence

```json
{
  "evidence_id": "bev_501",
  "belief_id": "bel_88",
  "user_id": "usr_17",
  "direction": "support",
  "event_id": "evt_1042",
  "observation_id": "obs_310",
  "source_event_ids": ["evt_1042"],
  "source_type": "recorded_event",
  "independence_group": "evt_1042",
  "proposed_aggregation_mode": "leaf_default",
  "context_key": "fitness",
  "strength": 0.85,
  "source_reliability": 0.95,
  "observed_at": "2026-08-22T17:20:00-05:00",
  "decay_lambda": 0.015,
  "created_at": "2026-08-22T17:21:10-05:00",
  "model_version": "observation-model-0.6",
  "prompt_version": "evidence_prompt_05",
  "schema_version": "6",
  "scoring_version": "belief-score-0.6",
  "canonicalizer_version": "canon-0.6",
  "policy_version": "policy-0.6"
}
```

## Why this is correct

- `source_type` is `recorded_event` because this evidence traces to an actual completed-goal event, not an inference about it.
- `source_reliability` (0.95) matches the canonical default for `recorded_event` exactly - no invented number.
- `proposed_aggregation_mode` is `leaf_default`; the row does not set `authorized_aggregation_mode` at all.
- `source_event_ids` contains the real event ID, and `independence_group` matches it - this is one genuinely independent leaf item, not a repackaged duplicate.

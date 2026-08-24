# Positive Example

## Input events (3 of many)

```json
[
  {"event_id": "evt_1040", "user_id": "usr_17", "event_type": "goal_completed", "timestamp": "2026-08-15T17:25:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:25"}},
  {"event_id": "evt_1041", "user_id": "usr_17", "event_type": "goal_completed", "timestamp": "2026-08-18T17:15:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:15"}},
  {"event_id": "evt_1042", "user_id": "usr_17", "event_type": "goal_completed", "timestamp": "2026-08-22T17:20:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:20"}}
]
```

## Correct output

```json
{
  "observation_id": "obs_310",
  "user_id": "usr_17",
  "category": "routine",
  "observation": "User has repeatedly completed after-work workouts (3 of the last 3 workout events, all completed within 25 minutes of the 17:00 scheduled time).",
  "importance": 0.62,
  "confidence": 0.7,
  "created_at": "2026-08-22T17:21:00-05:00",
  "schema_version": "6",
  "scoring_version": "belief-score-0.6",
  "canonicalizer_version": "canon-0.6",
  "policy_version": "policy-0.6"
}
```

```json
[
  {"observation_id": "obs_310", "event_id": "evt_1040", "link_role": "supporting"},
  {"observation_id": "obs_310", "event_id": "evt_1041", "link_role": "supporting"},
  {"observation_id": "obs_310", "event_id": "evt_1042", "link_role": "primary"}
]
```

## Why this is correct

- The observation text describes only what happened (completion pattern), not why, and not a claim about the user's character.
- Confidence (0.7) is moderate, not maxed out, even with 3 consistent events - leaving room for the belief engine's own scoring to be the actual authority on confidence.
- Every supporting event is explicitly linked via `observation_events`, with exactly one `primary`.

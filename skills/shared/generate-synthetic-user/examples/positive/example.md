# Positive Example - Persona A (Schedule Shift), Slice

## Hidden pattern (written down before generation)

Weeks 1-4: evening workouts (~17:00) succeed consistently. A new job starts at the beginning of week 5. Weeks 5-8: morning workouts (~06:30) succeed consistently instead; evening attempts, when they occur, are missed.

## Sample generated events (slice)

```json
[
  {"event_id": "evt_synth_scheduleshift_014", "user_id": "usr_synth_scheduleshift_01", "event_type": "goal_completed", "timestamp": "2026-06-08T17:10:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:10"}, "goal_id": "goal_synth_workout", "schema_version": "6", "scoring_version": "belief-score-0.6", "canonicalizer_version": "canon-0.6", "policy_version": "policy-0.6"},
  {"event_id": "evt_synth_scheduleshift_015", "user_id": "usr_synth_scheduleshift_01", "event_type": "goal_missed", "timestamp": "2026-07-06T17:00:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "17:00", "reason_hint": "new job start date"}, "goal_id": "goal_synth_workout", "schema_version": "6", "scoring_version": "belief-score-0.6", "canonicalizer_version": "canon-0.6", "policy_version": "policy-0.6"},
  {"event_id": "evt_synth_scheduleshift_016", "user_id": "usr_synth_scheduleshift_01", "event_type": "goal_completed", "timestamp": "2026-07-08T06:35:00-05:00", "structured_data": {"goal": "workout", "scheduled_time": "06:30", "completed_time": "06:35"}, "goal_id": "goal_synth_workout", "schema_version": "6", "scoring_version": "belief-score-0.6", "canonicalizer_version": "canon-0.6", "policy_version": "policy-0.6"}
]
```

## Why this is correct

- The shift is gradual and evidenced (a missed evening attempt right at the transition point, then a run of successful morning attempts), not an instant hard cut that would be trivial to detect without real recency-weighted reasoning.
- Timestamps span real weeks, so decay/recency logic in `update-user-beliefs` has something meaningful to act on.
- `goal_id` is reused consistently, so linkage across the shift is traceable.

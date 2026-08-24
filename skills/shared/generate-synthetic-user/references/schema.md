# Schema Checklist - generate-synthetic-user

Canonical source: Blueprint section 5.3 "Concrete JSON Contracts" (`user_event`).

## `user_event` fields to emit

`event_id, user_id, event_type, timestamp, raw_content, structured_data, source, goal_id, session_id, schema_version, scoring_version, canonicalizer_version, policy_version`

- `timestamp`: ISO 8601 with timezone offset, spread across real elapsed time - not all identical or all within one hour, unless the persona specifically calls for a burst.
- `raw_content`: `null` when the event is purely structured (e.g. a completed goal); a string when simulating a chat/journal-style entry (useful for exercising the raw-content trust boundary in adversarial cases).
- `structured_data`: a small JSON object describing what happened (e.g. `{"goal": "workout", "scheduled_time": "17:00", "completed_time": "17:20"}`).
- `user_id`: use a clearly synthetic namespace, e.g. `usr_synth_scheduleshift_01`, never a real-looking bare ID.

## Naming convention for generated IDs

- `event_id`: `evt_synth_<persona>_<sequence>`
- `session_id`, `goal_id`: reuse consistently within one persona's timeline where it makes narrative sense (e.g. the same `goal_id` across all workout-completion events).

# Schema Checklist - extract-user-observations

Canonical source: Blueprint section 5 "Data Model" and section 5.3 "Concrete JSON Contracts". This file is a checklist for quick recall, not a substitute for reading those sections when a field's meaning is unclear.

## `user_events` (input, read-only here)

`event_id, user_id, event_type, timestamp, raw_content, structured_data, source, goal_id, session_id, schema_version, scoring_version, canonicalizer_version, policy_version`

`raw_content` is untrusted (see `policy.md`).

## `user_observations` (output)

`observation_id, user_id, category, observation, importance, confidence, created_at, schema_version, scoring_version, canonicalizer_version, policy_version`

- `importance` and `confidence` are floats in `[0, 1]`.
- `observation` is prose describing an observed pattern, not a belief statement and not a value judgment.
- Do not add an `event_id` or `event_ids` field directly on this row - provenance goes through `observation_events` only.

## `observation_events` (output, required join)

`observation_id, event_id, link_role, created_at, schema_version, scoring_version, canonicalizer_version, policy_version`

- `link_role` is `primary` or `supporting`. Every observation needs at least one `primary` row.
- This table exists specifically to remove the old ambiguous `event_id` vs `event_ids` representation - never reintroduce that ambiguity.

## Version fields

Every row you emit needs `schema_version`, `scoring_version`, `canonicalizer_version`, and `policy_version` populated from the active configuration, not hard-coded literals invented by this skill.

# Adversarial Example - One-Off Event Must Not Look Like a Pattern

## Incorrect generation

```json
[
  {"event_id": "evt_synth_x_001", "user_id": "usr_synth_x_01", "event_type": "goal_completed", "timestamp": "2026-06-01T09:00:00-05:00", "structured_data": {"goal": "meditation", "scheduled_time": "09:00", "completed_time": "09:00"}}
]
```

This is the *only* meditation-related event in the whole dataset, yet it uses the same clean, on-time, high-confidence shape as a genuinely repeated behavior would. A naive extractor could over-fit and promote "user meditates at 9am" to a belief from a single event.

## Why this is wrong

Generating a single event with a "confident-looking" shape, with nothing to distinguish it from repeated behavior, creates a dataset that cannot test whether the belief engine correctly refuses to promote a belief from one data point (blueprint section 6.2: "no inferred belief may exceed candidate status" at very low event counts; section 14 "Stability": "No high-confidence promotion from a single weak event").

## Correct handling

If a one-off event is intentionally part of the dataset (to test that the system does *not* over-infer from it), label the intent explicitly in the dataset's accompanying hidden-pattern note:

```
Note: evt_synth_x_001 is a deliberate one-off meditation event with no
recurrence anywhere else in the dataset. Expected system behavior: no
belief about a meditation routine should reach beyond `candidate` status,
and ideally none should be created at all from a single instance.
```

This turns the one-off event from an accidental test-data flaw into a deliberate, documented adversarial case that `evaluate-user-model` (planned, not yet implemented) can check against once it exists, and that a human reviewer can check against by hand today.

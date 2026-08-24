---
name: generate-synthetic-user
description: This skill should be used when a coding agent needs to create synthetic chronological user_event datasets with hidden ground-truth patterns for the Better You evaluation harness - including the three canonical personas (schedule shift, intention vs behavior, temporary disruption) or new ones modeled on them. It should not be used to write the evaluation/scoring logic itself (evaluate-user-model, planned but not yet implemented) or to fabricate real user data.
---

# Generate Synthetic User

Create controlled, chronological synthetic `user_event` datasets with a hidden ground-truth pattern the belief engine is expected to discover, adapt to, or correctly refuse to over-infer from.

## When to use this skill

Use this skill when building or extending the evaluation harness (blueprint section 4.G, section 12.7, section 13), including:

- Generating a new persona dataset for a specific failure mode (cold-start, contradiction, temporal shift, intention-vs-habit).
- Regenerating one of the three required canonical personas for a regression suite.
- Producing adversarial event sequences (sarcasm, one-off events, missing context, contradictory statements) for the observation-extraction test suite.

Do not use this skill to write the code that scores or evaluates the generated data against golden expectations - that is `evaluate-user-model` (planned, not yet implemented; see `skills/shared/evaluate-user-model/SKILL.md` for what to do in the meantime). Do not use it to synthesize data intended to be mistaken for a real user; synthetic datasets must always be clearly marked as such (e.g. `usr_synth_*` IDs) so they can never leak into real personalization decisions.

## Canonical source of truth

```bash
python tools/document_reader/read_document.py "Blueprint/Better_You_Adaptive_User_Model_RnD_Blueprint_v0.6.2.docx" --format json
```

Then find, by heading text:

- section 13 "Three Initial Synthetic Evaluation Personas" - the required hidden patterns and what the system must demonstrate for each.
- section 6.2 "Cold-Start Behavior (0-30 Events)" - event-count bands a generated dataset should be able to exercise.
- section 5.3 "Concrete JSON Contracts" - the exact `user_event` shape to emit.
- section 18 "Definition of Done" - "At least three synthetic personas with 30-100 events each" is a release gate, not optional.

`scripts/generate_events.py` in this skill is a starting-point generator for the three canonical personas - extend it rather than writing a parallel generator from scratch.

## Procedure

1. **Every generated dataset needs a documented hidden ground-truth pattern**, written down *before* generating events, so downstream evaluation has something concrete to check against. Never generate "plausible-looking" events without a specific pattern in mind.
2. **Match the required personas exactly where the task calls for the canonical three** (section 13):
   - **A - Schedule shift**: evening exercise historically works, then a new job makes mornings work instead. The system must learn the initial pattern, then lower confidence and adapt as the pattern changes - not cling to the old pattern, and not flip instantly on the first contradicting event.
   - **B - Intention vs. behavior**: user states a goal (e.g. nightly study) but actual successful completions cluster elsewhere (e.g. Saturday mornings). The system must distinguish stated intention from observed habit without dismissing the stated goal outright.
   - **C - Temporary disruption**: a normally consistent user becomes inconsistent during a clearly-contextualized period (travel, poor sleep, new workload). The system must infer state/context, not assign a permanent negative trait.
3. **Generate 30-100 events per persona** (section 18 Definition of Done), spanning enough real time for decay/recency logic to matter - not all clustered on one timestamp.
4. **Emit the exact `user_event` contract** (`event_id, user_id, event_type, timestamp, raw_content, structured_data, source, goal_id, session_id, schema_version, scoring_version, canonicalizer_version, policy_version`) - do not invent fields the schema does not define.
5. **Mark every synthetic user unambiguously** (e.g. `user_id: "usr_synth_<persona>_<n>"`) so synthetic data can never be mistaken for or merged with real user data.
6. **Include adversarial/edge cases deliberately** when the task calls for them: a one-off event that should not create a pattern, a sarcastic or ambiguous statement, an event with missing context, directly contradictory statements in close succession.

## Completion check

A generated persona dataset is complete only when:

- It has a written hidden-pattern description alongside it (for scoring against later).
- It has 30-100 chronological events with real, spread-out timestamps.
- Every event matches the `user_event` schema exactly, with a `usr_synth_*`-style ID.
- For the three canonical personas, the specific pattern in section 13 is actually present in the generated timeline, not just implied.

## Examples

`examples/positive/example.md` shows a correctly structured Persona A (schedule shift) event slice. `examples/adversarial/example.md` shows a one-off event that must not be allowed to look like a repeated pattern, and why.

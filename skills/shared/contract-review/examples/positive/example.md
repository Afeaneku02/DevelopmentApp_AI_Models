# Positive Example - Passing Review

## Change under review

A PR adds a new `constraint_or_aversion` belief path: when a user repeatedly skips a recommended action type, `create-belief-evidence` proposes `contradict` evidence against the relevant `routine_or_preference` belief, and `update-user-beliefs` recomputes it.

## Review findings

```
PASS

Checked:
- belief_type used: routine_or_preference (registered, decay_lambda=0.015, expected_source_types=2) -- OK, no new type invented.
- source_type used: recorded_event (matches skip events) -- OK, canonical value, correct default reliability (0.95).
- direction: contradict -- correctly set based on the skip pattern, not forced to fit an existing belief.
- Confidence path: recomputed via compute_confidence() over the full ledger, not incremented -- OK.
- Version fields present on all new belief_evidence rows: schema_version, scoring_version, canonicalizer_version, policy_version -- OK.
- Regression fixture added: tests/belief_engine/test_contradiction_lowers_confidence.py reproduces the
  0.88 -> 0.79 -> 0.63 -> 0.41 -> contested sequence from Blueprint section 6.1 -- OK.

No violations found. Safe to merge.
```

## Why this is a correct review

It names the exact contracts checked (not a vague "looks good"), confirms the confidence path is ledger-driven rather than incremental, and confirms a regression fixture exists for the specific behavior the change adds - matching the completion check in SKILL.md.

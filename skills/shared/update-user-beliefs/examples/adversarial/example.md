# Adversarial Example - The "Just Add a Bump" Shortcut

## Tempting but incorrect approach

A new supporting evidence item arrives for a belief currently at `confidence = 0.68`. A shortcut implementation does:

```python
belief.confidence = min(0.98, belief.confidence + 0.05)  # WRONG
```

## Why this is wrong

- Confidence is not a running total that a new item nudges upward. section 3.4.1 requires confidence to be **recomputed from the full deduplicated evidence ledger** every time - `support_mass`, `contradiction_mass`, and `effective_support_count` recalculated from scratch, not incremented.
- This shortcut cannot decrease confidence when contradicting evidence arrives, because it only ever adds. It silently breaks the contradiction-responsiveness requirement (section 14 "Contradiction responsiveness" metric) and the worked example in section 6.1 where repeated new evidence should be able to pull confidence from `0.88` down through `0.79 -> 0.63 -> 0.41 -> contested`.
- It has no way to reproduce a past `scoring_version`'s result, because it depends on whatever the *previous* stored confidence happened to be rather than being a pure function of `(evidence, scoring_config)`.

## Correct handling

```python
from scripts.compute_confidence import compute_confidence, DEFAULT_WEIGHTS

# Re-run the full computation over ALL active, deduplicated evidence for the belief,
# including the new item -- never seed it from the belief's previous confidence.
result = compute_confidence(all_active_evidence_for_belief, DEFAULT_WEIGHTS, expected_source_types=2)
belief.confidence = result["confidence"]
```

This is deterministic and reproducible: given the same `belief_evidence` ledger and the same `scoring_config`, it always recomputes the same confidence, and it responds correctly in both directions as evidence accumulates or contradicts.

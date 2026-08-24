# Adversarial Example - A Change That Looks Fine but Crosses the Authorization Boundary

## Change under review

A PR "simplifies" evidence aggregation by having the LLM extractor set `authorized_aggregation_mode` directly whenever its own confidence in a `repeated_pattern_summary` proposal is high, reasoning: "the model is usually right, and this saves a review step."

```python
# inside the extractor, NOT backend policy code
if pattern_confidence > 0.8:
    evidence_row["authorized_aggregation_mode"] = "aggregate_replacement"
    evidence_row["aggregation_authorized_by"] = "extractor"
```

## Why a superficial pass would be wrong

The diff is small, the tests still pass (because the test suite never asserted *who* is allowed to set this field), and the feature "works" in the sense that aggregation now happens more often. A review that only checks "does aggregation still function" would pass this.

## Correct review outcome

```
FAIL

Violation: extractor code sets authorized_aggregation_mode and aggregation_authorized_by directly (Blueprint section 6.1.2, section 22 Decision Log: "Backend authorizes evidence aggregation... LLM/extractor proposals cannot suppress or replace evidence directly").

Why it matters: this lets model output single-handedly replace/suppress leaf evidence, which can inflate or deflate confidence without human-auditable authorization, and defeats the "leaf_default is the safe fallback" guarantee in section 14.1's contract tests.

Required fix: the extractor may only set proposed_aggregation_mode. authorized_aggregation_mode must be set exclusively by a backend evidence-policy component, gated by the deterministic checks in Blueprint section 6.1.2 (explicit valid replaces_evidence_ids, same belief/user scope, valid provenance).

Missing regression coverage: no test asserts that changing only the extractor's proposal cannot change authorized_aggregation_mode or effective counts -- this exact case is required by section 14.1 ("Uncertain/risky aggregation proposals remain leaf_default or review; changing only the LLM proposal cannot change effective counts").
```

## Why this is the correct review

It does not accept "the feature works" as sufficient. It names the exact section that the change violates, explains the concrete risk (unaudited confidence manipulation), and calls out the missing regression test by name rather than just the code defect.

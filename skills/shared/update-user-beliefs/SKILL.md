---
name: update-user-beliefs
description: This skill should be used when a coding agent needs to recompute belief confidence, effective evidence counts, and lifecycle status for the Better You adaptive user model from the belief_evidence ledger, or when implementing/reviewing the deterministic scoring function itself. It should not be used to propose new evidence rows (create-belief-evidence) or to decide recommendation risk policy (apply-risk-policy, planned but not yet implemented).
---

# Update User Beliefs

Recompute a belief's confidence, `D`/`Q`/`R`/`V`, effective counts, and lifecycle status from its active, non-suppressed `belief_evidence` ledger - deterministically, reproducibly, and fail-closed when recomputation cannot complete.

## When to use this skill

Use this skill when implementing, reviewing, or testing the Belief Engine's scoring/lifecycle logic (blueprint section 4.B), including:

- Implementing or reviewing the confidence function itself (section 3.4.1).
- Implementing or reviewing lifecycle status transitions (section 3.4.2) or the recompute-on-invalidation path (section 6.0.1-6.0.2).
- Reviewing a PR that touches `scoring_config`, `belief_type_registry`, or anything under `user_beliefs`.

Do not use this skill to propose new `belief_evidence` rows - that is `create-belief-evidence`. Do not use it to decide recommendation risk tier or exploration eligibility - that is `apply-risk-policy` (planned, not yet implemented; see `skills/shared/apply-risk-policy/SKILL.md`). Do not use it to rebuild belief state specifically *after* deletion/reset/suppression as its own workflow - that narrower recompute-triggered path is `recompute-user-model` (also planned, not yet implemented); this skill is the scoring function those recomputations will call.

## Canonical source of truth

```bash
python tools/document_reader/read_document.py "Blueprint/Better_You_Adaptive_User_Model_RnD_Blueprint_v0.6.2.docx" --format json
```

Then find, by heading text:

- section 3.4.1 "Concrete Belief-Update Algorithm" - the exact formulas and pseudocode this skill implements.
- section 3.4.2 "Belief Status Thresholds" - lifecycle status rules.
- section 3.4.3 "Versioned Scoring Configuration" - why every constant must come from `scoring_config`, never be hard-coded.
- section 6.0.1-6.0.2 - mandatory recompute and fail-closed behavior on recompute failure.

`scripts/compute_confidence.py` in this skill is a working reference implementation of section 3.4.1's algorithm, including its own regression check against the blueprint's worked small-sample example. Treat it as the executable specification, not just documentation - read it before hand-writing the formula again from memory.

## Procedure

1. **Never compute confidence from a raw event count.** Confidence is always recomputed from the deduplicated `belief_evidence` ledger (`support_mass`, `contradiction_mass`, `effective_support_count`), never by adding a fixed increment to a prior confidence value.
2. **Follow the exact branch order from section 3.4.1**: first check the invalidation/no-active-evidence branch (`recompute_reason` in `{deletion, reset, duplicate_suppression, policy_invalidation}` and `active_evidence_count == 0`) - this must return before the generic no-evidence branch. Then check the generic `support_mass + contradiction_mass == 0` branch, returning `confidence: 0.0, D: null, status: candidate, reason: no_evidence`. Only after both of those can `D`, `BaseSignal`, and `confidence` be computed.
3. **`D = null` means no evidence, not contradiction.** Never conflate a missing-evidence state with a contradicted-evidence state in logs or in status.
4. **Use `effective_support_count`, not raw counts, for the small-sample correction.** `support_factor = 1 - exp(-effective_support_count / 4.0)`, and `confidence = clamp(BaseSignal * support_factor, 0.02, 0.98)`. The 0.98 ceiling is intentional - never let inferred confidence reach or round to 1.0.
5. **Load every weight, decay lambda, source reliability, and threshold from `scoring_config`** for the recorded `scoring_version` - never hard-code `wD/wQ/wR/wV` or lifecycle thresholds inline. A new calibration creates a new `scoring_version`; it never silently changes what an old `scoring_version` means.
6. **Validate `belief_type` against the active `belief_type_registry_version`** before using its decay lambda / expected source types / sensitivity / persistence defaults. An unregistered `belief_type` must be rejected or routed to review - never silently mapped to a generic type.
7. **On recompute failure after an invalidation trigger, fail closed.** Set `locked_until_recompute=true`, exclude the belief from profile/recommendation use, and keep it locked through pending/running/failed retries. Never let a stale cached confidence/status serve traffic once its evidence has changed.
8. **Separate confidence from status/eligibility.** A numerically high confidence never overrides `sensitivity_class`, `disallowed_contexts`, `persistence_policy`, staleness, or an explicit user correction.

## Completion check

A scoring implementation or change is complete only when:

- `scripts/compute_confidence.py`'s built-in self-test passes (reproduces the blueprint's worked example: two recent reliable supporting items -> `effective_support_count ~= 1.8` -> `support_factor ~= 0.36`).
- The invalidation/no-active-evidence branch is provably ordered before the generic no-evidence branch (a unit test, not a read-through).
- Every constant used came from a loaded `scoring_config`, not an inline literal.
- Recompute failure leaves the belief locked and excluded, never silently serving a stale cache.

## Examples

`examples/positive/example.md` walks the worked confidence calculation end to end. `examples/adversarial/example.md` shows a tempting-but-wrong shortcut (adding a flat bump to prior confidence on new supporting evidence) and why it violates the ledger-driven recompute rule.

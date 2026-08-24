# Policy Checklist - update-user-beliefs

Canonical source: Blueprint section 3.4.1, section 3.4.2, section 6.0.1, section 6.0.2. `scripts/compute_confidence.py` is the executable version of this file - prefer reading/running it over re-deriving the formula from prose.

## Branch order (section 3.4.1, reinforced by section 6.0.2 and the v0.6.1 stabilization change)

1. If `recompute_reason` is one of `deletion|reset|duplicate_suppression|policy_invalidation` **and** `active_evidence_count == 0`: return `confidence=0.0, D=null, status=<lifecycle_policy.no_active_evidence_status>, reason="no_active_evidence_after_deletion_or_suppression"`. This branch runs **before** the generic no-evidence branch below.
2. Else if `support_mass + contradiction_mass == 0`: return `confidence=0.0, D=null, status="candidate", reason="no_evidence"`.
3. Else compute `D = support_mass / (support_mass + contradiction_mass)`, then `BaseSignal`, `support_factor`, and `confidence`.

## Formulas (section 3.4.1)

```
decay_i        = exp(-lambda_i * age_days_i)
quality_i      = strength_i * source_reliability_i
mass_i         = quality_i * decay_i
count_weight_i = source_reliability_i * decay_i

support_mass       = sum(mass_i for support evidence)
contradiction_mass = sum(mass_i for contradict evidence)

D = support_mass / (support_mass + contradiction_mass)     # only when denominator > 0
Q = weighted_mean(strength_i * source_reliability_i, weight=decay_i)
R = weighted_mean(decay_i, weight=source_reliability_i)
V = min(1, unique_independent_source_types / expected_source_types_for_belief_type)

BaseSignal      = wD*D + wQ*Q + wR*R + wV*V         # wD=.45, wQ=.25, wR=.15, wV=.15 by default, always from scoring_config
support_factor  = 1 - exp(-effective_support_count / 4.0)
confidence      = clamp(BaseSignal * support_factor, 0.02, 0.98)
```

Only one contradiction-sensitive term exists (`D`); never add a second ordinary "contradiction penalty" multiplier. Extreme contradiction is a lifecycle/status concern (`contested`), not a second confidence discount.

## Lifecycle thresholds (section 3.4.2)

| Status | Rule |
|---|---|
| `candidate` | `confidence < 0.35` OR `effective_support_count < 1.5` |
| `provisional` | `0.35-0.64` AND `effective_support_count >= 1.5` |
| `validated` | `>= 0.65` AND `effective_support_count >= 3.5` AND `D >= 0.75` |
| `contested` | `D <= 0.60` for a prior validated belief, or strong recent contradiction |
| `outdated` | evidence decays below usefulness / outside retention window |
| `rejected` | explicit user correction, policy/schema violation, or evidence review finds it unsupported |

## Fail-closed recompute (section 6.0.1-6.0.2)

- `scoring_recompute_status = failed` is an authorization failure, not a warning.
- On failure after an invalidation trigger: set `locked_until_recompute=true`, exclude the belief from profile assembly and recommendation ranking, and keep it locked through `pending`/`running` retries.
- Clear the lock and update `last_successful_recompute_at` only after a completed, successful recomputation.

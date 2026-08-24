# Definition of Done Focus Areas - contract-review

Canonical source: Blueprint section 18 "Definition of Done for R&D v0.6.1" and section 29 "v0.6.2 Change Summary" (the skill layer itself is now part of this contract). Read section 18 in full for a change of any real size; this file only orders the checks by how often they are actually violated in practice.

## Highest-value checks (check these first)

1. **Branch order on scoring invalidation** - the deletion/reset/suppression no-active-evidence return must execute before the generic no-evidence return (section 14.1, section 18).
2. **Evidence provenance stays on `source_event_ids`** - never reintroduce a singular `event_id`-only provenance path that could diverge from it.
3. **Aggregation stays backend-authorized** - a proposal changing only `proposed_aggregation_mode` must never change effective support/confidence on its own.
4. **Unknown `belief_type` / unknown recommendation context fail closed** - rejected or routed to review, never silently mapped to a generic type or a lenient risk tier.
5. **Confidence ceiling (`0.98`) and floor (`0.02`) are respected**, and no code path lets confidence reach exactly `1.0`.

## Also check when the change touches...

- **Deletion/reset**: does it cascade through events, observations, evidence, beliefs, snapshots as scoped, and does it force a recompute rather than leaving a stale cache?
- **Canonicalization**: does an automatic merge stay within "narrow, same-context, same-polarity, high-similarity" bounds, with anything riskier routed to review or kept separate?
- **Recommendations**: is `risk_tier` resolved by backend policy lookup (exact context -> domain -> global fallback), never by model output? Is `exploration_applied` forced `false` whenever `review_required` is `true`?
- **Manual review resolution**: for high-risk/sensitive cases, is the resolution mode reviewer/domain-policy-approval, not bare user confirmation, unless a domain policy explicitly authorizes that exception?

## What a "pass" review must not skip

A pass verdict without checking whether a regression fixture exists for the specific failure mode the change addresses is not a complete review - section 10.4 and section 18 both treat missing test coverage as a blocking gap, not a nice-to-have.

---
name: create-belief-evidence
description: This skill should be used when a coding agent needs to convert a validated user_observation or user_event into a belief_evidence proposal for the Better You adaptive user model - assigning canonical source_type, reliability, provenance, and a safe aggregation mode. It should not be used to compute confidence, decide belief status, or authorize evidence aggregation/suppression - those are backend/update-user-beliefs responsibilities.
---

# Create Belief Evidence

Convert an observation or event into a `belief_evidence` proposal: correctly typed, correctly provenanced, and defaulted to the safe aggregation mode. This skill proposes; it never authorizes a persistent write, a merge, or a suppression.

## When to use this skill

Use this skill when implementing or reviewing the step that turns an observation/event into a candidate `belief_evidence` row - the point where "this happened" becomes "this is evidence for or against belief X." This includes:

- Writing or reviewing the evidence-proposal function for the Belief Engine (blueprint section 4.B).
- Reviewing a PR that sets `source_type`, `source_reliability`, `strength`, `direction`, `source_event_ids`, or `proposed_aggregation_mode` on a `belief_evidence` row.
- Auditing whether a given evidence row's provenance is real (traces to actual events) or fabricated.

Do not use this skill to compute `confidence`, `D`, `Q`, `R`, `V`, or belief `status` - that is `update-user-beliefs`. Do not use it to decide `authorized_aggregation_mode` - only backend policy code may set that field; this skill may only ever write `proposed_aggregation_mode`.

## Canonical source of truth

```bash
python tools/document_reader/read_document.py "Blueprint/Better_You_Adaptive_User_Model_RnD_Blueprint_v0.6.2.docx" --format json
```

Then find, by heading text:

- section 5 "Data Model" - full `belief_evidence` field list.
- section 5.3 "Concrete JSON Contracts" - a worked `belief_evidence` example.
- section 6 "Evidence & Update Rules" - the seven canonical `source_type` values and their default trust/reliability.
- section 6.1.1 "Evidence Independence & Deduplication" and section 6.1.2 "repeated_pattern_summary Anti-Double-Counting" - the rules this skill must not silently violate.
- section 6.0.1 "Evidence Invalidation & Mandatory Recompute" - why this skill never deletes or replaces existing evidence itself.

See `references/schema.md` for the field checklist and `references/policy.md` for the canonical `source_type` enum, default reliabilities, and aggregation-authorization boundary distilled for this task.

## Procedure

1. **Pick exactly one of the seven canonical `source_type` values** - `explicit_user_correction`, `recorded_event`, `explicit_user_statement`, `repeated_pattern_summary`, `model_observation`, `llm_inference`, `unverified_hypothesis`. Never invent an eighth value or reuse a legacy string; if none of the seven genuinely fits, stop and flag it rather than guessing.
2. **Set `direction` (`support` or `contradict`) based on what the evidence actually shows**, not on what would be convenient for an existing belief. Evidence that is ambiguous about direction should not be forced into either bucket.
3. **Populate `source_event_ids` with the real, traceable event ID(s)** behind this evidence. This is the sole authoritative provenance field - never invent a separate singular `event_id`-only provenance path that could diverge from it.
4. **Always propose `proposed_aggregation_mode: "leaf_default"` unless there is a concrete, valid `replaces_evidence_ids` set** you can point to for an `aggregate_replacement` proposal. Even then, this skill only *proposes* - it must leave `authorized_aggregation_mode` unset/`leaf_default` and never write to it directly; only backend evidence-policy code authorizes `aggregate_replacement`.
5. **Never let a `repeated_pattern_summary` proposal double-count events already covered by `recorded_event` rows for the same belief.** If the summary's `source_event_ids` overlap existing supporting evidence, that overlap is exactly what backend deduplication (section 6.1.1) exists to catch - do not try to pre-empt it by silently dropping or merging rows yourself.
6. **Respect `sensitivity_class` / `persistence_policy` defaults from the belief's `belief_type`** (see the registry in section 3.4.4) - never propose persisting evidence for a `sensitive_or_high_impact_inference` belief type without flagging that it needs stricter review; never override a `do_not_persist` default.
7. **Run `scripts/validate_evidence.py`** against every proposed evidence row before treating the output as done.

## Completion check

A batch of evidence proposals is complete only when:

- Every row's `source_type` is one of the seven canonical values, with `source_reliability` matching (or explicitly justifying a deviation from) the versioned default for that type.
- Every row's `source_event_ids` is non-empty and traceable to real events.
- No row sets `authorized_aggregation_mode` itself - only `proposed_aggregation_mode`.
- `scripts/validate_evidence.py` exits 0 against the batch.

## Examples

`examples/positive/example.md` shows a correctly proposed `recorded_event` support row. `examples/adversarial/example.md` shows a proposal that tries to self-authorize an `aggregate_replacement` and self-assign `confidence`, and the correct rejection of both.

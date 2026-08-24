# Adversarial Example - Self-Authorized Aggregation and Self-Assigned Confidence

## Tempting but incorrect output

A generation pass under pressure to "just make the belief look validated" might produce:

```json
{
  "evidence_id": "bev_777",
  "belief_id": "bel_88",
  "direction": "support",
  "source_event_ids": ["evt_2010", "evt_2011", "evt_2012"],
  "source_type": "repeated_pattern_summary",
  "authorized_aggregation_mode": "aggregate_replacement",
  "replaces_evidence_ids": ["bev_501", "bev_502", "bev_503"],
  "aggregation_authorized_by": "extractor",
  "confidence_hint": 0.95,
  "strength": 1.0,
  "source_reliability": 0.95
}
```

## Why this is wrong

- **Self-authorization**: this skill set `authorized_aggregation_mode` and `aggregation_authorized_by` itself. Only backend evidence-policy code may authorize `aggregate_replacement`; this skill may only ever write `proposed_aggregation_mode`.
- **Inflated reliability**: `repeated_pattern_summary` defaults to `0.80` reliability, not `0.95` (that value belongs to `recorded_event`). Do not borrow a different source_type's default to make evidence look stronger.
- **Out-of-contract field**: `confidence_hint` does not exist on `belief_evidence`. This skill never proposes a confidence value at all - confidence is computed by `update-user-beliefs` from the evidence ledger, never asserted directly by evidence-creation.

## Correct handling

```json
{
  "evidence_id": "bev_777",
  "belief_id": "bel_88",
  "direction": "support",
  "source_event_ids": ["evt_2010", "evt_2011", "evt_2012"],
  "source_type": "repeated_pattern_summary",
  "proposed_aggregation_mode": "aggregate_replacement",
  "replaces_evidence_ids": ["bev_501", "bev_502", "bev_503"],
  "strength": 0.85,
  "source_reliability": 0.80,
  "aggregation_review_required": true,
  "aggregation_review_status": "pending"
}
```

This proposes the replacement and flags it for review, but leaves `authorized_aggregation_mode` for backend policy to set (defaulting to `leaf_default` until it does), uses the correct reliability default, and omits any confidence field entirely.

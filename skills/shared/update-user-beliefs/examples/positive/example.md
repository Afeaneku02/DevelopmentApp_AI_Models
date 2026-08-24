# Positive Example - Worked Confidence Calculation

## Input evidence for belief `bel_88` ("higher_adherence_after_work")

Four supporting `recorded_event` items and one contradicting `recorded_event` item, all recent:

```python
from scripts.compute_confidence import EvidenceItem, compute_confidence, DEFAULT_WEIGHTS

evidence = [
    EvidenceItem("support", strength=0.9, source_reliability=0.95, decay_lambda=0.015, age_days=2, source_type="recorded_event"),
    EvidenceItem("support", strength=0.9, source_reliability=0.95, decay_lambda=0.015, age_days=5, source_type="recorded_event"),
    EvidenceItem("support", strength=0.85, source_reliability=0.85, decay_lambda=0.015, age_days=10, source_type="explicit_user_statement"),
    EvidenceItem("support", strength=0.8, source_reliability=0.95, decay_lambda=0.015, age_days=1, source_type="recorded_event"),
    EvidenceItem("contradict", strength=0.6, source_reliability=0.95, decay_lambda=0.015, age_days=3, source_type="recorded_event"),
]

result = compute_confidence(evidence, DEFAULT_WEIGHTS, expected_source_types=2)
```

## Why this is correct

- `D` comes out well above 0.5 but not 1.0 - four supporting items against one contradicting item, mass-weighted by recency and reliability, not a simple 4-to-1 vote.
- `effective_support_count` accumulates from `source_reliability * decay` per supporting item, so the five-day-old and ten-day-old items contribute less than the near-zero-decay one-day-old item - recency matters, not just raw count.
- `V` reaches `1.0` because two genuinely independent source types (`recorded_event` and `explicit_user_statement`) are present, meeting `expected_source_types=2` for this belief's type - no need for unrelated context padding.
- The final `confidence` is capped below `0.98` regardless of how strong the evidence looks, because `support_factor` from a handful of items never reaches `1.0`.
- Status determination (via `determine_status`) is a separate step from the raw `confidence` number, using the belief-type's registered thresholds - never inlined as a magic number comparison.

# Policy Checklist - generate-synthetic-user

Canonical source: Blueprint section 13 "Three Initial Synthetic Evaluation Personas" and section 6.2 "Cold-Start Behavior".

## The three canonical personas (section 13)

| Persona | Hidden pattern | What the system must demonstrate |
|---|---|---|
| A - Schedule shift | Evening exercise historically works; a new job later makes mornings successful. | Learn the initial pattern, then lower confidence and adapt when it changes. |
| B - Intention vs. behavior | User says they want nightly study; actual successful study repeatedly occurs Saturday mornings. | Distinguish intention from observed successful habit without dismissing the stated goal. |
| C - Temporary disruption | Normally consistent user becomes inconsistent during travel, poor sleep, and new workload. | Infer state/context rather than assigning a permanent negative characteristic. |

## Cold-start bands to exercise (section 6.2)

| Event count | What the dataset should let a scorer check |
|---|---|
| 0-5 | Only generic personalization should be justified; no belief should reach beyond `candidate`. |
| 6-15 | Provisional beliefs become possible once a pattern repeats and is traceable. |
| 16-30 | Context-specific personalization becomes possible once thresholds are met. |
| 30+ | Event count alone must not be enough - include some noise/contradiction so quality/diversity/recency still gate confidence. |

## What this skill must never do

- Never generate a dataset with no documented hidden pattern - an evaluator cannot score against an unstated ground truth.
- Never let a single one-off event look identical in shape to a genuinely repeated pattern (see `examples/adversarial/example.md`) - vary timing/wording enough that overfitting a naive extractor would be visible.
- Never use a `user_id` that could be confused with a real user.

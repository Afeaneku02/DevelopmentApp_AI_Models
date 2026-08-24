---
name: evaluate-user-model
description: PLANNED, NOT YET IMPLEMENTED. This is a placeholder for a future Better You skill (blueprint section 10.2) that will compare generated observations/beliefs against golden expectations and report unsupported inference, traceability, adaptation, and stability. Do not select this skill for any current task; it has no procedure yet. Until it exists, use generate-synthetic-user for fixtures and evaluate results by hand against blueprint section 14's metrics.
---

# Evaluate User Model (Planned - Not Yet Implemented)

This skill does not exist yet. This file is a placeholder so that other skills (for example `generate-synthetic-user`, which names this skill as the consumer of its output) can reference it without pointing at a broken path.

## What this skill will do, once built

Per the skill responsibility table in the blueprint's Development Agent Skills section: compare generated observations/beliefs against golden expectations, reporting unsupported-inference rate, evidence traceability, contradiction responsiveness, temporal adaptation, and profile stability. It will need to respect the evaluation metrics, contract tests, and golden fixtures described in section 14 "Evaluation Metrics & Initial Targets" and the Evaluation Harness component (section 4.G, section 12.7).

## When to build it

Per section 10.2: add this skill once its corresponding runtime module enters the roadmap - that is Phase 8, "Evaluation hardening", in the Master Build Roadmap (section 11), once golden labeled cases exist to evaluate against.

## What to do in the meantime

For any task that needs to judge belief-engine output quality before this skill exists:

1. Use `generate-synthetic-user` to produce a persona dataset with a documented hidden pattern.
2. Read blueprint section 14 directly for the specific metrics and initial targets (evidence traceability, unsupported inference rate, contradiction responsiveness, temporal adaptation, stability, correction compliance).
3. Check results against those targets by hand, and record findings as a golden fixture candidate for when this skill exists to automate the comparison.

## Definition of done for this stub

This file should be replaced with a real `SKILL.md` (with the standard `references/`, `examples/`, and `scripts/` layout used by the other five skills) once Phase 8 evaluation-hardening work begins - not extended in place with implementation details before then.

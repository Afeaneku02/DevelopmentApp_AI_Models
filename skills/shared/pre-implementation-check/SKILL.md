---
name: pre-implementation-check
description: PLANNED, NOT YET IMPLEMENTED. This is a placeholder for a future Better You skill (blueprint section 10.2) that will identify, before coding, which blueprint contracts, schemas, policies, tests, and migrations a requested change touches. Do not select this skill for any current task; it has no procedure yet. Until it exists, read the relevant blueprint sections directly before coding and use contract-review after.
---

# Pre-Implementation Check (Planned - Not Yet Implemented)

This skill does not exist yet. This file is a placeholder so that other skills (for example `contract-review`, which names a `pre-implementation-check` -> implement -> `contract-review` loop) can reference it without pointing at a broken path.

## What this skill will do, once built

Per the skill responsibility table in the blueprint's Development Agent Skills section: before coding, identify which blueprint contracts, schemas, policies, tests, and migrations a requested change touches, producing a blueprint impact map, an affected-modules list, and a required-tests list.

This is the "before" half of the workflow that `contract-review` (the "after" half) already implements; the intent is a matched pair.

## When to build it

The blueprint's initial-five list (section 10.2) deliberately does not include this skill in the first batch, even though it is a general-purpose process skill rather than one tied to a specific runtime module. Build it once repeated manual pre-implementation analysis (of the kind `contract-review` currently has to do retroactively) becomes a noticeable cost - in practice, once real feature work on the Belief Engine or Recommendation Engine is underway (Phase 3 onward in the Master Build Roadmap, section 11).

## What to do in the meantime

Before implementing any change to the Better You adaptive user model:

1. Read the specific blueprint sections the change touches directly (via the document reader), rather than relying on memory of them.
2. Check `skills/shared/contract-review/references/schema.md` for the closed-contract checklist relevant to the area being changed.
3. Implement the change using the appropriate task skill (`extract-user-observations`, `create-belief-evidence`, `update-user-beliefs`, `generate-synthetic-user`).
4. Run `contract-review` afterward as the completion gate.

## Definition of done for this stub

This file should be replaced with a real `SKILL.md` (with the standard `references/`, `examples/`, and `scripts/` layout used by the other five skills) once repeated manual pre-implementation analysis justifies formalizing it - not extended in place with implementation details before then.

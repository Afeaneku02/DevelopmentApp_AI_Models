---
name: recompute-user-model
description: PLANNED, NOT YET IMPLEMENTED. This is a placeholder for a future Better You skill (blueprint section 10.2) that will safely rebuild active belief state after deletion, reset, suppression, or policy invalidation. Do not select this skill for any current task; it has no procedure yet. Until it exists, use update-user-beliefs for the underlying scoring function and apply the fail-closed rules in blueprint sections 6.0.1-6.0.2 by hand.
---

# Recompute User Model (Planned - Not Yet Implemented)

This skill does not exist yet. This file is a placeholder so that other skills (for example `update-user-beliefs`, `contract-review`) can reference it without pointing at a broken path.

## What this skill will do, once built

Per the skill responsibility table in the blueprint's Development Agent Skills section: safely rebuild active belief state after deletion, reset, suppression, policy invalidation, or scoring-version changes. It will need to respect the recompute state machine, the `D=null` branches, `locked_until_recompute`, and active-evidence filters described in sections 6.0.1 "Evidence Invalidation & Mandatory Recompute" and 6.0.2 "Failed Recompute - Fail Closed".

This is distinct from `update-user-beliefs`, which is already built: `update-user-beliefs` (specifically `scripts/compute_confidence.py`) is the deterministic scoring function itself. `recompute-user-model` will be the surrounding workflow that decides *when* a recompute must run, tracks `recompute_attempt_id`/`recompute_error`/`recompute_failed_at`, and enforces that a failed recompute leaves the belief locked rather than serving a stale cache.

## When to build it

Per section 10.2: add this skill once its corresponding runtime module enters the roadmap - that is Phase 3, "Belief engine", specifically the deletion/reset/invalidation and fail-closed retry state work called out in the Master Build Roadmap (section 11).

## What to do in the meantime

For any task touching deletion/reset-triggered recomputation before this skill exists:

1. Use `update-user-beliefs` for the confidence/status math itself.
2. Read blueprint sections 6.0.1-6.0.2 directly for the invalidation and fail-closed rules, and apply them by hand.
3. Run `contract-review` afterward - it already checks branch order and fail-closed behavior even without a dedicated `recompute-user-model` skill.

## Definition of done for this stub

This file should be replaced with a real `SKILL.md` (with the standard `references/`, `examples/`, and `scripts/` layout used by the other five skills) once Phase 3's deletion/reset/invalidation work begins - not extended in place with implementation details before then.

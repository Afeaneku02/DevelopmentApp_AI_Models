---
name: apply-risk-policy
description: PLANNED, NOT YET IMPLEMENTED. This is a placeholder for a future Better You skill (blueprint section 10.2) that will resolve recommendation risk tier and exploration/manual-review eligibility. Do not select this skill for any current task; it has no procedure yet. Until it exists, apply section 6.4 of the blueprint directly and flag risk-policy work for contract-review.
---

# Apply Risk Policy (Planned - Not Yet Implemented)

This skill does not exist yet. This file is a placeholder so that other skills (for example `contract-review`) can reference it without pointing at a broken path, and so no agent mistakes its absence for the risk-policy rules themselves being unimplemented in the blueprint.

## What this skill will do, once built

Per the skill responsibility table in the blueprint's Development Agent Skills section: resolve exact-context -> domain -> global-fallback recommendation risk policy, and enforce exploration/manual-review eligibility rules. It will need to respect `recommendation_context_policy`, `risk_domain_policy`, and the manual-review resolution rules in section 6.4 "Recommendation Risk Tiers & Exploration Policy".

## When to build it

Per section 10.2: add this skill once its corresponding runtime module enters the roadmap - that is Phase 6, "Recommendation loop", in the Master Build Roadmap (section 11/12.5). Do not build it earlier than that; a skill with no corresponding runtime code to validate against would have no way to earn its required regression fixture (section 10.4).

## What to do in the meantime

For any task touching recommendation risk tier, exploration, or manual-review resolution before this skill exists:

1. Read blueprint section 6.4 directly (via the document reader: `python tools/document_reader/read_document.py "Blueprint/Better_You_Adaptive_User_Model_RnD_Blueprint_v0.6.2.docx" --format json`, then locate the section 6.4 heading).
2. Apply its rules by hand rather than guessing.
3. Run `contract-review` afterward - it already checks the risk-tier/authorization contracts in section 6.4 even without a dedicated `apply-risk-policy` skill.

## Definition of done for this stub

This file should be replaced with a real `SKILL.md` (with the standard `references/`, `examples/`, and where useful `scripts/` layout used by the other five skills) once Phase 6 work begins - not extended in place with implementation details before then.

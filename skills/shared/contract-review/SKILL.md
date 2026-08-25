---
name: contract-review
description: This skill should be used after a code change touching the Better You adaptive user model - before treating the change as complete - to verify enums, schemas, versioned scoring/policy configuration, authorization boundaries, and version fields still match the blueprint. It should not be used to author new features itself; it only reviews and flags contract violations in changes already made.
---

# Contract Review

After a code change, verify that enums, schemas, versioned configuration, authorization boundaries, and version fields still match the blueprint's frozen contracts - and that regression tests/examples were updated to match. This skill is a gate, not an implementation step.

## When to use this skill

Use this skill after any change to code, schema, or configuration under the Better You adaptive user model, and before calling that change done. In particular:

- After implementing or modifying anything in the Behavior Interpretation Model, Belief Engine, Profile Builder, Recommendation Engine, or Outcome/Learning Engine.
- Before merging a PR that touches `user_events`, `user_observations`, `belief_evidence`, `user_beliefs`, `recommendations`, `scoring_config`, `belief_type_registry`, `recommendation_context_policy`, or `risk_domain_policy`.
- As the last step of the `pre-implementation-check` -> implement -> `contract-review` loop for any blueprint-governed change, once `pre-implementation-check` exists (see below).

Do not use this skill to design or implement the change itself - it reviews what already exists and flags violations. Once it exists, use `pre-implementation-check` before writing code; today, read the relevant blueprint sections directly instead (see `skills/shared/pre-implementation-check/SKILL.md`, which is a planned, not-yet-implemented skill). Use the specific task skill (`extract-user-observations`, `create-belief-evidence`, `update-user-beliefs`, etc.) while writing the change itself.

## Canonical source of truth

```bash
python tools/document_reader/read_document.py "Blueprint/Better_You_Adaptive_User_Model_RnD_Blueprint_v0.6.2.docx" --format json
```

The single most important section for this skill is section 18 "Definition of Done" - nearly every line item there is a contract this skill checks. Also check, by heading text, whichever of these are relevant to the change under review:

- section 3.4.3 "Versioned Scoring Configuration" and section 3.4.4 "Canonical belief_type Registry" - enum/config contracts.
- section 6.0.1-6.0.2, section 6.1.1-6.1.2 - evidence invalidation, deduplication, and aggregation authorization contracts.
- section 6.4 "Recommendation Risk Tiers & Exploration Policy" - risk/authorization contracts.
- section 14.1 "Required v0.6.1/v0.6.2 Contract Tests" - the specific regression tests a change must not silently break.
- section 22 "Decision Log" - standing decisions a change must not contradict without an explicit, documented reason.

`references/schema.md` lists every closed enum and required version-field set in one place for quick recall. `scripts/validate_output.py` runs the deterministic parts of this check (enum membership, required version fields) against a JSON record. For belief_evidence, pick the record type deliberately: `evidence` validates a pre-authorization proposal and fails if it carries any backend-owned field at all; `evidence_record` validates an already-authorized, persisted row and instead checks its authorization metadata is internally consistent. Running `evidence` against a backend-authorized record will always fail - that is a misuse of the check, not a real violation.

## Procedure

1. **Identify which contracts the change actually touches.** Do not run a generic checklist blindly - a PR touching only `generate-synthetic-user` output does not need a recommendation-risk review, but does need the `user_event` schema check.
2. **Check every closed enum is still closed.** `belief_type` (nine values + registry version), `source_type` (seven values), belief `status` (six values), recommendation `risk_tier` (low/medium/high), `review_status`. A change must never silently accept an eighth `belief_type` or an unlisted `source_type` string.
3. **Check the model-proposes / backend-authorizes boundary is intact.** Nothing an LLM or extractor produces should be able to set `authorized_aggregation_mode`, `review_status: approved`, `risk_tier`, or persist a `sensitive_or_high_impact_inference` belief directly. If a change makes any of these settable from model output, that is a blocking finding.
4. **Check every persisted record still carries its version fields** (`schema_version`, `scoring_version`, `canonicalizer_version`, `policy_version`, plus `risk_policy_version`/`risk_domain_policy_version` on recommendations) - a change that drops one silently breaks reproducibility.
5. **Check confidence/status/eligibility separation held.** A change must never let numeric confidence alone bypass `sensitivity_class`, `disallowed_contexts`, `persistence_policy`, staleness, an explicit user correction, or a risk-policy gate.
6. **Check the branch order on invalidation/no-evidence logic**, if the change touches scoring - the invalidation/no-active-evidence branch must still run before the generic no-evidence branch (section 6.0.2, section 14.1).
7. **Check tests and examples were updated, not just code.** A contract change without an updated regression fixture is incomplete per section 10.4 ("a skill is not complete until at least one regression fixture proves it prevents the failure mode") and per section 18's own requirement for regression coverage.
8. **Run `scripts/validate_output.py`** against representative output records from the change (a belief, an `evidence` proposal and/or an `evidence_record`, a recommendation) as the deterministic part of the review; do the authorization-boundary and test-coverage checks above by reading the diff.

## Completion check

A review is complete only when it produces one of two outcomes, stated explicitly:

- **Pass**: every touched contract still holds, version fields are intact, the model-proposes/backend-authorizes boundary is unbroken, and at least one regression fixture covers the change.
- **Fail**: a specific list of violated contracts, each citing the exact blueprint section it violates, and what must change to pass.

Never report "looks fine" without checking the specific contracts the change actually touches.

## Examples

`examples/positive/example.md` shows a passing review of a small belief-engine change. `examples/adversarial/example.md` shows a change that looks reasonable on the surface but silently lets model output set `authorized_aggregation_mode`, and how contract-review should catch it.

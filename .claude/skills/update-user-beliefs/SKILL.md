---
name: update-user-beliefs
description: This skill should be used when a coding agent needs to recompute belief confidence, effective evidence counts, and lifecycle status from the belief_evidence ledger, or when implementing/reviewing the deterministic scoring function itself. It should not be used to propose new evidence or decide recommendation risk policy.
---

# Update User Beliefs

This is the Claude Code discovery entry point. The full, authoritative procedure, formulas, policy rules, reference implementation (`scripts/compute_confidence.py`), and examples live in the tool-neutral shared copy - read it in full before acting; do not rely on this pointer alone:

```
skills/shared/update-user-beliefs/SKILL.md
```

If this file and the shared copy ever disagree, the shared copy at `skills/shared/update-user-beliefs/` is the source of truth. Do not duplicate its content here - update the shared copy instead so Claude Code and Codex never see diverging instructions.

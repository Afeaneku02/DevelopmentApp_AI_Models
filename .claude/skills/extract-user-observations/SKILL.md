---
name: extract-user-observations
description: This skill should be used when a coding agent needs to convert Better You user_events into structured, evidence-linked user_observation records, or when reviewing code/PRs that touch the Behavior Interpretation Model. It should not be used to decide belief confidence or status.
---

# Extract User Observations

This is the Claude Code discovery entry point. The full, authoritative procedure, schema checklist, policy rules, and examples live in the tool-neutral shared copy - read it in full before acting; do not rely on this pointer alone:

```
skills/shared/extract-user-observations/SKILL.md
```

If this file and the shared copy ever disagree, the shared copy at `skills/shared/extract-user-observations/` is the source of truth. Do not duplicate its content here - update the shared copy instead so Claude Code and Codex never see diverging instructions.

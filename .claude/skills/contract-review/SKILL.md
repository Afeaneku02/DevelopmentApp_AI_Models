---
name: contract-review
description: This skill should be used after a code change touching the Better You adaptive user model, before treating it as complete, to verify enums, schemas, versioned config, authorization boundaries, and version fields still match the blueprint. It should not be used to author new features.
---

# Contract Review

This is the Claude Code discovery entry point. The full, authoritative review procedure, closed-contract checklist, and deterministic validator (`scripts/validate_output.py`) live in the tool-neutral shared copy - read it in full before acting; do not rely on this pointer alone:

```
skills/shared/contract-review/SKILL.md
```

If this file and the shared copy ever disagree, the shared copy at `skills/shared/contract-review/` is the source of truth. Do not duplicate its content here - update the shared copy instead so Claude Code and Codex never see diverging instructions.

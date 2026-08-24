---
name: generate-synthetic-user
description: This skill should be used when a coding agent needs to create synthetic chronological user_event datasets with hidden ground-truth patterns for the Better You evaluation harness, including the three canonical personas (schedule shift, intention vs behavior, temporary disruption). It should not be used to write evaluation/scoring logic or to fabricate real user data.
---

# Generate Synthetic User

This is the Claude Code discovery entry point. The full, authoritative procedure, the three canonical persona definitions, and a working generator script (`scripts/generate_events.py`) live in the tool-neutral shared copy - read it in full before acting; do not rely on this pointer alone:

```
skills/shared/generate-synthetic-user/SKILL.md
```

If this file and the shared copy ever disagree, the shared copy at `skills/shared/generate-synthetic-user/` is the source of truth. Do not duplicate its content here - update the shared copy instead so Claude Code and Codex never see diverging instructions.

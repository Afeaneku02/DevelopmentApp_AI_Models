"""Bridge to the update-user-beliefs skill's reference confidence algorithm.

Per the task instruction to use the existing shared skills rather than
re-deriving blueprint section 3.4.1's formula a second time: this module
imports the already-built, already-self-tested implementation directly from
``skills/shared/update-user-beliefs/scripts/compute_confidence.py`` instead of
duplicating it here. That script is dependency-free by design specifically so
it can be imported like this.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SKILL_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "shared" / "update-user-beliefs" / "scripts"
)
if str(_SKILL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS_DIR))

from compute_confidence import (  # type: ignore  # noqa: E402
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    EvidenceItem,
    compute_confidence,
    determine_status,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "EvidenceItem",
    "compute_confidence",
    "determine_status",
]

"""user_events: ground truth about what occurred (blueprint section 5).

This is the only append-only, immutable source-of-truth table in the whole
model; everything downstream (observations, evidence, beliefs) must trace
back to rows here through explicit provenance, never invent facts a
UserEvent does not contain.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from src.common.versioning import VersionedModel


class UserEvent(VersionedModel):
    event_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    timestamp: datetime
    raw_content: str | None = None
    structured_data: dict[str, Any] | None = None
    source: str = Field(min_length=1)
    goal_id: str | None = None
    session_id: str | None = None

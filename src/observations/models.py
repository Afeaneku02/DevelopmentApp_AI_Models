"""user_observations and observation_events (blueprint section 5).

observation_events is the join table that replaces the ambiguous
event_id/event_ids representation the blueprint explicitly called out as a
past mistake (section 5, section 18 Definition of Done: "Observation-to-event
linkage uses observation_events consistently; no event_id/event_ids schema
mismatch remains"). UserObservation therefore deliberately has no event_id or
event_ids field at all -- provenance only ever goes through ObservationEvent
rows, enforced structurally by ``extra="forbid"`` on both models.

``importance``/``confidence`` use ``strict=True`` so Pydantic's default lax
coercion cannot silently accept ``True``/``False`` (bool is a subclass of int
in Python, so lax mode would otherwise coerce it to 1.0/0.0) or a numeric
string like ``"0.5"`` as a real JSON number. Strict mode still accepts a
plain JSON int (e.g. ``1``), only real numbers -- never bools or strings.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.common.enums import LinkRole
from src.common.versioning import VersionedModel


class UserObservation(VersionedModel):
    observation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    importance: float = Field(ge=0.0, le=1.0, strict=True)
    confidence: float = Field(ge=0.0, le=1.0, strict=True)
    created_at: datetime


class ObservationEvent(VersionedModel):
    observation_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    link_role: LinkRole
    created_at: datetime

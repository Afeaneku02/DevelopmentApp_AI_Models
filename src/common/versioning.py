"""Shared version-field contract for every persisted Better You record.

Blueprint section 18 (Definition of Done): "All major persisted records carry
schema_version, scoring_version, canonicalizer_version, and policy_version."
Making this a required base class, rather than a convention every model
remembers to repeat, is the deterministic enforcement mechanism: a record
missing any of these four fields fails Pydantic validation at construction
time rather than passing review by accident.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VersionedModel(BaseModel):
    """Base class for any record that must carry the four-field version contract.

    ``extra="forbid"`` is deliberate, not just strict-by-default: it is the
    structural half of the model-proposes/backend-authorizes boundary
    (blueprint section 6.1.2, section 22 Decision Log). A payload that tries
    to smuggle in a field a given model does not declare -- for example a
    proposal payload carrying ``authorized_aggregation_mode`` -- fails
    construction instead of being silently accepted.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    scoring_version: str = Field(min_length=1)
    canonicalizer_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)

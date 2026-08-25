"""user_model_deletion_requests and user_model_reset_events (blueprint section 5,
section 6.0.1-6.0.2).

Both tables track the same fail-closed recompute state machine: a deletion or
reset invalidates active evidence, which mandates a recompute, and a failed
recompute must lock affected beliefs out of live use rather than let a stale
cache serve traffic (section 6.0.2 "Failed Recompute - Fail Closed"). That
shared trailing block is factored into ``RecomputeTrackingFields`` so both
tables carry it with identical field names -- the blueprint itself uses these
exact field names identically on both tables (unlike ``user_beliefs``, which
uses the differently-named ``last_recompute_attempt_id`` for the same idea;
that distinction is preserved deliberately in ``src/beliefs/models.py`` rather
than unified, since unifying it would silently diverge from the blueprint's
actual per-table field names).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.common.enums import DeletionRequestedScope, DeletionStatus, RecomputeStatus, ResetScope
from src.common.versioning import VersionedModel


class RecomputeTrackingFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scoring_recompute_status: RecomputeStatus
    recompute_attempt_id: str | None = None
    recompute_error: str | None = None
    recompute_failed_at: datetime | None = None
    locked_until_recompute: bool = False
    last_successful_recompute_at: datetime | None = None
    recomputed_at: datetime | None = None

    @property
    def is_fail_closed_locked(self) -> bool:
        """True whenever a recompute is not in a completed-successful state.

        Section 6.0.2: pending, running, or failed recompute after
        invalidation must keep affected beliefs locked from live
        profile/recommendation use -- only ``completed`` may unlock them, and
        even then only in combination with ``locked_until_recompute=False``
        having been explicitly cleared.
        """
        return self.locked_until_recompute or self.scoring_recompute_status != RecomputeStatus.COMPLETED


class UserModelDeletionRequest(VersionedModel, RecomputeTrackingFields):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    requested_scope: DeletionRequestedScope
    requested_at: datetime
    status: DeletionStatus
    completed_at: datetime | None = None
    request_reason: str | None = None
    affected_belief_ids: list[str] = Field(default_factory=list)


class UserModelResetEvent(VersionedModel, RecomputeTrackingFields):
    model_config = ConfigDict(extra="forbid")

    reset_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    reset_scope: ResetScope
    requested_at: datetime
    executed_at: datetime | None = None
    keys_or_contexts_json: dict[str, Any] | list[str] | None = None
    affected_belief_ids: list[str] = Field(default_factory=list)

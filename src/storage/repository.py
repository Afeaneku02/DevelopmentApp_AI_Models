"""Minimal local persistence/repository layer for the Phase 1-3 records.

A single SQLite-backed ``Repository`` class (using the stdlib ``sqlite3``
module -- no new dependency), storing each record type as JSON in a small
table keyed by whatever id/scope columns the required query patterns
actually need. JSON, not a fully normalized relational schema, because the
record shapes are already the single source of truth (the Pydantic models
in ``src/``); mapping every field onto its own column would just be a
second, driftable copy of that schema. Real columns exist only for what
must actually be queried on: ``user_id``, ``belief_id``, ``event_id``,
``observation_id``.

Two things are deliberately reused here rather than re-implemented at the
SQL level, because both already have a single, tested definition elsewhere
in ``src/`` and a second definition would risk drifting from it:

- "Active evidence" -- ``list_active_evidence()`` calls the same
  ``active_evidence()`` helper ``recompute_belief()`` itself uses, on rows
  already deserialized in Python, instead of a parallel
  ``WHERE is_active = 1`` that could disagree with it.
- "Valid provenance" -- ``insert_observation()``/``insert_evidence()`` call
  the same ``validate_observation_provenance()``/
  ``validate_belief_evidence_provenance()`` used by Phase 3's pipeline
  wrapper, against the actual ``UserEvent`` rows already stored, so
  wrong-user or unknown-event provenance fails at save time too -- not only
  in an earlier pipeline step a caller might have skipped.

Every insert also re-validates through the corresponding Pydantic model
(``Model.model_validate(instance.model_dump())``) before writing, and every
read re-validates through ``Model.model_validate_json(...)``. This
re-validation is the actual enforcement mechanism for "no backend-owned
fields accepted from proposal input": ``insert_evidence()`` is type-hinted
to take a ``BeliefEvidence``, but a type hint alone is not enforcement --
what actually stops a bare ``BeliefEvidenceProposal`` (or anything else
missing ``evidence_id``, ``independence_group``, or the aggregation-
authorization fields) is that it fails ``BeliefEvidence.model_validate(...)``
before a single byte is written.

Deliberately narrow: no ORM, no migrations, no connection pooling, no
recommendation-engine tables, no deletion-cascade execution. One
``Repository`` per SQLite connection; callers own the connection's lifetime.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from src.beliefs.models import BeliefEvidence, UserBelief, active_evidence, invalidate_evidence
from src.common.provenance import validate_belief_evidence_provenance, validate_observation_provenance
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_observations (
    observation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observation_events_observation
    ON observation_events (observation_id);

CREATE TABLE IF NOT EXISTS belief_evidence (
    evidence_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_belief_evidence_scope
    ON belief_evidence (user_id, belief_id);

CREATE TABLE IF NOT EXISTS user_beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_beliefs_scope
    ON user_beliefs (user_id, belief_id, id);
"""


class Repository:
    """A single SQLite connection's worth of storage for one adaptive-user-
    model dataset. Construct via ``Repository.in_memory()`` for tests or
    ``Repository.at_path(path)`` for an on-disk database; both run the same
    idempotent schema-creation step (``CREATE TABLE IF NOT EXISTS``).
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def in_memory(cls) -> "Repository":
        return cls(sqlite3.connect(":memory:"))

    @classmethod
    def at_path(cls, path: str) -> "Repository":
        return cls(sqlite3.connect(path))

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- events --

    def insert_event(self, event: UserEvent) -> None:
        validated = UserEvent.model_validate(event.model_dump())
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_events (event_id, user_id, data) VALUES (?, ?, ?)",
                (validated.event_id, validated.user_id, validated.model_dump_json()),
            )

    def get_event(self, event_id: str) -> UserEvent | None:
        row = self._conn.execute("SELECT data FROM user_events WHERE event_id = ?", (event_id,)).fetchone()
        return UserEvent.model_validate_json(row[0]) if row else None

    def _get_events(self, event_ids: list[str]) -> list[UserEvent]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        rows = self._conn.execute(
            f"SELECT data FROM user_events WHERE event_id IN ({placeholders})", event_ids
        ).fetchall()
        return [UserEvent.model_validate_json(row[0]) for row in rows]

    # ------------------------------------------------------- observations --

    def insert_observation(self, observation: UserObservation, links: list[ObservationEvent]) -> None:
        """Inserts the observation and its observation_events links as one
        atomic transaction. Raises ``ValueError`` -- writing nothing -- if
        ``validate_observation_provenance()`` finds a link to a missing
        event, a link to another user's event, or no primary link at all,
        checked against the actual ``UserEvent`` rows already stored.
        """
        validated_observation = UserObservation.model_validate(observation.model_dump())
        validated_links = [ObservationEvent.model_validate(link.model_dump()) for link in links]

        referenced_event_ids = sorted({link.event_id for link in validated_links})
        known_events = self._get_events(referenced_event_ids)
        errors = validate_observation_provenance([validated_observation], validated_links, known_events)
        if errors:
            raise ValueError(
                f"cannot insert observation {validated_observation.observation_id!r}: " + "; ".join(errors)
            )

        with self._conn:
            self._conn.execute(
                "INSERT INTO user_observations (observation_id, user_id, data) VALUES (?, ?, ?)",
                (
                    validated_observation.observation_id,
                    validated_observation.user_id,
                    validated_observation.model_dump_json(),
                ),
            )
            self._conn.executemany(
                "INSERT INTO observation_events (observation_id, event_id, data) VALUES (?, ?, ?)",
                [(link.observation_id, link.event_id, link.model_dump_json()) for link in validated_links],
            )

    def get_observation(self, observation_id: str) -> UserObservation | None:
        row = self._conn.execute(
            "SELECT data FROM user_observations WHERE observation_id = ?", (observation_id,)
        ).fetchone()
        return UserObservation.model_validate_json(row[0]) if row else None

    def list_observation_events(self, observation_id: str) -> list[ObservationEvent]:
        rows = self._conn.execute(
            "SELECT data FROM observation_events WHERE observation_id = ?", (observation_id,)
        ).fetchall()
        return [ObservationEvent.model_validate_json(row[0]) for row in rows]

    # ----------------------------------------------------------- evidence --

    def insert_evidence(self, evidence: BeliefEvidence) -> None:
        """Only ever accepts an already-authorized ``BeliefEvidence`` -- see
        the module docstring for why re-validation, not the parameter's type
        hint, is what actually enforces that. Raises ``ValueError`` --
        writing nothing -- if ``validate_belief_evidence_provenance()`` finds
        an unknown or wrong-user ``source_event_id``, checked against the
        actual ``UserEvent`` rows already stored.
        """
        validated = BeliefEvidence.model_validate(evidence.model_dump())

        known_events = self._get_events(validated.source_event_ids)
        errors = validate_belief_evidence_provenance([validated], known_events)
        if errors:
            raise ValueError(f"cannot insert evidence {validated.evidence_id!r}: " + "; ".join(errors))

        with self._conn:
            self._conn.execute(
                "INSERT INTO belief_evidence (evidence_id, user_id, belief_id, data) VALUES (?, ?, ?, ?)",
                (validated.evidence_id, validated.user_id, validated.belief_id, validated.model_dump_json()),
            )

    def get_evidence(self, evidence_id: str) -> BeliefEvidence | None:
        row = self._conn.execute(
            "SELECT data FROM belief_evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return BeliefEvidence.model_validate_json(row[0]) if row else None

    def list_evidence(self, *, user_id: str, belief_id: str) -> list[BeliefEvidence]:
        """All evidence for this user/belief, active and inactive alike --
        use ``list_active_evidence()`` for the filtered view most callers
        actually want (in particular, what ``recompute_belief()`` should
        see)."""
        rows = self._conn.execute(
            "SELECT data FROM belief_evidence WHERE user_id = ? AND belief_id = ?", (user_id, belief_id)
        ).fetchall()
        return [BeliefEvidence.model_validate_json(row[0]) for row in rows]

    def list_active_evidence(self, *, user_id: str, belief_id: str) -> list[BeliefEvidence]:
        """Active evidence = ``is_active=True and not is_duplicate_suppressed``
        (blueprint section 6.0.1), via the same ``active_evidence()`` helper
        used elsewhere -- see the module docstring for why this is not a
        second, SQL-level definition of "active."""
        return active_evidence(self.list_evidence(user_id=user_id, belief_id=belief_id))

    def mark_evidence_inactive(self, evidence_id: str, *, reason: str, invalidated_at: datetime) -> BeliefEvidence:
        """Loads the row, applies ``invalidate_evidence()`` (the same
        function used everywhere else in the pipeline), writes the updated
        row back, and returns it. Raises ``ValueError`` for an unknown
        ``evidence_id``.

        Also fail-closes the belief this evidence belongs to (blueprint
        section 6.0.2: "set locked_until_recompute=true... treat the live
        belief as outdated/unavailable"): see ``_lock_latest_belief()``.
        Without this, ``get_latest_belief()`` would keep serving whatever
        confidence was computed *before* this invalidation as if it were
        still current, right up until some caller happens to recompute and
        save a new one -- a real stale-cache window, not a hypothetical one.
        """
        current = self.get_evidence(evidence_id)
        if current is None:
            raise ValueError(f"cannot invalidate unknown evidence_id: {evidence_id!r}")
        updated = invalidate_evidence(current, reason=reason, invalidated_at=invalidated_at)
        with self._conn:
            self._conn.execute(
                "UPDATE belief_evidence SET data = ? WHERE evidence_id = ?",
                (updated.model_dump_json(), evidence_id),
            )
        self._lock_latest_belief(user_id=current.user_id, belief_id=current.belief_id)
        return updated

    def _lock_latest_belief(self, *, user_id: str, belief_id: str) -> None:
        """If a belief has been saved for this scope and is not already
        locked, appends a copy of it with ``locked_until_recompute=True`` --
        never mutates the existing row, preserving ``save_belief()``'s
        append-only history. If none has ever been saved, there is nothing
        to lock (a first recompute will naturally see the now-inactive
        evidence).

        Deliberately leaves ``confidence``/``status``/every other field on
        the locked copy unchanged from the prior belief: blueprint section
        6.0.2 says a locked belief's cached confidence/status become
        "non-authoritative while the lock is active," not that they must be
        rewritten at lock time -- forcing them to 0.0/outdated is what an
        actual recompute does (section 6.0.1's post-invalidation rule), and
        that is a separate step this method does not perform. A caller must
        check ``locked_until_recompute`` before trusting any field on a
        belief returned by ``get_latest_belief()``; the numbers ceasing to
        be trustworthy is exactly what the flag communicates.
        """
        latest = self.get_latest_belief(user_id=user_id, belief_id=belief_id)
        if latest is None or latest.locked_until_recompute:
            return
        self.save_belief(latest.model_copy(update={"locked_until_recompute": True}))

    # ------------------------------------------------------------ beliefs --

    def save_belief(self, belief: UserBelief) -> None:
        """Append-only: every recompute is a new row, never an in-place
        overwrite, so there is always a full history and "the latest one"
        (``get_latest_belief()``) is an explicit query rather than
        something a concurrent writer could silently clobber."""
        validated = UserBelief.model_validate(belief.model_dump())
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_beliefs (belief_id, user_id, data) VALUES (?, ?, ?)",
                (validated.belief_id, validated.user_id, validated.model_dump_json()),
            )

    def get_latest_belief(self, *, user_id: str, belief_id: str) -> UserBelief | None:
        """Returns the most recently saved belief for this scope, or
        ``None`` if none has ever been saved. This may be a belief with
        ``locked_until_recompute=True`` -- see ``_lock_latest_belief()`` --
        in which case its confidence/status must not be treated as current
        until a fresh, unlocked recompute has been saved on top of it.
        """
        row = self._conn.execute(
            "SELECT data FROM user_beliefs WHERE user_id = ? AND belief_id = ? ORDER BY id DESC LIMIT 1",
            (user_id, belief_id),
        ).fetchone()
        return UserBelief.model_validate_json(row[0]) if row else None

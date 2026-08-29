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
from pathlib import Path

from src.beliefs.models import (
    BeliefEvidence,
    UserBelief,
    active_evidence,
    find_duplicate_evidence,
    invalidate_evidence,
    suppress_duplicate_evidence,
)
from src.beliefs.canonicalization import BeliefKeyCanonicalization
from src.common.provenance import validate_belief_evidence_provenance, validate_observation_provenance
from src.events.models import UserEvent
from src.observations.models import ObservationEvent, UserObservation
from src.recommendations.models import RecommendationOutcome, UserRecommendation

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

CREATE TABLE IF NOT EXISTS belief_key_canonicalizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonicalization_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    belief_type TEXT NOT NULL,
    proposed_key TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_belief_key_canonicalizations_scope
    ON belief_key_canonicalizations (user_id, belief_type, proposed_key, id);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    recommendation_context TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recommendations_scope
    ON recommendations (user_id, recommendation_context, id);

CREATE TABLE IF NOT EXISTS recommendation_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outcome_id TEXT NOT NULL UNIQUE,
    recommendation_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recommendation_outcomes_scope
    ON recommendation_outcomes (recommendation_id, id);
"""


class Repository:
    """A single SQLite connection's worth of storage for one adaptive-user-
    model dataset. Construct via ``Repository.in_memory()`` for tests,
    ``Repository.at_path(path)`` for a writable on-disk database (both run
    the same idempotent schema-creation step, ``CREATE TABLE IF NOT
    EXISTS``), or ``Repository.readonly_at_path(path)`` for a strictly
    read-only view of an existing on-disk database (no schema creation, no
    writes possible at the SQLite connection level -- see that
    constructor's own docstring).
    """

    def __init__(self, connection: sqlite3.Connection, *, run_schema: bool = True) -> None:
        self._conn = connection
        if run_schema:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @classmethod
    def in_memory(cls) -> "Repository":
        return cls(sqlite3.connect(":memory:"))

    @classmethod
    def at_path(cls, path: str) -> "Repository":
        return cls(sqlite3.connect(path))

    @classmethod
    def readonly_at_path(cls, path: str) -> "Repository":
        """Opens an existing on-disk database strictly read-only, via
        sqlite3's own URI ``mode=ro`` -- for a caller (``tools/
        inspect_user_model.py``) that must never create a database file or
        schema that doesn't already exist, and must never write to one that
        does, at the SQLite connection level rather than merely by
        convention (an accidental ``INSERT``/``UPDATE`` call against the
        returned ``Repository`` fails with ``sqlite3.OperationalError:
        attempt to write a readonly database`` instead of silently
        succeeding).

        Unlike ``at_path()``, this skips the schema-creation step entirely:
        running ``CREATE TABLE IF NOT EXISTS`` against a read-only
        connection would itself raise (SQLite still attempts the write even
        when the table already exists), and there is no writable connection
        here to create a schema with in the first place.

        Raises ``FileNotFoundError`` with a clear, specific message if
        ``path`` does not name an existing file, checked explicitly before
        SQLite ever opens the URI connection: SQLite's own failure for a
        missing file under ``mode=ro`` is an opaque
        ``sqlite3.OperationalError: unable to open database file``, which
        does not by itself distinguish "no such file" (most likely a typoed
        path) from other open failures -- checking first here gives callers
        a specific, actionable error instead of one they would have to
        parse SQLite's message to interpret. This is also what stops this
        method from ever creating the file: SQLite's ``mode=ro`` alone would
        already refuse to create one, but failing before SQLite is even
        invoked makes that guarantee explicit here rather than incidental to
        a flag on the URI.
        """
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"no such database file: {path!r}")
        uri = f"file:{resolved.resolve().as_posix()}?mode=ro"
        return cls(sqlite3.connect(uri, uri=True), run_schema=False)

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

    def list_events(self, *, user_id: str | None = None) -> list[UserEvent]:
        """Every stored event, optionally scoped to one user. Read-only --
        for inspection (``tools/inspect_user_model.py``), not for any
        pipeline step, which already knows exactly which event_ids it needs
        via ``get_event()``/``_get_events()``."""
        if user_id is not None:
            rows = self._conn.execute(
                "SELECT data FROM user_events WHERE user_id = ?", (user_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT data FROM user_events").fetchall()
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

    def list_observations(self, *, user_id: str | None = None) -> list[UserObservation]:
        """Every stored observation, optionally scoped to one user. Read-only
        -- for inspection (``tools/inspect_user_model.py``)."""
        if user_id is not None:
            rows = self._conn.execute(
                "SELECT data FROM user_observations WHERE user_id = ?", (user_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT data FROM user_observations").fetchall()
        return [UserObservation.model_validate_json(row[0]) for row in rows]

    def list_observation_events_for(self, observation_ids: list[str]) -> list[ObservationEvent]:
        """Bulk form of ``list_observation_events()`` for a known set of
        observation_ids. Read-only -- for inspection
        (``tools/inspect_user_model.py``), which already has the observation
        list it wants links for."""
        if not observation_ids:
            return []
        placeholders = ",".join("?" for _ in observation_ids)
        rows = self._conn.execute(
            f"SELECT data FROM observation_events WHERE observation_id IN ({placeholders})", observation_ids
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

    def list_all_evidence(
        self, *, user_id: str | None = None, belief_id: str | None = None
    ) -> list[BeliefEvidence]:
        """Every stored evidence row (active and inactive alike) matching
        either or both optional filters. Unlike ``list_evidence()``, both
        filters are optional -- this is a general-purpose lister for
        inspection (``tools/inspect_user_model.py``), not a belief-scoped
        query a pipeline step already knows both ids for. Apply
        ``active_evidence()`` to the result for an active-only view, the same
        helper ``list_active_evidence()`` itself uses -- this method
        deliberately does not filter by activity so a caller can choose."""
        clauses: list[str] = []
        params: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if belief_id is not None:
            clauses.append("belief_id = ?")
            params.append(belief_id)
        query = "SELECT data FROM belief_evidence"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(query, params).fetchall()
        return [BeliefEvidence.model_validate_json(row[0]) for row in rows]

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

    def suppress_duplicate_evidence(self, *, user_id: str, belief_id: str, reason: str) -> list[BeliefEvidence]:
        """Finds and marks duplicate evidence rows for one (user_id,
        belief_id) scope via ``find_duplicate_evidence()``/
        ``suppress_duplicate_evidence()`` (the same backend-owned,
        non-destructive marking used everywhere else in this module -- see
        the module docstring for why "active" and "duplicate" are never
        redefined at the SQL level). Returns only the rows newly marked by
        this call, in the same transaction; a row ``find_duplicate_evidence()``
        would still flag but that is already marked is left untouched and
        excluded from the return value, so calling this again after a prior
        call (or after nothing changed) is a safe no-op rather than a second
        write.

        Also fail-closes the belief this evidence belongs to, exactly as
        ``mark_evidence_inactive()`` does: suppressing evidence changes what
        ``active_evidence()`` returns for this scope just as invalidating it
        does, so the same ``locked_until_recompute`` guarantee applies -- see
        ``_lock_latest_belief()``. Only runs when at least one row was
        actually newly suppressed, so a call that finds nothing to do never
        appends a redundant locked copy of an already-current belief.
        """
        all_evidence = self.list_evidence(user_id=user_id, belief_id=belief_id)
        duplicates = find_duplicate_evidence(all_evidence)

        newly_suppressed: list[BeliefEvidence] = []
        with self._conn:
            for row in duplicates:
                if row.is_duplicate_suppressed:
                    continue
                updated = suppress_duplicate_evidence(row, reason=reason)
                self._conn.execute(
                    "UPDATE belief_evidence SET data = ? WHERE evidence_id = ?",
                    (updated.model_dump_json(), updated.evidence_id),
                )
                newly_suppressed.append(updated)

        if newly_suppressed:
            self._lock_latest_belief(user_id=user_id, belief_id=belief_id)
        return newly_suppressed

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

    def list_latest_beliefs(
        self, *, user_id: str | None = None, belief_id: str | None = None
    ) -> list[UserBelief]:
        """The latest saved belief for every (user_id, belief_id) scope
        matching either or both optional filters -- the general-purpose,
        possibly-multi-scope form of ``get_latest_belief()`` for inspection
        (``tools/inspect_user_model.py``), which does not necessarily know a
        single scope up front. Each scope's own append-only history (see
        ``save_belief()``) is still resolved the same way ``get_latest_belief()``
        resolves it -- the most recently inserted row for that scope -- just
        for every matching scope at once instead of one named pair."""
        clauses: list[str] = []
        params: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if belief_id is not None:
            clauses.append("belief_id = ?")
            params.append(belief_id)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            "SELECT b.data FROM user_beliefs b "
            "INNER JOIN ("
            f"SELECT user_id, belief_id, MAX(id) AS max_id FROM user_beliefs{where_sql} "
            "GROUP BY user_id, belief_id"
            ") latest ON b.id = latest.max_id"
        )
        rows = self._conn.execute(query, params).fetchall()
        return [UserBelief.model_validate_json(row[0]) for row in rows]

    # ------------------------------------------------- belief-key canonicalization --

    def insert_belief_key_canonicalization(self, decision: BeliefKeyCanonicalization) -> None:
        """Append-only audit log insert -- every canonicalization decision
        ever authorized is kept, never overwritten or replaced, exactly like
        ``save_belief()``'s own append-only history. Re-validates through
        ``BeliefKeyCanonicalization`` first (the same "re-validation, not
        the type hint, is what enforces this" pattern as every other
        insert_*() method), so a bare ``BeliefKeyCanonicalizationProposal``
        (missing ``canonical_key``, ``decision``, or any other backend-owned
        field) fails before a single byte is written.

        ``canonicalization_id`` is ``UNIQUE`` at the schema level, raising
        ``sqlite3.IntegrityError`` for a repeat -- the same stable-unique-id
        contract every other externally supplied record id already has
        (``event_id``, ``observation_id``, ``evidence_id``). Re-resolving
        the same ``proposed_key`` is still fully allowed and expected (that
        is exactly what grows the audit trail; see
        ``get_latest_belief_key_canonicalization()``), it just requires a
        new ``canonicalization_id`` for each appended decision, same as a
        new evidence row needs its own ``evidence_id`` even when it is
        evidence for the same belief as an earlier row.
        """
        validated = BeliefKeyCanonicalization.model_validate(decision.model_dump())
        with self._conn:
            self._conn.execute(
                "INSERT INTO belief_key_canonicalizations "
                "(canonicalization_id, user_id, belief_type, proposed_key, data) VALUES (?, ?, ?, ?, ?)",
                (
                    validated.canonicalization_id, validated.user_id, validated.belief_type.value,
                    validated.proposed_key, validated.model_dump_json(),
                ),
            )

    def get_latest_belief_key_canonicalization(
        self, *, user_id: str, belief_type: str, proposed_key: str
    ) -> BeliefKeyCanonicalization | None:
        """The most recently authorized canonicalization decision for one
        exact (user_id, belief_type, proposed_key) scope, or ``None`` if
        this proposed_key has never been resolved before -- the same
        "most recently inserted row for this scope" resolution as
        ``get_latest_belief()``."""
        row = self._conn.execute(
            "SELECT data FROM belief_key_canonicalizations "
            "WHERE user_id = ? AND belief_type = ? AND proposed_key = ? "
            "ORDER BY id DESC LIMIT 1",
            (user_id, belief_type, proposed_key),
        ).fetchone()
        return BeliefKeyCanonicalization.model_validate_json(row[0]) if row else None

    def list_belief_key_canonicalizations(
        self, *, user_id: str | None = None, belief_type: str | None = None
    ) -> list[BeliefKeyCanonicalization]:
        """Every canonicalization decision ever authorized, optionally
        scoped by user and/or belief_type -- the full audit trail, not just
        the latest decision per key (see ``get_latest_belief_key_canonicalization()``
        for that)."""
        clauses: list[str] = []
        params: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if belief_type is not None:
            clauses.append("belief_type = ?")
            params.append(belief_type)
        query = "SELECT data FROM belief_key_canonicalizations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [BeliefKeyCanonicalization.model_validate_json(row[0]) for row in rows]

    # --------------------------------------------------- recommendations --

    def insert_recommendation(self, recommendation: UserRecommendation) -> None:
        """Append-only insert for one recommendation decision (blueprint
        section 5 "recommendations"). Re-validates through
        ``UserRecommendation`` first -- the same "re-validation, not the type
        hint, is what enforces this" pattern as every other insert_*() method
        -- so a record missing a version field, an inconsistent review gate,
        or an unfrozen decision state fails before a byte is written.

        ``recommendation_id`` is ``UNIQUE`` at the schema level, raising
        ``sqlite3.IntegrityError`` for a repeat, the same stable-unique-id
        contract as ``event_id`` / ``evidence_id`` / ``canonicalization_id``.
        This never recomputes a belief, reads live belief rows at read time,
        or issues a review decision -- the record's ``frozen_belief_state``
        is authoritative for what the recommendation saw.
        """
        validated = UserRecommendation.model_validate(recommendation.model_dump())
        with self._conn:
            self._conn.execute(
                "INSERT INTO recommendations "
                "(recommendation_id, user_id, recommendation_context, risk_tier, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    validated.recommendation_id,
                    validated.user_id,
                    validated.recommendation_context,
                    validated.risk_tier.value,
                    validated.model_dump_json(),
                ),
            )

    def get_recommendation(self, recommendation_id: str) -> UserRecommendation | None:
        row = self._conn.execute(
            "SELECT data FROM recommendations WHERE recommendation_id = ?", (recommendation_id,)
        ).fetchone()
        return UserRecommendation.model_validate_json(row[0]) if row else None

    def list_recommendations(
        self, *, user_id: str | None = None, recommendation_context: str | None = None
    ) -> list[UserRecommendation]:
        """Every stored recommendation, optionally scoped by user and/or
        resolved context, oldest first."""
        clauses: list[str] = []
        params: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if recommendation_context is not None:
            clauses.append("recommendation_context = ?")
            params.append(recommendation_context)
        query = "SELECT data FROM recommendations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [UserRecommendation.model_validate_json(row[0]) for row in rows]

    # -------------------------------------------- recommendation outcomes --

    def insert_recommendation_outcome(self, outcome: RecommendationOutcome) -> None:
        """Append-only insert for one recommendation outcome (blueprint
        section 5 "recommendation_outcomes"). Re-validates through
        ``RecommendationOutcome`` first, then -- writing nothing on failure
        -- checks that ``outcome.recommendation_id`` names a recommendation
        that actually exists in this database, the same fail-before-write
        provenance discipline as ``insert_observation()`` /
        ``insert_evidence()``.

        Many outcomes may reference one ``recommendation_id`` (a first
        report, a later follow-up, a measured signal arriving separately);
        only ``outcome_id`` is ``UNIQUE``, raising ``sqlite3.IntegrityError``
        for a repeat. This never touches the ``recommendations`` row it
        points at -- the recommendation's frozen decision state is
        immutable once written.
        """
        validated = RecommendationOutcome.model_validate(outcome.model_dump())
        if self.get_recommendation(validated.recommendation_id) is None:
            raise ValueError(
                f"cannot insert outcome {validated.outcome_id!r}: no recommendation "
                f"{validated.recommendation_id!r} exists in this database"
            )
        with self._conn:
            self._conn.execute(
                "INSERT INTO recommendation_outcomes (outcome_id, recommendation_id, data) "
                "VALUES (?, ?, ?)",
                (validated.outcome_id, validated.recommendation_id, validated.model_dump_json()),
            )

    def get_recommendation_outcome(self, outcome_id: str) -> RecommendationOutcome | None:
        row = self._conn.execute(
            "SELECT data FROM recommendation_outcomes WHERE outcome_id = ?", (outcome_id,)
        ).fetchone()
        return RecommendationOutcome.model_validate_json(row[0]) if row else None

    def list_recommendation_outcomes(
        self, *, recommendation_id: str | None = None
    ) -> list[RecommendationOutcome]:
        """Every stored outcome, optionally scoped to one recommendation,
        oldest first -- the full append-only trail for that recommendation."""
        if recommendation_id is not None:
            rows = self._conn.execute(
                "SELECT data FROM recommendation_outcomes WHERE recommendation_id = ? ORDER BY id ASC",
                (recommendation_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM recommendation_outcomes ORDER BY id ASC"
            ).fetchall()
        return [RecommendationOutcome.model_validate_json(row[0]) for row in rows]

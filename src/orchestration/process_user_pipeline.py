"""Local/internal orchestration: run one user through the whole existing
adaptive-user-model chain in a single call --

    unprocessed user_events -> user_observations -> belief_evidence
    -> belief recomputation

This module is a *coordinator*, not a third drafting mechanism: it never
decides what an event means or what a belief should say. Every actual
decision is still made by the two already-built, already-tested pipeline
steps this wraps --

- ``src.observations.process_events.process_user_events`` (events ->
  observations), and
- ``src.beliefs.process_observations.process_user_observations``
  (observations -> belief_evidence -> recompute),

reused exactly as they are, via the ``event_ids``/``observation_ids``
scoping parameters those functions already expose for this purpose. Nothing
about drafting, idempotency, provenance, or scoring is re-implemented here.

Why per-record, not one whole-user batch call per stage
---------------------------------------------------------
Both wrapped functions already treat *known* bad input gracefully (an
unsupported event_type, malformed structured_data, or a category with no
confident signal all become a reported ``skipped_*`` outcome, never a raised
exception). But neither function isolates an outright *unexpected* failure
(a corrupted row, a caller-supplied ``as_of`` that predates already-recorded
evidence, or any other bug) from the rest of its batch -- an exception
partway through one whole-user call aborts every record in that call,
including ones that had already been safely decided.

This module closes that gap the same way the rest of this codebase isolates
one unit of work from another: by giving each pipeline step exactly one
record at a time (``event_ids=[event.event_id]`` /
``observation_ids=[observation.observation_id]``) and wrapping each call in
its own ``try/except``. One record's unexpected failure is caught, counted,
and recorded in ``ProcessUserPipelineResult.failures`` -- every other record,
before or after it in the run, is processed exactly as if the failing one
were not there.

Fail-closed recomputation
--------------------------
The observation stage's call additionally runs inside ``repo.transaction()``
(the same primitive ``src.recommendations.review`` already uses to bind a
promotion to its audit row) around ``persist=True, recompute=True``: the new
``belief_evidence`` row and the belief recomputed from it commit together, or
-- if recomputation itself raises -- neither does. A belief's cached
confidence therefore never ends up reflecting evidence that a failed
recompute never accounted for, and it never needs a separate
``locked_until_recompute`` fallback here, because there is nothing partial
left behind to lock: recompute is mandatory for this operation (blueprint
section 6.0.2's own path for "evidence changed through a path other than
invalidation"), not the caller-opt-in default those two functions otherwise
expose.

User isolation
---------------
Every read this module does is scoped with ``user_id=user_id`` (via
``repo.list_events``/``repo.list_observations``), and both wrapped functions
independently re-check ownership of anything they touch. No other user's
records are ever read, written, or counted by one call.

Dry run == a real run against a disposable clone, not a second preview path
------------------------------------------------------------------------------
``persist=False`` does not ask either wrapped function to draft-without-
writing and stop there. On its own, that would only preview the events ->
observations half of the chain: neither wrapped function's ``persist=False``
mode writes anything, so an observation a dry run says it "would create" from
a not-yet-processed event was never actually inserted, and the observation ->
belief_evidence -> recompute stage that re-reads via
``repo.list_observations()`` immediately afterward would never see it --
silently truncating the preview to stage one for exactly the events a dry
run exists to show the full effect of.

Instead, ``persist=False`` clones ``repo``'s data into a disposable
in-memory database (``Repository.clone_in_memory()``, SQLite's own backup
API) and recurses into this same function with ``persist=True`` against
*that* clone -- a real run, through the same code path a persisted call
takes, including a stage-one observation this same run just wrote feeding
stage two and a stage-two evidence row this same run just wrote feeding
recomputation. Only the clone is written to; it is closed and discarded
before this function returns, and ``repo`` itself is never opened for
writing. The result is relabeled (``created`` -> ``would_create`` in
``event_outcomes``/``observation_outcomes``, ``persisted`` forced to
``False``) to keep the ``would_create``/``skipped_*`` vocabulary this
module's callers already read, but every count, every recomputed belief's
previewed confidence/status, and every failure it reports reflects what a
``persist=True`` call would actually do right now -- not a partial guess.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from src.beliefs.process_observations import (
    ObservationProcessingOutcome,
    process_user_observations,
)
from src.observations.process_events import EventProcessingOutcome, process_user_events

_EVENT_STAGE = "event"
_OBSERVATION_STAGE = "observation"

# Outcome actions that represent a genuinely new record (or, in a dry run,
# one that would be created) -- everything else is some flavor of "already
# handled" or "cannot safely handle," i.e. a skip.
_CREATED_ACTIONS = frozenset({"created", "would_create"})


@dataclass(frozen=True)
class RecordFailure:
    """One record (an event or an observation) whose processing raised
    during this run -- see the module docstring for why this is reserved for
    the unexpected: every ordinary "cannot safely process this" case is
    already a graceful ``skipped_*`` outcome from the wrapped functions
    themselves, not a ``RecordFailure``."""

    stage: str  # "event" | "observation"
    record_id: str
    error: str


@dataclass(frozen=True)
class ProcessUserPipelineResult:
    """What one ``process_user_full_pipeline`` run did for exactly one
    user. ``events_*``/``observations_*`` are per-stage detail;
    ``counts()`` is the flattened processed/created/skipped/failed/
    recomputed summary this operation is required to report."""

    user_id: str
    persisted: bool
    events_created: int
    events_skipped: int
    events_failed: int
    observations_created: int
    observations_skipped: int
    observations_failed: int
    beliefs_recomputed: int
    recomputed_belief_ids: list[str]
    failures: list[RecordFailure]
    event_outcomes: list[EventProcessingOutcome]
    observation_outcomes: list[ObservationProcessingOutcome]
    recomputed_beliefs: list[dict[str, Any]]

    @property
    def events_processed(self) -> int:
        return self.events_created + self.events_skipped + self.events_failed

    @property
    def observations_processed(self) -> int:
        return self.observations_created + self.observations_skipped + self.observations_failed

    def counts(self) -> dict[str, int]:
        """Processed/created/skipped/failed summed across both stages, plus
        ``recomputed`` -- the count of *distinct* beliefs successfully
        recomputed this run (a belief that received evidence from several
        observations in one run is recomputed once per contributing
        observation internally, but is only counted once here)."""
        return {
            "processed": self.events_processed + self.observations_processed,
            "created": self.events_created + self.observations_created,
            "skipped": self.events_skipped + self.observations_skipped,
            "failed": self.events_failed + self.observations_failed,
            "recomputed": self.beliefs_recomputed,
        }


def process_user_full_pipeline(
    repo: Any,
    *,
    user_id: str,
    as_of: datetime,
    persist: bool = True,
) -> ProcessUserPipelineResult:
    """Run the whole chain for ``user_id`` and return a structured result.

    ``persist=False`` is a dry run that accurately simulates the *complete*
    chain -- events -> observations -> belief_evidence -> recomputation,
    including a stage-one observation this same dry run would have just
    created feeding stage two -- without writing anything to ``repo``. See
    "Dry run == a real run against a disposable clone" in the module
    docstring for how (``Repository.clone_in_memory()`` plus a recursive
    ``persist=True`` call against the clone) and why a shallower
    "draft-without-writing" preview would silently miss exactly the records
    a dry run exists to show the full effect of.

    Idempotent: rerunning against unchanged data creates no duplicate
    observations or evidence (the wrapped functions' own idempotency, not
    re-implemented here) and reports zero newly created/recomputed records.

    Scoped to ``user_id`` only -- every event and observation considered is
    read via ``repo.list_events(user_id=user_id)`` /
    ``repo.list_observations(user_id=user_id)``, and never touches another
    user's records.

    Never raises for anything about an individual event or observation --
    see the module docstring for why an unexpected per-record failure is
    caught, counted in ``failures``, and does not stop the rest of the run.
    Still raises ``ValueError`` for a blank ``user_id`` up front, the same
    "fail before doing anything" contract as a malformed request elsewhere in
    this codebase.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-blank string")

    if not persist:
        return _simulate_dry_run(repo, user_id=user_id, as_of=as_of)

    event_outcomes: list[EventProcessingOutcome] = []
    failures: list[RecordFailure] = []
    events_created = events_skipped = events_failed = 0

    events = sorted(repo.list_events(user_id=user_id), key=lambda e: (e.timestamp, e.event_id))
    for event in events:
        try:
            step = process_user_events(
                repo, user_id=user_id, event_ids=[event.event_id], as_of=as_of, persist=persist,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad record, see module docstring
            failures.append(RecordFailure(_EVENT_STAGE, event.event_id, f"{type(exc).__name__}: {exc}"))
            events_failed += 1
            continue
        for outcome in step.outcomes:
            event_outcomes.append(outcome)
            if outcome.action in _CREATED_ACTIONS:
                events_created += 1
            else:
                events_skipped += 1

    observation_outcomes: list[ObservationProcessingOutcome] = []
    recomputed_beliefs: list[dict[str, Any]] = []
    observations_created = observations_skipped = observations_failed = 0

    # Re-read after the event stage: any observation just created above (when
    # persist=True) must be picked up by this stage in the same run, not left
    # for a second call.
    observations = sorted(
        repo.list_observations(user_id=user_id), key=lambda o: (o.created_at, o.observation_id)
    )
    for observation in observations:
        try:
            # Evidence-insert and its mandatory recompute commit together or
            # not at all -- see "Fail-closed recomputation" in the module
            # docstring. repo.transaction() is reentrant, so this nests
            # safely even if repo already has one open.
            with repo.transaction():
                step = process_user_observations(
                    repo, user_id=user_id, observation_ids=[observation.observation_id],
                    as_of=as_of, persist=persist, recompute=True,
                )
        except Exception as exc:  # noqa: BLE001 - isolate one bad record, see module docstring
            failures.append(
                RecordFailure(_OBSERVATION_STAGE, observation.observation_id, f"{type(exc).__name__}: {exc}")
            )
            observations_failed += 1
            continue
        for outcome in step.outcomes:
            observation_outcomes.append(outcome)
            if outcome.action in _CREATED_ACTIONS:
                observations_created += 1
            else:
                observations_skipped += 1
        recomputed_beliefs.extend(step.recomputed_beliefs)

    recomputed_belief_ids = sorted({belief["belief_id"] for belief in recomputed_beliefs})

    return ProcessUserPipelineResult(
        user_id=user_id,
        persisted=persist,
        events_created=events_created,
        events_skipped=events_skipped,
        events_failed=events_failed,
        observations_created=observations_created,
        observations_skipped=observations_skipped,
        observations_failed=observations_failed,
        beliefs_recomputed=len(recomputed_belief_ids),
        recomputed_belief_ids=recomputed_belief_ids,
        failures=failures,
        event_outcomes=event_outcomes,
        observation_outcomes=observation_outcomes,
        recomputed_beliefs=recomputed_beliefs,
    )


def _relabel_would_create(action: str) -> str:
    """``created`` (what a real ``persist=True`` run against the clone
    reports) -> ``would_create`` (what ``persist=False`` has always
    documented). Every other action -- every ``skipped_*`` reason, in
    particular -- is left alone: *why* a record can't be safely processed is
    identical whether or not the run that decided that actually persists
    anything, so relabeling those would misrepresent, not preserve, the
    dry-run contract."""
    return "would_create" if action == "created" else action


def _simulate_dry_run(repo: Any, *, user_id: str, as_of: datetime) -> ProcessUserPipelineResult:
    """The ``persist=False`` implementation: see "Dry run == a real run
    against a disposable clone" in the module docstring for why this clones
    ``repo`` and recurses into ``persist=True`` against the clone instead of
    calling the wrapped functions' own ``persist=False`` modes directly."""
    clone = repo.clone_in_memory()
    try:
        simulated = process_user_full_pipeline(clone, user_id=user_id, as_of=as_of, persist=True)
    finally:
        clone.close()

    return replace(
        simulated,
        persisted=False,
        event_outcomes=[
            replace(outcome, action=_relabel_would_create(outcome.action)) for outcome in simulated.event_outcomes
        ],
        observation_outcomes=[
            replace(outcome, action=_relabel_would_create(outcome.action))
            for outcome in simulated.observation_outcomes
        ],
    )

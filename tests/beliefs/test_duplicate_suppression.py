"""Tests for src/beliefs/models.py's duplicate-evidence suppression:
find_duplicate_evidence() and suppress_duplicate_evidence().

Backend-owned, non-destructive marking (blueprint section 6.0.1's
is_duplicate_suppressed/suppression_reason fields) to prevent confidence
inflation when multiple belief_evidence rows represent the same underlying
claim -- never an LLM/proposal-owned decision (BeliefEvidenceProposal has no
such fields at all; see test_belief_evidence_authorization.py's own coverage
of that boundary).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.beliefs.models import (
    BeliefEvidence,
    BeliefEvidenceProposal,
    active_evidence,
    authorize_evidence,
    find_duplicate_evidence,
    invalidate_evidence,
    suppress_duplicate_evidence,
)
from src.beliefs.recompute import recompute_belief
from src.common.enums import SourceType
from src.common.registry import SOURCE_TYPE_RELIABILITY

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

BASE_TIME = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


def _evidence(
    evidence_id: str, *, created_at: datetime, belief_id: str = "bel_1", user_id: str = "usr_1",
    direction: str = "support", source_type: str = "recorded_event", context_key: str = "fitness",
    source_event_ids: list[str] | None = None, is_active: bool = True,
) -> BeliefEvidence:
    event_ids = source_event_ids or ["evt_1"]
    proposal = BeliefEvidenceProposal(
        belief_id=belief_id, user_id=user_id, direction=direction, event_id=event_ids[0],
        source_event_ids=event_ids, source_type=source_type, context_key=context_key,
        strength=0.85, source_reliability=SOURCE_TYPE_RELIABILITY[SourceType(source_type)],
        observed_at=created_at, decay_lambda=0.015, model_version="test-model-0.1", **VERSION_FIELDS,
    )
    evidence = authorize_evidence(
        proposal, evidence_id=evidence_id, created_at=created_at,
        aggregation_policy_version="evidence-aggregation-0.6",
    )
    if not is_active:
        evidence = invalidate_evidence(evidence, reason="deletion", invalidated_at=created_at)
    return evidence


class ExactDuplicateTests(unittest.TestCase):
    def test_same_user_belief_direction_and_source_events_are_flagged_as_duplicates(self) -> None:
        earlier = _evidence("bev_1", created_at=BASE_TIME)
        later = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))

        duplicates = find_duplicate_evidence([earlier, later])

        self.assertEqual([row.evidence_id for row in duplicates], ["bev_2"])

    def test_three_way_exact_duplicate_keeps_only_the_earliest(self) -> None:
        first = _evidence("bev_1", created_at=BASE_TIME)
        second = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))
        third = _evidence("bev_3", created_at=BASE_TIME + timedelta(hours=2))

        duplicates = find_duplicate_evidence([third, first, second])  # order-independent

        self.assertEqual({row.evidence_id for row in duplicates}, {"bev_2", "bev_3"})

    def test_tied_created_at_breaks_ties_by_evidence_id(self) -> None:
        a = _evidence("bev_a", created_at=BASE_TIME)
        b = _evidence("bev_b", created_at=BASE_TIME)  # identical timestamp

        duplicates = find_duplicate_evidence([b, a])

        # "bev_a" sorts before "bev_b" -- deterministic regardless of input order.
        self.assertEqual([row.evidence_id for row in duplicates], ["bev_b"])


class DifferentBeliefNotSuppressedTests(unittest.TestCase):
    def test_same_source_event_but_different_belief_id_is_not_suppressed(self) -> None:
        for_bel_1 = _evidence("bev_1", created_at=BASE_TIME, belief_id="bel_1")
        for_bel_2 = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1), belief_id="bel_2")

        duplicates = find_duplicate_evidence([for_bel_1, for_bel_2])

        self.assertEqual(duplicates, [])


class OppositeDirectionNotSuppressedTests(unittest.TestCase):
    def test_same_belief_but_opposite_direction_is_not_suppressed(self) -> None:
        supporting = _evidence("bev_1", created_at=BASE_TIME, direction="support")
        contradicting = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1), direction="contradict")

        duplicates = find_duplicate_evidence([supporting, contradicting])

        self.assertEqual(duplicates, [])


class DifferentSourceTypeOrContextKeyNotSuppressedTests(unittest.TestCase):
    def test_different_source_type_is_not_suppressed(self) -> None:
        recorded = _evidence("bev_1", created_at=BASE_TIME, source_type="recorded_event")
        stated = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1), source_type="explicit_user_statement")

        self.assertEqual(find_duplicate_evidence([recorded, stated]), [])

    def test_different_context_key_is_not_suppressed(self) -> None:
        fitness = _evidence("bev_1", created_at=BASE_TIME, context_key="fitness")
        nutrition = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1), context_key="nutrition")

        self.assertEqual(find_duplicate_evidence([fitness, nutrition]), [])


class AlreadyInactiveNotConsideredTests(unittest.TestCase):
    def test_an_already_invalidated_row_is_not_returned_as_a_duplicate(self) -> None:
        active_row = _evidence("bev_1", created_at=BASE_TIME)
        inactive_row = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1), is_active=False)

        # Only one is_active row remains, so there is no group of size >= 2
        # to suppress from -- an inactive duplicate needs no suppression
        # decision, it is already excluded from active_evidence().
        self.assertEqual(find_duplicate_evidence([active_row, inactive_row]), [])


class AlreadySuppressedRowIsNotAKeeperTests(unittest.TestCase):
    """Regression test: an already-suppressed row must never be treated as
    a duplicate group's canonical "keeper" just because its created_at
    happens to be earliest -- that would strip the group down to zero
    scoring-eligible rows by suppressing the one row that was never
    suppressed in the first place, exactly the confidence corruption this
    module exists to prevent."""

    def test_already_suppressed_earlier_row_does_not_cause_the_unsuppressed_later_row_to_be_suppressed(
        self,
    ) -> None:
        already_suppressed = _evidence("bev_1", created_at=BASE_TIME)
        already_suppressed = suppress_duplicate_evidence(already_suppressed, reason="duplicate_evidence")
        never_suppressed = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))

        duplicates = find_duplicate_evidence([already_suppressed, never_suppressed])

        self.assertEqual(duplicates, [])
        # The never-suppressed row must still be the sole scoring-eligible
        # row for this claim -- not silently zeroed out.
        self.assertEqual(
            [row.evidence_id for row in active_evidence([already_suppressed, never_suppressed])],
            ["bev_2"],
        )


class SuppressedExcludedFromConfidenceTests(unittest.TestCase):
    def test_suppressed_duplicate_is_excluded_by_active_evidence(self) -> None:
        earlier = _evidence("bev_1", created_at=BASE_TIME)
        later = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))

        duplicates = find_duplicate_evidence([earlier, later])
        suppressed_later = suppress_duplicate_evidence(duplicates[0], reason="duplicate_evidence")

        result = active_evidence([earlier, suppressed_later])
        self.assertEqual([row.evidence_id for row in result], ["bev_1"])

    def test_recompute_belief_confidence_matches_a_single_row_after_suppression(self) -> None:
        earlier = _evidence("bev_1", created_at=BASE_TIME)
        later = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))
        duplicates = find_duplicate_evidence([earlier, later])
        suppressed_later = suppress_duplicate_evidence(duplicates[0], reason="duplicate_evidence")

        as_of = BASE_TIME + timedelta(days=1)
        with_duplicate_kept_out = recompute_belief(
            belief_id="bel_1", user_id="usr_1", belief_type="behavioral_tendency",
            belief_key="x", belief_value=True, evidence=[earlier, suppressed_later],
            as_of=as_of, first_observed=BASE_TIME, **VERSION_FIELDS,
        )
        single_row_only = recompute_belief(
            belief_id="bel_1", user_id="usr_1", belief_type="behavioral_tendency",
            belief_key="x", belief_value=True, evidence=[earlier],
            as_of=as_of, first_observed=BASE_TIME, **VERSION_FIELDS,
        )

        # The suppressed duplicate must contribute nothing -- confidence with
        # it present-but-suppressed must equal confidence computed from the
        # single surviving row alone, not double-counted.
        self.assertEqual(with_duplicate_kept_out.confidence, single_row_only.confidence)
        self.assertEqual(with_duplicate_kept_out.supporting_evidence_count, 1)


class IdempotentReRunTests(unittest.TestCase):
    def test_finding_duplicates_again_after_applying_suppression_yields_nothing_further(self) -> None:
        earlier = _evidence("bev_1", created_at=BASE_TIME)
        later = _evidence("bev_2", created_at=BASE_TIME + timedelta(hours=1))

        first_pass = find_duplicate_evidence([earlier, later])
        suppressed_later = suppress_duplicate_evidence(first_pass[0], reason="duplicate_evidence")

        # Re-running against the now-partially-suppressed list must find
        # nothing left to do: the suppressed row has dropped out of
        # consideration entirely (it is no longer scoring-eligible), not
        # been re-identified as the same duplicate a second time. This is
        # the safer idempotence -- re-flagging the same already-suppressed
        # row would, in a group with more members, risk treating it as the
        # group's "keeper" purely by having the earliest created_at while
        # already suppressed (see AlreadySuppressedRowIsNotAKeeperTests).
        second_pass = find_duplicate_evidence([earlier, suppressed_later])
        self.assertEqual(second_pass, [])


class AuditFieldsIntactTests(unittest.TestCase):
    def test_suppression_changes_only_the_two_suppression_fields(self) -> None:
        original = _evidence("bev_1", created_at=BASE_TIME)

        suppressed = suppress_duplicate_evidence(original, reason="duplicate_evidence")

        self.assertTrue(suppressed.is_duplicate_suppressed)
        self.assertEqual(suppressed.suppression_reason, "duplicate_evidence")
        before = original.model_dump(exclude={"is_duplicate_suppressed", "suppression_reason"})
        after = suppressed.model_dump(exclude={"is_duplicate_suppressed", "suppression_reason"})
        self.assertEqual(before, after)

    def test_suppressed_row_is_still_active_and_still_returned_by_list_evidence_style_queries(self) -> None:
        # Suppression is orthogonal to is_active: the row is not deleted or
        # deactivated, only flagged -- it remains a fully auditable ledger
        # entry, just excluded from active_evidence()'s scoring view.
        original = _evidence("bev_1", created_at=BASE_TIME)
        suppressed = suppress_duplicate_evidence(original, reason="duplicate_evidence")
        self.assertTrue(suppressed.is_active)


if __name__ == "__main__":
    unittest.main()

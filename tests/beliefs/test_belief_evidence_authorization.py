from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.beliefs.models import (
    BeliefEvidence,
    BeliefEvidenceProposal,
    active_evidence,
    authorize_evidence,
    invalidate_evidence,
)

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _proposal(**overrides) -> BeliefEvidenceProposal:
    base = dict(
        belief_id="bel_88", user_id="usr_17", direction="support",
        event_id="evt_1042", observation_id="obs_310", source_event_ids=["evt_1042"],
        source_type="recorded_event", context_key="fitness", strength=0.85,
        source_reliability=0.95, observed_at=datetime(2026, 8, 22, 17, 20, tzinfo=timezone.utc),
        decay_lambda=0.015, model_version="observation-model-0.6", prompt_version="evidence_prompt_05",
        **VERSION_FIELDS,
    )
    base.update(overrides)
    return BeliefEvidenceProposal(**base)


class BeliefEvidenceProposalBoundaryTests(unittest.TestCase):
    """The model-proposes/backend-authorizes boundary (blueprint section 6.1.2,
    section 22 Decision Log), enforced structurally."""

    def test_proposal_has_no_backend_owned_fields_at_all(self) -> None:
        backend_owned = {
            "authorized_aggregation_mode", "aggregation_authorized_by", "aggregation_authorized_at",
            "aggregation_review_required", "aggregation_review_status", "is_duplicate_suppressed",
            "suppression_reason", "is_active", "invalidated_at", "invalidation_reason",
            "evidence_id", "independence_group",
        }
        self.assertEqual(backend_owned.intersection(BeliefEvidenceProposal.model_fields), set())

    def test_proposal_rejects_a_payload_that_tries_to_self_authorize(self) -> None:
        with self.assertRaises(ValidationError):
            _proposal(authorized_aggregation_mode="aggregate_replacement")

    def test_proposal_rejects_a_payload_that_tries_to_self_assign_confidence(self) -> None:
        # confidence is never a belief_evidence field at all -- it belongs
        # only to user_beliefs and is computed by update-user-beliefs.
        with self.assertRaises(ValidationError):
            _proposal(confidence=1.0)

    def test_a_full_belief_evidence_dict_cannot_be_accepted_as_a_mere_proposal(self) -> None:
        full_record = authorize_evidence(
            _proposal(), evidence_id="bev_501", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        with self.assertRaises(ValidationError):
            BeliefEvidenceProposal(**full_record.model_dump())


class EventIdSourceEventIdsConsistencyTests(unittest.TestCase):
    """source_event_ids is the sole authoritative provenance field; event_id
    is a nullable convenience pointer that must never diverge from it."""

    def test_event_id_matching_source_event_ids_is_valid(self) -> None:
        proposal = _proposal(event_id="evt_1042", source_event_ids=["evt_1042"])
        self.assertEqual(proposal.event_id, "evt_1042")

    def test_event_id_absent_with_source_event_ids_present_is_valid(self) -> None:
        # event_id is nullable; a proposal never needs to set it.
        proposal = _proposal(event_id=None, source_event_ids=["evt_1042", "evt_1043"])
        self.assertIsNone(proposal.event_id)

    def test_event_id_not_in_source_event_ids_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _proposal(event_id="evt_9999", source_event_ids=["evt_1042"])

    def test_event_id_only_without_source_event_ids_is_rejected(self) -> None:
        # source_event_ids is a required field (min_length=1); there is no
        # way to construct evidence that relies on event_id alone.
        base = dict(
            belief_id="bel_88", user_id="usr_17", direction="support", event_id="evt_1042",
            source_type="recorded_event", context_key="fitness", strength=0.85,
            source_reliability=0.95, observed_at=datetime.now(timezone.utc),
            decay_lambda=0.015, model_version="observation-model-0.6", **VERSION_FIELDS,
        )
        with self.assertRaises(ValidationError) as ctx:
            BeliefEvidenceProposal(**base)
        missing_fields = {error["loc"][0] for error in ctx.exception.errors()}
        self.assertIn("source_event_ids", missing_fields)


class SourceReliabilityValidationTests(unittest.TestCase):
    """source_reliability must match (or explicitly justify a deviation from)
    the versioned default for the chosen source_type -- an arbitrary 0-1
    value is not accepted silently, since it directly changes downstream
    confidence math (src/beliefs/scoring.py)."""

    def test_recorded_event_at_exact_default_is_accepted(self) -> None:
        proposal = _proposal(source_type="recorded_event", source_reliability=0.95)
        self.assertEqual(proposal.source_reliability, 0.95)

    def test_recorded_event_within_tolerance_is_accepted_without_a_reason(self) -> None:
        # 0.95 default, 0.05 tolerance -> 0.90 is right at the edge and OK.
        proposal = _proposal(source_type="recorded_event", source_reliability=0.90)
        self.assertEqual(proposal.source_reliability, 0.90)

    def test_recorded_event_exactly_at_the_tolerance_boundary_is_accepted_without_a_reason(self) -> None:
        # Regression test for a float-precision edge case: 1.0 - 0.95 ==
        # 0.050000000000000044 in binary floating point, not exactly 0.05,
        # which would spuriously reject a value sitting exactly on the
        # tolerance boundary without the small epsilon in the comparison.
        proposal = _proposal(source_type="recorded_event", source_reliability=1.0)
        self.assertEqual(proposal.source_reliability, 1.0)

    def test_recorded_event_at_0_20_is_rejected_without_a_reason(self) -> None:
        with self.assertRaises(ValidationError):
            _proposal(source_type="recorded_event", source_reliability=0.20)

    def test_recorded_event_at_0_20_is_accepted_with_an_explicit_reason(self) -> None:
        proposal = _proposal(
            source_type="recorded_event", source_reliability=0.20,
            reliability_deviation_reason="Event came from an unverified third-party integration; "
                                          "trust deliberately lowered pending source audit.",
        )
        self.assertEqual(proposal.source_reliability, 0.20)
        self.assertTrue(proposal.reliability_deviation_reason)

    def test_unverified_hypothesis_at_0_99_is_rejected_without_a_reason(self) -> None:
        with self.assertRaises(ValidationError):
            _proposal(
                source_type="unverified_hypothesis", source_reliability=0.99,
                event_id=None, source_event_ids=["evt_9001"],
            )

    def test_unverified_hypothesis_at_0_99_is_accepted_with_an_explicit_reason(self) -> None:
        proposal = _proposal(
            source_type="unverified_hypothesis", source_reliability=0.99,
            event_id=None, source_event_ids=["evt_9001"],
            reliability_deviation_reason="Hypothesis independently corroborated by three unrelated "
                                          "high-confidence sources in the same session; escalated deliberately.",
        )
        self.assertEqual(proposal.source_reliability, 0.99)

    def test_blank_reason_does_not_count_as_justification(self) -> None:
        with self.assertRaises(ValidationError):
            _proposal(source_type="recorded_event", source_reliability=0.20, reliability_deviation_reason="   ")


class AuthorizeEvidenceTests(unittest.TestCase):
    def test_matches_blueprint_worked_example_shape(self) -> None:
        record = authorize_evidence(
            _proposal(), evidence_id="bev_501", created_at=datetime(2026, 8, 22, 17, 21, 10, tzinfo=timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertTrue(record.is_active)
        self.assertFalse(record.is_duplicate_suppressed)
        self.assertEqual(record.independence_group, "evt_1042")

    def test_aggregate_replacement_defaults_to_leaf_default_when_not_approved(self) -> None:
        # This is the core safe-fallback contract: an extractor proposing
        # aggregate_replacement, on its own, must never actually get it.
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement",
            replaces_evidence_ids=["bev_100", "bev_101"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_502", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            # backend_validation_passed intentionally omitted (defaults False)
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "pending")
        self.assertTrue(record.aggregation_review_required)

    def test_aggregate_replacement_is_granted_only_with_explicit_backend_approval(self) -> None:
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement",
            replaces_evidence_ids=["bev_100", "bev_101"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_503", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "aggregate_replacement")
        self.assertEqual(record.aggregation_review_status.value, "approved")

    def test_aggregate_replacement_without_replaces_evidence_ids_is_never_granted(self) -> None:
        # Even with backend_validation_passed=True, a proposal with no
        # explicit replaces_evidence_ids must not be granted replacement.
        proposal = _proposal(proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=[])
        record = authorize_evidence(
            proposal, evidence_id="bev_504", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")

    def test_result_is_a_valid_belief_evidence_instance(self) -> None:
        record = authorize_evidence(
            _proposal(), evidence_id="bev_505", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        self.assertIsInstance(record, BeliefEvidence)


class AuthorizeEvidenceWithReplacedEvidenceValidationTests(unittest.TestCase):
    """When the caller supplies the rows a replacement claims to replace,
    authorize_evidence() verifies same-user/same-belief/coverage itself
    rather than trusting backend_validation_passed blindly."""

    def _existing_row(self, evidence_id: str, **overrides) -> BeliefEvidence:
        defaults = dict(evidence_id=evidence_id, created_at=datetime.now(timezone.utc),
                         aggregation_policy_version="evidence-aggregation-0.6")
        proposal_overrides = {k: v for k, v in overrides.items() if k in BeliefEvidenceProposal.model_fields}
        return authorize_evidence(_proposal(**proposal_overrides), **defaults)

    def test_valid_replacement_covering_all_replaced_rows_is_granted(self) -> None:
        replaced_a = self._existing_row("bev_100", event_id="evt_1042", source_event_ids=["evt_1042"])
        replaced_b = self._existing_row("bev_101", event_id="evt_1043", source_event_ids=["evt_1043"])
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement",
            replaces_evidence_ids=["bev_100", "bev_101"],
            event_id=None, source_event_ids=["evt_1042", "evt_1043"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_600", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            replaced_evidence={"bev_100": replaced_a, "bev_101": replaced_b},
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "aggregate_replacement")
        self.assertEqual(record.aggregation_review_status.value, "approved")

    def test_replacement_from_a_different_user_is_rejected_even_if_backend_validation_passed(self) -> None:
        # A caller asserting backend_validation_passed=True cannot override
        # concrete proof, supplied via replaced_evidence, that the rows
        # belong to a different user.
        replaced = self._existing_row("bev_200", user_id="usr_other", belief_id="bel_88",
                                       event_id="evt_2000", source_event_ids=["evt_2000"])
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_200"],
            event_id=None, source_event_ids=["evt_1042", "evt_2000"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_601", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            replaced_evidence={"bev_200": replaced},
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "rejected")

    def test_replacement_from_a_different_belief_is_rejected(self) -> None:
        replaced = self._existing_row("bev_300", user_id="usr_17", belief_id="bel_other",
                                       event_id="evt_3000", source_event_ids=["evt_3000"])
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_300"],
            event_id=None, source_event_ids=["evt_1042", "evt_3000"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_602", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            replaced_evidence={"bev_300": replaced},
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "rejected")

    def test_replacement_that_narrows_source_event_id_coverage_is_rejected(self) -> None:
        # The replaced row covers evt_4000, but the replacement proposal's
        # own source_event_ids drops it -- silently losing provenance.
        replaced = self._existing_row("bev_400", event_id="evt_4000", source_event_ids=["evt_4000"])
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_400"],
            event_id="evt_1042", source_event_ids=["evt_1042"],  # missing evt_4000
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_603", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            replaced_evidence={"bev_400": replaced},
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "rejected")

    def test_replacement_referencing_a_row_not_supplied_for_validation_is_rejected(self) -> None:
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_500", "bev_501"],
            event_id=None, source_event_ids=["evt_1042", "evt_5000"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_604", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            replaced_evidence={"bev_500": self._existing_row("bev_500", event_id="evt_5000", source_event_ids=["evt_5000"])},
            # bev_501 is missing from replaced_evidence
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "rejected")

    def test_valid_replacement_without_backend_validation_passed_still_only_pends(self) -> None:
        # Passing the data to check is not the same as approving it: the
        # explicit flag is still required even when the rows check out.
        replaced = self._existing_row("bev_700", event_id="evt_7000", source_event_ids=["evt_7000"])
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_700"],
            event_id=None, source_event_ids=["evt_1042", "evt_7000"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_605", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            replaced_evidence={"bev_700": replaced},
            # backend_validation_passed intentionally omitted (defaults False)
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "leaf_default")
        self.assertEqual(record.aggregation_review_status.value, "pending")

    def test_bare_backend_validation_passed_without_replaced_evidence_is_scaffold_only_not_production_safe(
        self,
    ) -> None:
        # Trip-wire, not an endorsement: this pins down today's documented
        # scaffold-only permissiveness (see authorize_evidence()'s docstring
        # TODO) so it cannot silently drift. A caller can currently get
        # aggregate_replacement granted by asserting backend_validation_passed
        # with NO verifiable data behind the claim at all -- that is exactly
        # the non-production-safe path the docstring warns about. If this
        # test starts failing after a change to authorize_evidence(), that
        # means the boundary has genuinely moved (e.g. replaced_evidence or a
        # typed backend-policy decision artifact became mandatory) -- update
        # this test deliberately, and update the docstring/decision log to
        # match, rather than "fixing" the test to make it pass again.
        proposal = _proposal(
            proposed_aggregation_mode="aggregate_replacement",
            replaces_evidence_ids=["bev_900"],
        )
        record = authorize_evidence(
            proposal, evidence_id="bev_606", created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
            backend_validation_passed=True,
            # replaced_evidence intentionally omitted -- nothing here was
            # actually checked, yet replacement is still granted below.
        )
        self.assertEqual(record.authorized_aggregation_mode.value, "aggregate_replacement")
        self.assertEqual(record.aggregation_review_status.value, "approved")


class InvalidateAndActiveEvidenceTests(unittest.TestCase):
    def _record(self, evidence_id: str, **overrides) -> BeliefEvidence:
        record = authorize_evidence(
            _proposal(), evidence_id=evidence_id, created_at=datetime.now(timezone.utc),
            aggregation_policy_version="evidence-aggregation-0.6",
        )
        return record.model_copy(update=overrides) if overrides else record

    def test_invalidate_evidence_sets_inactive_with_reason_and_timestamp(self) -> None:
        record = self._record("bev_601")
        when = datetime.now(timezone.utc)
        invalidated = invalidate_evidence(record, reason="deletion", invalidated_at=when)
        self.assertFalse(invalidated.is_active)
        self.assertEqual(invalidated.invalidation_reason, "deletion")
        self.assertEqual(invalidated.invalidated_at, when)
        # Original record is untouched (Pydantic models here are treated as
        # immutable; invalidate_evidence returns a new instance).
        self.assertTrue(record.is_active)

    def test_active_evidence_excludes_invalidated_and_suppressed_rows(self) -> None:
        active = self._record("bev_602")
        invalidated = invalidate_evidence(self._record("bev_603"), reason="reset", invalidated_at=datetime.now(timezone.utc))
        suppressed = self._record("bev_604", is_duplicate_suppressed=True)
        result = active_evidence([active, invalidated, suppressed])
        self.assertEqual([item.evidence_id for item in result], ["bev_602"])


if __name__ == "__main__":
    unittest.main()

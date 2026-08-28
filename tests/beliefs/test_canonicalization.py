"""Tests for src/beliefs/canonicalization.py -- belief-key canonicalization
(blueprint section 5.2).

Backend-owned authorization, never an LLM/proposal-owned decision:
BeliefKeyCanonicalizationProposal is the complete set of fields a skill or
extractor may set (a candidate key and its own guess at how it should
resolve); only authorize_belief_key_canonicalization() can produce a
persisted BeliefKeyCanonicalization, and it never simply trusts the
proposal's own suggestion.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.canonicalization import (
    BeliefKeyCanonicalization,
    BeliefKeyCanonicalizationProposal,
    authorize_belief_key_canonicalization,
)
from src.beliefs.models import BeliefEvidenceProposal, authorize_evidence
from src.beliefs.recompute import recompute_belief
from src.common.registry import CANONICAL_BELIEF_KEY_REGISTRY_VERSION

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)

AUTHORIZED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _proposal(**overrides) -> BeliefKeyCanonicalizationProposal:
    base = dict(
        user_id="usr_1", belief_type="behavioral_tendency", proposed_key="some_key", **VERSION_FIELDS,
    )
    base.update(overrides)
    return BeliefKeyCanonicalizationProposal(**base)


def _authorize(proposal: BeliefKeyCanonicalizationProposal, **overrides) -> BeliefKeyCanonicalization:
    kwargs = dict(canonicalization_id="can_1", authorized_at=AUTHORIZED_AT)
    kwargs.update(overrides)
    return authorize_belief_key_canonicalization(proposal, **kwargs)


class ExactKnownAliasTests(unittest.TestCase):
    def test_known_alias_resolves_to_its_canonical_key(self) -> None:
        proposal = _proposal(proposed_key="prefers_evening_exercise_sessions")

        result = _authorize(proposal)

        self.assertEqual(result.canonical_key, "higher_adherence_after_work")
        self.assertEqual(result.decision.value, "alias")
        self.assertEqual(result.canonical_key_registry_version, CANONICAL_BELIEF_KEY_REGISTRY_VERSION)

    def test_a_second_known_alias_resolves_to_the_same_canonical_key(self) -> None:
        proposal = _proposal(proposed_key="more_consistent_after_work_workouts")

        result = _authorize(proposal)

        self.assertEqual(result.canonical_key, "higher_adherence_after_work")
        self.assertEqual(result.decision.value, "alias")

    def test_registry_hit_wins_even_when_the_proposal_itself_suggests_keeping_separate(self) -> None:
        # The registry is the actual backend policy data; a proposal's own
        # (safe-default) guess must not override a known deterministic match.
        proposal = _proposal(
            proposed_key="prefers_evening_exercise_sessions", proposed_decision="keep_separate",
        )

        result = _authorize(proposal)

        self.assertEqual(result.canonical_key, "higher_adherence_after_work")
        self.assertEqual(result.decision.value, "alias")

    def test_custom_registry_override_is_honored_for_testability(self) -> None:
        custom_registry = {("behavioral_tendency", "gym_after_5pm"): "higher_adherence_after_work"}
        proposal = _proposal(proposed_key="gym_after_5pm")

        result = _authorize(proposal, registry=custom_registry, registry_version="test-registry-1")

        self.assertEqual(result.canonical_key, "higher_adherence_after_work")
        self.assertEqual(result.decision.value, "alias")
        self.assertEqual(result.canonical_key_registry_version, "test-registry-1")


class UnknownKeyStaysSeparateTests(unittest.TestCase):
    def test_unknown_key_is_kept_as_its_own_canonical_key(self) -> None:
        proposal = _proposal(proposed_key="a_totally_novel_belief_key")

        result = _authorize(proposal)

        self.assertEqual(result.canonical_key, "a_totally_novel_belief_key")
        self.assertEqual(result.decision.value, "keep_separate")

    def test_unknown_key_for_a_different_belief_type_is_also_kept_separate(self) -> None:
        # The same spelling that IS a known alias for one belief_type must
        # not accidentally match under a different belief_type -- the
        # registry key is (belief_type, proposed_key), not proposed_key alone.
        proposal = _proposal(belief_type="goal_or_intention", proposed_key="prefers_evening_exercise_sessions")

        result = _authorize(proposal)

        self.assertEqual(result.canonical_key, "prefers_evening_exercise_sessions")
        self.assertEqual(result.decision.value, "keep_separate")


class RiskyOrAmbiguousMergeGoesToReviewTests(unittest.TestCase):
    def test_merge_request_without_a_registry_match_is_downgraded_to_manual_review(self) -> None:
        proposal = _proposal(
            proposed_key="some_new_key", proposed_decision="merge", reason="seems related to an existing belief",
        )

        result = _authorize(proposal)

        self.assertEqual(result.decision.value, "manual_review")
        self.assertEqual(result.canonical_key, "some_new_key")

    def test_alias_request_to_an_unverified_target_is_downgraded_to_manual_review(self) -> None:
        proposal = _proposal(
            proposed_key="another_key", proposed_decision="alias",
            proposed_canonical_key="higher_adherence_after_work",
        )

        result = _authorize(proposal)

        self.assertEqual(result.decision.value, "manual_review")
        self.assertEqual(result.canonical_key, "another_key")

    def test_alias_request_naming_itself_as_the_canonical_key_is_not_treated_as_risky(self) -> None:
        # proposed_canonical_key == proposed_key carries no real alias
        # information (it is not requesting a merge into anything else), so
        # this is equivalent to no request at all: falls through to the
        # unknown-key default.
        proposal = _proposal(
            proposed_key="some_key", proposed_decision="alias", proposed_canonical_key="some_key",
        )

        result = _authorize(proposal)

        self.assertEqual(result.decision.value, "keep_separate")
        self.assertEqual(result.canonical_key, "some_key")


class ProposalCannotOverrideBackendPolicyTests(unittest.TestCase):
    def test_proposal_has_no_backend_owned_fields_at_all(self) -> None:
        backend_owned = {
            "canonicalization_id", "canonical_key", "decision", "decision_reason",
            "authorized_by", "authorized_at", "canonical_key_registry_version",
        }
        self.assertEqual(backend_owned.intersection(BeliefKeyCanonicalizationProposal.model_fields), set())

    def test_proposal_rejects_a_payload_that_tries_to_self_authorize(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            _proposal(decision="alias")

    def test_requesting_merge_never_actually_grants_merge(self) -> None:
        # However confidently a proposal asks for "merge", authorization
        # never grants it in this phase -- there is no deterministic guard
        # here that could clear the blueprint's high-similarity/narrow-case
        # bar, so the LLM's own judgment call is never the final word.
        proposal = _proposal(proposed_decision="merge", reason="I am very confident these are the same")

        result = _authorize(proposal)

        self.assertNotEqual(result.decision.value, "merge")
        self.assertEqual(result.decision.value, "manual_review")

    def test_requesting_alias_never_grants_alias_without_a_registry_match(self) -> None:
        proposal = _proposal(
            proposed_decision="alias", proposed_canonical_key="some_other_key",
            reason="I am confident this is the same belief",
        )

        result = _authorize(proposal)

        self.assertNotEqual(result.decision.value, "alias")
        self.assertEqual(result.decision.value, "manual_review")


class RecomputeUsesCanonicalKeyOnlyAfterAuthorizationTests(unittest.TestCase):
    def _evidence_for(self, belief_id: str, *, created_at: datetime):
        proposal = BeliefEvidenceProposal(
            belief_id=belief_id, user_id="usr_1", direction="support", event_id="evt_1",
            source_event_ids=["evt_1"], source_type="recorded_event", context_key="fitness",
            strength=0.85, source_reliability=0.95, observed_at=created_at,
            decay_lambda=0.015, model_version="test-model-0.1", **VERSION_FIELDS,
        )
        return authorize_evidence(
            proposal, evidence_id=f"bev_{belief_id}", created_at=created_at,
            aggregation_policy_version="evidence-aggregation-0.6",
        )

    def test_belief_key_persisted_by_recompute_is_the_authorized_canonical_key(self) -> None:
        proposal = _proposal(proposed_key="prefers_evening_exercise_sessions")
        authorized = _authorize(proposal)
        self.assertNotEqual(authorized.canonical_key, proposal.proposed_key)

        evidence = self._evidence_for("bel_1", created_at=AUTHORIZED_AT)
        belief = recompute_belief(
            belief_id="bel_1", user_id="usr_1", belief_type="behavioral_tendency",
            belief_key=authorized.canonical_key, belief_value=True, evidence=[evidence],
            as_of=AUTHORIZED_AT, first_observed=AUTHORIZED_AT, **VERSION_FIELDS,
        )

        # The persisted belief_key must be the authorized canonical form,
        # never the raw, pre-authorization proposed_key.
        self.assertEqual(belief.belief_key, "higher_adherence_after_work")
        self.assertNotEqual(belief.belief_key, proposal.proposed_key)

    def test_two_different_proposed_keys_aliasing_to_the_same_canonical_key_recompute_under_one_belief_key(
        self,
    ) -> None:
        first_authorized = _authorize(_proposal(proposed_key="prefers_evening_exercise_sessions"))
        second_authorized = _authorize(
            _proposal(proposed_key="more_consistent_after_work_workouts"), canonicalization_id="can_2",
        )

        # Both proposals -- different raw spellings -- must resolve to the
        # exact same canonical_key, which is what prevents them from
        # fragmenting evidence across two different UserBelief rows.
        self.assertEqual(first_authorized.canonical_key, second_authorized.canonical_key)

        evidence = self._evidence_for("bel_1", created_at=AUTHORIZED_AT)
        belief = recompute_belief(
            belief_id="bel_1", user_id="usr_1", belief_type="behavioral_tendency",
            belief_key=second_authorized.canonical_key, belief_value=True, evidence=[evidence],
            as_of=AUTHORIZED_AT, first_observed=AUTHORIZED_AT, **VERSION_FIELDS,
        )
        self.assertEqual(belief.belief_key, first_authorized.canonical_key)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for skills/shared/contract-review/scripts/validate_output.py,
specifically the evidence (pre-authorization proposal) vs evidence_record
(authorized, persisted row) split -- these check genuinely different things,
and running the wrong one against the wrong shape either misses real
violations or false-flags legitimate output (see the module's own docstring).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "skills" / "shared" / "contract-review" / "scripts"
_SCRIPT_PATH = _SCRIPT_DIR / "validate_output.py"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validate_output import (  # type: ignore  # noqa: E402
    validate_belief,
    validate_evidence,
    validate_evidence_record,
    validate_observation,
)

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)


def _proposal(**overrides) -> dict:
    base = dict(
        evidence_id="bev_1", belief_id="bel_88", user_id="usr_17", direction="support",
        event_id="evt_1042", source_event_ids=["evt_1042"], source_type="recorded_event",
        context_key="fitness", strength=0.85, source_reliability=0.95, **VERSION_FIELDS,
    )
    base.update(overrides)
    return base


def _authorized_record(**overrides) -> dict:
    base = _proposal(
        authorized_aggregation_mode="leaf_default",
        aggregation_authorized_by="evidence_policy",
        aggregation_authorized_at="2026-08-22T17:21:11-05:00",
    )
    base.update(overrides)
    return base


def _belief(**overrides) -> dict:
    base = dict(
        belief_id="bel_88", belief_type="behavioral_tendency", status="validated",
        confidence=0.68, **VERSION_FIELDS,
    )
    base.update(overrides)
    return base


def _observation(**overrides) -> dict:
    base = dict(observation_id="obs_1", importance=0.72, confidence=0.86, **VERSION_FIELDS)
    base.update(overrides)
    return base


class ValidateEvidenceProposalTests(unittest.TestCase):
    """The `evidence` check: pre-authorization proposal shape."""

    def test_valid_proposal_passes(self) -> None:
        self.assertEqual(validate_evidence(_proposal()), [])

    def test_proposal_rejects_backend_owned_fields(self) -> None:
        errors = validate_evidence(_proposal(
            authorized_aggregation_mode="leaf_default",
            aggregation_authorized_by="evidence_policy",
            aggregation_authorized_at="2026-08-22T17:21:11-05:00",
        ))
        self.assertTrue(any("backend-only field" in e for e in errors))

    def test_proposal_rejects_a_fully_authorized_record(self) -> None:
        # This is the exact scenario that used to confuse contract-review
        # runs: a legitimately authorized record must still fail the
        # proposal check, because that check's whole point is "did this come
        # straight from a model/extractor with no backend involvement."
        errors = validate_evidence(_authorized_record())
        self.assertTrue(any("backend-only field" in e for e in errors))

    def test_proposal_rejects_event_id_not_in_source_event_ids(self) -> None:
        errors = validate_evidence(_proposal(event_id="evt_ghost", source_event_ids=["evt_1042"]))
        self.assertTrue(any("not included in source_event_ids" in e for e in errors))

    def test_proposal_rejects_unknown_source_type(self) -> None:
        errors = validate_evidence(_proposal(source_type="made_up_source"))
        self.assertTrue(any("canonical values" in e for e in errors))


class ValidateEvidenceRecordTests(unittest.TestCase):
    """The `evidence_record` check: already-authorized, persisted row."""

    def test_valid_leaf_default_record_passes(self) -> None:
        self.assertEqual(validate_evidence_record(_authorized_record()), [])

    def test_valid_aggregate_replacement_record_passes(self) -> None:
        errors = validate_evidence_record(_authorized_record(
            authorized_aggregation_mode="aggregate_replacement",
            replaces_evidence_ids=["bev_100", "bev_101"],
        ))
        self.assertEqual(errors, [])

    def test_record_does_not_flag_authorized_aggregation_mode_as_a_violation(self) -> None:
        # The exact false positive being fixed: evidence_record must NOT
        # treat the presence of these backend fields as a violation.
        errors = validate_evidence_record(_authorized_record())
        self.assertFalse(any("backend-only field" in e for e in errors))

    def test_record_rejects_invalid_aggregation_mode(self) -> None:
        errors = validate_evidence_record(_authorized_record(authorized_aggregation_mode="not_a_real_mode"))
        self.assertTrue(any("must be one of" in e for e in errors))

    def test_record_rejects_missing_aggregation_mode(self) -> None:
        record = _authorized_record()
        del record["authorized_aggregation_mode"]
        errors = validate_evidence_record(record)
        self.assertTrue(any("must be one of" in e for e in errors))

    def test_record_rejects_missing_authorized_by(self) -> None:
        record = _authorized_record()
        del record["aggregation_authorized_by"]
        errors = validate_evidence_record(record)
        self.assertTrue(any("aggregation_authorized_by is missing" in e for e in errors))

    def test_record_rejects_aggregate_replacement_without_replaces_evidence_ids(self) -> None:
        errors = validate_evidence_record(_authorized_record(authorized_aggregation_mode="aggregate_replacement"))
        self.assertTrue(any("replaces_evidence_ids is empty" in e for e in errors))

    def test_record_rejects_aggregate_replacement_without_authorized_at(self) -> None:
        record = _authorized_record(authorized_aggregation_mode="aggregate_replacement", replaces_evidence_ids=["bev_100"])
        del record["aggregation_authorized_at"]
        errors = validate_evidence_record(record)
        self.assertTrue(any("aggregation_authorized_at is missing" in e for e in errors))

    def test_record_rejects_a_bare_proposal_missing_authorization_metadata(self) -> None:
        errors = validate_evidence_record(_proposal())
        self.assertTrue(any("must be one of" in e for e in errors))
        self.assertTrue(any("aggregation_authorized_by is missing" in e for e in errors))


class EvidenceStrengthAndReliabilityNumericTests(unittest.TestCase):
    """strength and source_reliability must be real numbers in [0, 1] --
    shared by both evidence (proposal) and evidence_record (authorized),
    since blueprint section 5.3 declares both fields identically on the
    underlying belief_evidence shape."""

    def test_proposal_rejects_boolean_strength(self) -> None:
        errors = validate_evidence(_proposal(strength=True))
        self.assertTrue(any("strength must be a real number" in e for e in errors))

    def test_proposal_rejects_boolean_source_reliability(self) -> None:
        errors = validate_evidence(_proposal(source_reliability=False))
        self.assertTrue(any("source_reliability must be a real number" in e for e in errors))

    def test_proposal_rejects_numeric_string_strength(self) -> None:
        errors = validate_evidence(_proposal(strength="0.85"))
        self.assertTrue(any("strength must be a real number" in e for e in errors))

    def test_proposal_rejects_numeric_string_source_reliability(self) -> None:
        errors = validate_evidence(_proposal(source_reliability="0.95"))
        self.assertTrue(any("source_reliability must be a real number" in e for e in errors))

    def test_proposal_rejects_out_of_range_strength(self) -> None:
        errors = validate_evidence(_proposal(strength=1.5))
        self.assertTrue(any("strength must be a real number in [0, 1]" in e for e in errors))

    def test_proposal_rejects_missing_strength(self) -> None:
        record = _proposal()
        del record["strength"]
        errors = validate_evidence(record)
        self.assertTrue(any("strength must be a real number" in e for e in errors))

    def test_record_rejects_boolean_strength(self) -> None:
        errors = validate_evidence_record(_authorized_record(strength=True))
        self.assertTrue(any("strength must be a real number" in e for e in errors))

    def test_record_rejects_numeric_string_source_reliability(self) -> None:
        errors = validate_evidence_record(_authorized_record(source_reliability="0.95"))
        self.assertTrue(any("source_reliability must be a real number" in e for e in errors))

    def test_record_rejects_out_of_range_source_reliability(self) -> None:
        errors = validate_evidence_record(_authorized_record(source_reliability=-0.1))
        self.assertTrue(any("source_reliability must be a real number in [0, 1]" in e for e in errors))


class ValidateBeliefTests(unittest.TestCase):
    def test_valid_belief_passes(self) -> None:
        self.assertEqual(validate_belief(_belief()), [])

    def test_confidence_of_0_0_passes(self) -> None:
        self.assertEqual(validate_belief(_belief(confidence=0.0, status="candidate")), [])

    def test_boolean_confidence_is_rejected(self) -> None:
        errors = validate_belief(_belief(confidence=True))
        self.assertTrue(any("confidence must be a real number" in e for e in errors))

    def test_boolean_false_confidence_is_rejected(self) -> None:
        # False == 0 numerically, but must not be silently treated as the
        # valid confidence=0.0 no-evidence case.
        errors = validate_belief(_belief(confidence=False))
        self.assertTrue(any("confidence must be a real number" in e for e in errors))

    def test_numeric_string_confidence_is_rejected(self) -> None:
        errors = validate_belief(_belief(confidence="0.68"))
        self.assertTrue(any("confidence must be a real number" in e for e in errors))

    def test_confidence_outside_band_is_rejected(self) -> None:
        errors = validate_belief(_belief(confidence=0.01))
        self.assertTrue(any("outside the [0.02, 0.98] band" in e for e in errors))

    def test_confidence_of_1_0_is_rejected(self) -> None:
        errors = validate_belief(_belief(confidence=1.0))
        self.assertTrue(any("outside the [0.02, 0.98] band" in e for e in errors))

    def test_unknown_belief_type_is_rejected(self) -> None:
        errors = validate_belief(_belief(belief_type="made_up_type"))
        self.assertTrue(any("canonical values" in e for e in errors))

    def test_unknown_status_is_rejected(self) -> None:
        errors = validate_belief(_belief(status="super_valid"))
        self.assertTrue(any("canonical lifecycle values" in e for e in errors))


class ValidateObservationTests(unittest.TestCase):
    def test_valid_observation_passes(self) -> None:
        self.assertEqual(validate_observation(_observation()), [])

    def test_boolean_importance_is_rejected(self) -> None:
        errors = validate_observation(_observation(importance=True))
        self.assertTrue(any("importance must be a real number" in e for e in errors))

    def test_boolean_confidence_is_rejected(self) -> None:
        errors = validate_observation(_observation(confidence=False))
        self.assertTrue(any("confidence must be a real number" in e for e in errors))

    def test_numeric_string_importance_is_rejected(self) -> None:
        errors = validate_observation(_observation(importance="0.72"))
        self.assertTrue(any("importance must be a real number" in e for e in errors))

    def test_numeric_string_confidence_is_rejected(self) -> None:
        errors = validate_observation(_observation(confidence="0.86"))
        self.assertTrue(any("confidence must be a real number" in e for e in errors))

    def test_out_of_range_importance_is_rejected(self) -> None:
        errors = validate_observation(_observation(importance=1.5))
        self.assertTrue(any("importance must be a real number in [0, 1]" in e for e in errors))


class ValidateOutputCliTests(unittest.TestCase):
    """End-to-end CLI checks, including the id-field-labeling fix for the
    evidence_record record type (there is no evidence_record_id field --
    the id is still evidence_id)."""

    def _run(self, record_type: str, records: list[dict]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), record_type, str(path)],
                capture_output=True, text=True,
            )

    def test_cli_evidence_proposal_passes(self) -> None:
        result = self._run("evidence", [_proposal()])
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK", result.stdout)

    def test_cli_evidence_record_passes_and_labels_by_evidence_id(self) -> None:
        result = self._run("evidence_record", [_authorized_record()])
        self.assertEqual(result.returncode, 0)

    def test_cli_evidence_record_failure_labels_error_with_evidence_id_not_placeholder(self) -> None:
        broken = _authorized_record(authorized_aggregation_mode="not_a_real_mode")
        result = self._run("evidence_record", [broken])
        self.assertEqual(result.returncode, 1)
        self.assertIn("(bev_1)", result.stderr)
        self.assertNotIn("(?)", result.stderr)

    def test_cli_evidence_against_authorized_record_fails(self) -> None:
        result = self._run("evidence", [_authorized_record()])
        self.assertEqual(result.returncode, 1)
        self.assertIn("backend-only field", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for
skills/shared/create-belief-evidence/scripts/validate_evidence.py, in
particular the reliability-deviation-justification contract, which must stay
aligned with BeliefEvidenceProposal in src/beliefs/models.py: a
source_reliability more than RELIABILITY_TOLERANCE from the canonical
default is accepted only when reliability_deviation_reason is present and
non-blank.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "skills" / "shared" / "create-belief-evidence" / "scripts"
_SCRIPT_PATH = _SCRIPT_DIR / "validate_evidence.py"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validate_evidence import validate_row  # type: ignore  # noqa: E402


def _row(**overrides) -> dict:
    base = dict(
        evidence_id="bev_1", direction="support", source_event_ids=["evt_1042"],
        source_type="recorded_event", source_reliability=0.95, strength=0.85,
    )
    base.update(overrides)
    return base


class ReliabilityMatchesDefaultTests(unittest.TestCase):
    def test_exact_default_passes(self) -> None:
        self.assertEqual(validate_row(_row()), [])

    def test_within_tolerance_passes_without_a_reason(self) -> None:
        self.assertEqual(validate_row(_row(source_reliability=0.90)), [])

    def test_exactly_at_the_tolerance_boundary_passes_without_a_reason(self) -> None:
        # Regression test matching src/beliefs/models.py: 1.0 - 0.95 ==
        # 0.050000000000000044 in binary floating point, not exactly 0.05;
        # must not be spuriously rejected as "beyond tolerance."
        self.assertEqual(validate_row(_row(source_reliability=1.0)), [])


class ReliabilityDeviationJustificationTests(unittest.TestCase):
    def test_deviation_beyond_tolerance_without_reason_is_rejected(self) -> None:
        errors = validate_row(_row(source_reliability=0.20))
        self.assertTrue(any("reliability_deviation_reason" in e for e in errors))

    def test_deviation_beyond_tolerance_with_justified_reason_is_accepted(self) -> None:
        errors = validate_row(_row(
            source_reliability=0.20,
            reliability_deviation_reason="Event came from an unverified third-party integration; "
                                          "trust deliberately lowered pending source audit.",
        ))
        self.assertEqual(errors, [])

    def test_deviation_with_blank_reason_is_rejected(self) -> None:
        errors = validate_row(_row(source_reliability=0.20, reliability_deviation_reason="   "))
        self.assertTrue(any("reliability_deviation_reason" in e for e in errors))

    def test_deviation_with_missing_reason_key_is_rejected(self) -> None:
        row = _row(source_reliability=0.20)
        self.assertNotIn("reliability_deviation_reason", row)
        errors = validate_row(row)
        self.assertTrue(any("reliability_deviation_reason" in e for e in errors))

    def test_unverified_hypothesis_at_0_99_is_rejected_without_a_reason(self) -> None:
        errors = validate_row(_row(source_type="unverified_hypothesis", source_reliability=0.99))
        self.assertTrue(any("reliability_deviation_reason" in e for e in errors))

    def test_unverified_hypothesis_at_0_99_is_accepted_with_a_reason(self) -> None:
        errors = validate_row(_row(
            source_type="unverified_hypothesis", source_reliability=0.99,
            reliability_deviation_reason="Independently corroborated by three unrelated high-confidence sources.",
        ))
        self.assertEqual(errors, [])


class ReliabilityTypeAndRangeTests(unittest.TestCase):
    """These are unconditional -- no reason can justify them, unlike a mere
    deviation from the canonical default."""

    def test_nonnumeric_reliability_is_rejected_even_with_a_reason(self) -> None:
        errors = validate_row(_row(source_reliability="high", reliability_deviation_reason="trust me"))
        self.assertTrue(any("must be numeric" in e for e in errors))

    def test_boolean_reliability_is_rejected(self) -> None:
        # bool is a subclass of int in Python; must not silently pass as 1.0/0.0.
        errors = validate_row(_row(source_reliability=True))
        self.assertTrue(any("must be numeric" in e for e in errors))

    def test_out_of_range_reliability_is_rejected_even_with_a_reason(self) -> None:
        errors = validate_row(_row(source_reliability=1.5, reliability_deviation_reason="trust me"))
        self.assertTrue(any("outside the [0, 1] range" in e for e in errors))

    def test_negative_reliability_is_rejected(self) -> None:
        errors = validate_row(_row(source_reliability=-0.1))
        self.assertTrue(any("outside the [0, 1] range" in e for e in errors))


class StrengthValidationTests(unittest.TestCase):
    """The blueprint schema declares strength: [0,1] (section 5.3), but the
    script previously only validated source_reliability, not strength."""

    def test_valid_strength_passes(self) -> None:
        self.assertEqual(validate_row(_row(strength=0.85)), [])

    def test_missing_strength_is_rejected(self) -> None:
        row = _row()
        del row["strength"]
        errors = validate_row(row)
        self.assertTrue(any("strength must be numeric" in e for e in errors))

    def test_nonnumeric_strength_is_rejected(self) -> None:
        errors = validate_row(_row(strength="high"))
        self.assertTrue(any("strength must be numeric" in e for e in errors))

    def test_boolean_strength_is_rejected(self) -> None:
        errors = validate_row(_row(strength=True))
        self.assertTrue(any("strength must be numeric" in e for e in errors))

    def test_out_of_range_strength_is_rejected(self) -> None:
        errors = validate_row(_row(strength=1.5))
        self.assertTrue(any("strength" in e and "outside the [0, 1] range" in e for e in errors))

    def test_negative_strength_is_rejected(self) -> None:
        errors = validate_row(_row(strength=-0.1))
        self.assertTrue(any("strength" in e and "outside the [0, 1] range" in e for e in errors))


class UnknownSourceTypeTests(unittest.TestCase):
    def test_unknown_source_type_is_rejected(self) -> None:
        errors = validate_row(_row(source_type="made_up_source_type"))
        self.assertTrue(any("is not one of the seven canonical values" in e for e in errors))
        # No reliability-band error should also fire once source_type is unknown
        # (there is no default to compare against).
        self.assertFalse(any("deviates from the canonical default" in e for e in errors))


class ValidateEvidenceCliTests(unittest.TestCase):
    def _run(self, rows: list[dict]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "batch.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(_SCRIPT_PATH), str(path)], capture_output=True, text=True,
            )

    def test_cli_accepts_justified_deviation_batch(self) -> None:
        result = self._run([_row(
            source_reliability=0.20,
            reliability_deviation_reason="Deliberately lowered trust pending source audit.",
        )])
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK", result.stdout)

    def test_cli_rejects_unjustified_deviation_batch(self) -> None:
        result = self._run([_row(source_reliability=0.20)])
        self.assertEqual(result.returncode, 1)
        self.assertIn("reliability_deviation_reason", result.stderr)


if __name__ == "__main__":
    unittest.main()

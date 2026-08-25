from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.common.versioning import VersionedModel


class _Record(VersionedModel):
    name: str


class VersionedModelTests(unittest.TestCase):
    def test_all_four_version_fields_are_required(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            _Record(name="x")
        missing = {error["loc"][0] for error in ctx.exception.errors()}
        self.assertEqual(
            missing,
            {"schema_version", "scoring_version", "canonicalizer_version", "policy_version"},
        )

    def test_blank_version_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            _Record(name="x", schema_version="", scoring_version="v", canonicalizer_version="v", policy_version="v")

    def test_valid_record_constructs(self) -> None:
        record = _Record(
            name="x", schema_version="6", scoring_version="v", canonicalizer_version="v", policy_version="v"
        )
        self.assertEqual(record.schema_version, "6")

    def test_extra_field_is_rejected(self) -> None:
        # This is the structural half of the model-proposes/backend-authorizes
        # boundary: any subclass inherits extra="forbid" for free.
        with self.assertRaises(ValidationError):
            _Record(
                name="x",
                schema_version="6",
                scoring_version="v",
                canonicalizer_version="v",
                policy_version="v",
                not_a_real_field="nope",
            )


if __name__ == "__main__":
    unittest.main()

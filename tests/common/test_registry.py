from __future__ import annotations

import unittest

from src.common.enums import BeliefType, SourceType
from src.common.registry import BELIEF_TYPE_REGISTRY, SOURCE_TYPE_RELIABILITY


class RegistryCompletenessTests(unittest.TestCase):
    def test_every_belief_type_has_registry_defaults(self) -> None:
        self.assertEqual(set(BELIEF_TYPE_REGISTRY.keys()), set(BeliefType))

    def test_every_source_type_has_a_reliability_default(self) -> None:
        self.assertEqual(set(SOURCE_TYPE_RELIABILITY.keys()), set(SourceType))

    def test_reliability_defaults_match_blueprint_section_6_2_1(self) -> None:
        self.assertEqual(SOURCE_TYPE_RELIABILITY[SourceType.EXPLICIT_USER_CORRECTION], 1.00)
        self.assertEqual(SOURCE_TYPE_RELIABILITY[SourceType.RECORDED_EVENT], 0.95)
        self.assertEqual(SOURCE_TYPE_RELIABILITY[SourceType.UNVERIFIED_HYPOTHESIS], 0.35)

    def test_sensitive_belief_type_defaults_to_restricted_and_do_not_persist(self) -> None:
        defaults = BELIEF_TYPE_REGISTRY[BeliefType.SENSITIVE_OR_HIGH_IMPACT_INFERENCE]
        self.assertEqual(defaults.default_sensitivity_class.value, "restricted")
        self.assertEqual(defaults.default_persistence_policy.value, "do_not_persist")
        self.assertEqual(defaults.expected_source_types, 4)

    def test_current_state_related_defaults_to_short_term_and_fast_decay(self) -> None:
        defaults = BELIEF_TYPE_REGISTRY[BeliefType.CURRENT_STATE_RELATED]
        self.assertEqual(defaults.default_persistence_policy.value, "short_term")
        self.assertEqual(defaults.default_decay_lambda, 0.080)


if __name__ == "__main__":
    unittest.main()

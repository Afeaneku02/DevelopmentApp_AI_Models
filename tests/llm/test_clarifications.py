"""Bounded clarification context on POST /users/{user_id}/mentor-guidance.

`clarifications` carries the mentor's earlier clarifying questions in one
interaction and the user's answers, resent by the client each request. This
service keeps no conversation state: the turns reach the provider for that
request only, pass the same high-stakes gate as goal/question, never bypass
policy or evidence gates, and never touch the database.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from src.api.app import create_app
from src.llm.base import MockMentorProvider
from src.llm.config import Settings
from src.llm.models import ClarificationTurn, MentorGuidance, MentorRequest
from src.llm.openai_provider import OpenAIMentorProvider
from src.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from src.mentor.guidance import build_context
from src.storage.repository import Repository
from tests.mentor._helpers import seed_belief

TURN = {"question": "What time of day could you usually walk?", "answer": "10 minutes"}


def guidance(belief_id="b1"):
    return MentorGuidance(
        summary="A short morning walk may fit.",
        recommendations=[{"action": "Try a 10-minute walk.", "reason": "You said 10 minutes.",
                          "grounded_in_belief_ids": [belief_id]}],
        clarifying_question=None, needs_more_information=False, needs_web=False,
    )


class ClarificationRequestModelTests(unittest.TestCase):
    def test_defaults_to_no_turns(self):
        self.assertEqual(MentorRequest(context_key="fitness_scheduling").clarifications, [])

    def test_accepts_up_to_two_turns_and_trims(self):
        request = MentorRequest.model_validate_json(json.dumps({
            "context_key": "fitness_scheduling",
            "clarifications": [{"question": "  Q1? ", "answer": " A1 "}, TURN],
        }))
        self.assertEqual(request.clarifications[0], ClarificationTurn(question="Q1?", answer="A1"))

    def test_rejects_more_than_two_turns(self):
        with self.assertRaises(ValidationError):
            MentorRequest.model_validate_json(json.dumps({"context_key": "x", "clarifications": [TURN] * 3}))

    def test_rejects_blank_long_missing_or_extra_fields(self):
        bad_turns = (
            {"question": "  ", "answer": "a"},
            {"question": "q", "answer": ""},
            {"question": "q" * 501, "answer": "a"},
            {"question": "q", "answer": "a" * 501},
            {"answer": "a"},
            {"question": "q", "answer": "a", "confidence": 1.0},
        )
        for turn in bad_turns:
            with self.subTest(turn=turn), self.assertRaises(ValidationError):
                MentorRequest.model_validate_json(json.dumps({"context_key": "x", "clarifications": [turn]}))

    def test_limits_count_unicode_code_points(self):
        # 500 emoji: 500 code points (allowed), 1000 UTF-16 units.
        MentorRequest.model_validate_json(json.dumps(
            {"context_key": "x", "clarifications": [{"question": "q", "answer": "\U0001F642" * 500}]}))


class ClarificationContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "user.sqlite3"
        self.repo = Repository.at_path(self.db)
        self.addCleanup(self.repo.close)
        seed_belief(self.repo, user_id="u1", belief_id="b1")

    def post(self, provider, payload, user="u1"):
        before = self.db.read_bytes()
        with TestClient(create_app(self.db, mentor_provider=provider)) as client:
            result = client.post(f"/users/{user}/mentor-guidance", json=payload)
        self.assertEqual(before, self.db.read_bytes(), "clarifications must never be written")
        return result

    def test_turns_reach_the_provider_in_order(self):
        provider = MockMentorProvider(guidance())
        turns = [{"question": "Which activity?", "answer": "Walking"}, TURN]
        result = self.post(provider, {"context_key": "fitness_scheduling", "goal": "Get fitter", "clarifications": turns})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "mock")
        sent = provider.contexts[0].clarifications
        self.assertEqual([(t.question, t.answer) for t in sent],
                         [("Which activity?", "Walking"), (TURN["question"], TURN["answer"])])

    def test_turns_change_the_snapshot(self):
        plain = build_context(self.repo, "u1", MentorRequest(context_key="fitness_scheduling"))
        with_turns = build_context(self.repo, "u1", MentorRequest(
            context_key="fitness_scheduling", clarifications=[ClarificationTurn(**TURN)]))
        self.assertNotEqual(plain.snapshot_id, with_turns.snapshot_id)

    def test_over_limit_payload_is_422_without_provider_call(self):
        provider = MockMentorProvider(guidance())
        result = self.post(provider, {"context_key": "fitness_scheduling", "clarifications": [TURN] * 3})
        self.assertEqual(result.status_code, 422)
        self.assertEqual(provider.contexts, [])

    def test_high_stakes_text_in_an_answer_or_question_blocks_the_provider(self):
        for turn in ({"question": "Anything else?", "answer": "should I change medication"},
                     {"question": "Should you change medication?", "answer": "yes"}):
            with self.subTest(turn=turn):
                provider = MockMentorProvider(guidance())
                result = self.post(provider, {"context_key": "fitness_scheduling", "clarifications": [turn]})
                self.assertEqual(result.json()["reason_code"], "consequential_request")
                self.assertEqual(provider.contexts, [])

    def test_turns_cannot_bypass_risk_policy(self):
        provider = MockMentorProvider(guidance())
        result = self.post(provider, {"context_key": "financial_planning", "clarifications": [TURN]})
        self.assertEqual(result.json()["reason_code"], "policy_requires_resolution")
        self.assertEqual(provider.contexts, [])

    def test_turns_cannot_substitute_for_evidence(self):
        provider = MockMentorProvider(guidance())
        result = self.post(provider, {"context_key": "fitness_scheduling", "clarifications": [TURN]}, user="no-evidence")
        self.assertEqual(result.json()["reason_code"], "insufficient_authorized_evidence")
        self.assertEqual(provider.contexts, [])


class ClarificationProviderTests(unittest.TestCase):
    def test_openai_payload_includes_turns_and_prompt_explains_them(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Repository.at_path(Path(tmp.name) / "db.sqlite3")
        self.addCleanup(repo.close)
        seed_belief(repo, user_id="u1", belief_id="b1")
        context = build_context(repo, "u1", MentorRequest(
            context_key="fitness_scheduling", clarifications=[ClarificationTurn(**TURN)]))

        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            _request_id="req", status="completed", output=[], output_text=guidance().model_dump_json(),
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2,
                                  input_tokens_details=SimpleNamespace(cached_tokens=0),
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=0)))
        settings = Settings(api_key=SecretStr("test-not-real"), ledger_path=Path(tmp.name) / "usage.sqlite3")
        result = OpenAIMentorProvider(settings, client=client).generate_guidance(context)

        sent = json.dumps(client.responses.create.call_args.kwargs)
        self.assertIn("What time of day could you usually walk?", sent)
        self.assertIn("10 minutes", sent)
        self.assertIn("clarifications", SYSTEM_PROMPT)
        self.assertEqual(PROMPT_VERSION, "mentor-guidance-2")
        self.assertEqual(result.telemetry.prompt_version, PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()

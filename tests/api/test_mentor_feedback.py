"""Tests for GET /users/{user_id}/mentor-feedback -- the read-only mentor
feedback endpoint (src/api/app.py + src/mentor/feedback.py).

Confirms it is a read: opens the DB read-only, never mutates it, 503s on a
missing DB, and is GET-only. Behaviour depth lives in
tests/mentor/test_feedback.py; this only checks the HTTP surface.
"""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore", message=r".*httpx.*starlette\.testclient.*")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

from tests.mentor._helpers import seed_belief  # noqa: E402


def _seed(db_path: str) -> None:
    repo = Repository.at_path(db_path)
    try:
        seed_belief(repo, user_id="u1", belief_id="b1", support=3)
        seed_belief(
            repo, user_id="u_high", belief_id="bh", support=8, strength=0.95,
            belief_type="routine_or_preference", belief_key="prefers_short_focused_sessions",
        )
    finally:
        repo.close()


class MentorFeedbackRouteTests(unittest.TestCase):
    def test_ready_feedback_for_a_grounded_user(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            _seed(db)
            r = TestClient(create_app(db)).get("/users/u1/mentor-feedback")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["feedback"])
        item = body["feedback"][0]
        self.assertEqual(item["grounded_in_belief_ids"], ["b1"])
        self.assertIn("confidence", item)
        self.assertIn(item["risk_tier"], ("low", "medium", "high"))
        self.assertIsNone(body["needs_more_data_reason"])

    def test_needs_more_data_for_an_unknown_user(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            _seed(db)
            r = TestClient(create_app(db)).get("/users/nobody/mentor-feedback")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "needs_more_data")
        self.assertEqual(body["feedback"], [])
        self.assertTrue(body["needs_more_data_reason"])

    def test_context_key_query_param_is_honoured(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            _seed(db)
            client = TestClient(create_app(db))
            low = client.get("/users/u1/mentor-feedback?context_key=fitness_scheduling").json()
            high = client.get("/users/u_high/mentor-feedback?context_key=mental_health_support").json()

        self.assertEqual(low["feedback"][0]["risk_tier"], "low")
        self.assertIsNotNone(low["feedback"][0]["recommended_next_action"])

        self.assertEqual(high["status"], "ready")
        self.assertEqual(high["feedback"][0]["risk_tier"], "high")
        self.assertIsNone(high["feedback"][0]["recommended_next_action"])

    def test_the_request_does_not_mutate_the_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            _seed(db)
            before = Path(db).read_bytes()
            client = TestClient(create_app(db))
            client.get("/users/u1/mentor-feedback")
            client.get("/users/u1/mentor-feedback?context_key=fitness_scheduling")
            client.get("/users/nobody/mentor-feedback")
            self.assertEqual(Path(db).read_bytes(), before)

    def test_missing_database_is_503(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.sqlite3")
            r = TestClient(create_app(missing, init_db=False)).get("/users/u1/mentor-feedback")
            self.assertEqual(r.status_code, 503)
            self.assertFalse(Path(missing).exists())

    def test_route_is_get_only(self) -> None:
        app = create_app(":memory:", init_db=False)
        methods = {
            m
            for route in app.routes
            if getattr(route, "path", None) == "/users/{user_id}/mentor-feedback"
            for m in getattr(route, "methods", set())
        }
        self.assertEqual(methods, {"GET"})

        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            _seed(db)
            client = TestClient(create_app(db))
            self.assertEqual(client.post("/users/u1/mentor-feedback", json={}).status_code, 405)
            self.assertEqual(client.delete("/users/u1/mentor-feedback").status_code, 405)

    def test_route_is_not_part_of_the_write_surface(self) -> None:
        app = create_app(":memory:", init_db=False)
        post_paths = {
            getattr(route, "path", "")
            for route in app.routes
            for m in getattr(route, "methods", set())
            if m == "POST"
        }
        self.assertNotIn("/users/{user_id}/mentor-feedback", post_paths)


if __name__ == "__main__":
    unittest.main()

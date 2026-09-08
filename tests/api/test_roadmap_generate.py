"""Tests for POST /roadmaps/generate -- the stateless roadmap bridge for the
Better You app (src/api/roadmap.py + the route in src/api/app.py).

Proves the bridge contract and its safety constraints:

- request validation: the payload is exactly {goalCategory, goalTitle};
  anything else (userId, profile, checkIns, notes, a full Goal) is a 422,
  rejected not ignored;
- the response is RoadmapDraft-compatible and carries no id, user id,
  status, or persistence field;
- the endpoint opens no database, does not create one, and mutates nothing;
- it creates no belief / evidence / recommendation / outcome;
- route visibility: POST only, present in the app's routes.
"""
from __future__ import annotations

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

# fastapi.testclient prints a Starlette/httpx deprecation notice on import.
warnings.filterwarnings("ignore", message=r".*httpx.*starlette\.testclient.*")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.storage.repository import Repository  # noqa: E402

_VALID = {"goalCategory": "fitness", "goalTitle": "Run a 10k without stopping"}

# Better You's own validateRoadmapDraft() limits (roadmapValidation.ts).
_MAX_MILESTONES = 10
_MAX_STEPS = 10
_MAX_TITLE = 200
_MAX_DESC = 1000


def _client() -> TestClient:
    # A db path that does not exist + init_db=False: the route must still work,
    # proving it needs no database.
    return TestClient(create_app("no_such_db_for_roadmap_bridge.sqlite3", init_db=False))


class RequestValidationTests(unittest.TestCase):
    def test_valid_minimal_payload_succeeds(self) -> None:
        r = _client().post("/roadmaps/generate", json=_VALID)
        self.assertEqual(r.status_code, 200, r.text)

    def test_missing_goal_title_is_422(self) -> None:
        self.assertEqual(
            _client().post("/roadmaps/generate", json={"goalCategory": "career"}).status_code, 422
        )

    def test_missing_goal_category_is_422(self) -> None:
        self.assertEqual(
            _client().post("/roadmaps/generate", json={"goalTitle": "Learn Spanish"}).status_code, 422
        )

    def test_empty_or_blank_strings_are_422(self) -> None:
        for bad in (
            {"goalCategory": "", "goalTitle": "x"},
            {"goalCategory": "career", "goalTitle": ""},
            {"goalCategory": "career", "goalTitle": "   "},
            {"goalCategory": "  ", "goalTitle": "x"},
        ):
            self.assertEqual(_client().post("/roadmaps/generate", json=bad).status_code, 422)

    def test_wrong_types_are_422(self) -> None:
        for bad in (
            {"goalCategory": "career", "goalTitle": 5},
            {"goalCategory": ["career"], "goalTitle": "x"},
            {"goalCategory": "career", "goalTitle": {"nested": "x"}},
        ):
            self.assertEqual(_client().post("/roadmaps/generate", json=bad).status_code, 422)

    def test_over_length_title_is_422(self) -> None:
        r = _client().post(
            "/roadmaps/generate", json={"goalCategory": "career", "goalTitle": "x" * 201}
        )
        self.assertEqual(r.status_code, 422)

    def test_unknown_category_is_still_accepted(self) -> None:
        # The bridge contract types goalCategory as a string; a category
        # Better You adds later must fall back to a generic frame, not 4xx/5xx.
        r = _client().post(
            "/roadmaps/generate", json={"goalCategory": "spirituality", "goalTitle": "Meditate daily"}
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreaterEqual(len(r.json()["milestones"]), 1)

    def test_get_body_is_not_json_is_400_or_422(self) -> None:
        r = _client().post(
            "/roadmaps/generate", content=b"not json", headers={"content-type": "application/json"}
        )
        self.assertIn(r.status_code, (400, 422))


class NoPrivateFieldDependencyTests(unittest.TestCase):
    def test_better_you_private_fields_are_rejected_not_ignored(self) -> None:
        for banned in (
            "userId", "user_id", "profile", "checkIns", "notes", "description",
            "goal", "id", "status", "createdAt", "updatedAt", "suggestedGoalId", "source",
        ):
            r = _client().post("/roadmaps/generate", json={**_VALID, banned: "whatever"})
            self.assertEqual(r.status_code, 422, f"{banned!r} should be rejected, got {r.status_code}")

    def test_a_full_goal_object_is_rejected(self) -> None:
        full_goal = {
            "id": "g1", "userId": "u1", "title": "Get promoted",
            "description": "private context the model should never see",
            "category": "career", "source": "custom", "status": "active",
            "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
        }
        self.assertEqual(_client().post("/roadmaps/generate", json=full_goal).status_code, 422)

    def test_the_two_allowed_fields_are_sufficient(self) -> None:
        self.assertEqual(_client().post("/roadmaps/generate", json=_VALID).status_code, 200)


class ResponseShapeTests(unittest.TestCase):
    def _draft(self) -> dict:
        r = _client().post("/roadmaps/generate", json=_VALID)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_response_is_roadmapdraft_compatible(self) -> None:
        draft = self._draft()
        self.assertEqual(list(draft.keys()), ["milestones"])
        self.assertIsInstance(draft["milestones"], list)
        self.assertGreaterEqual(len(draft["milestones"]), 1)
        self.assertLessEqual(len(draft["milestones"]), _MAX_MILESTONES)

        for milestone in draft["milestones"]:
            self.assertEqual(set(milestone.keys()), {"title", "description", "actionSteps"})
            self.assertIsInstance(milestone["title"], str)
            self.assertTrue(milestone["title"].strip())
            self.assertLessEqual(len(milestone["title"]), _MAX_TITLE)
            self.assertIsInstance(milestone["description"], str)
            self.assertLessEqual(len(milestone["description"]), _MAX_DESC)

            self.assertIsInstance(milestone["actionSteps"], list)
            self.assertGreaterEqual(len(milestone["actionSteps"]), 1)
            self.assertLessEqual(len(milestone["actionSteps"]), _MAX_STEPS)
            for step in milestone["actionSteps"]:
                self.assertEqual(set(step.keys()), {"title", "description"})
                self.assertIsInstance(step["title"], str)
                self.assertTrue(step["title"].strip())
                self.assertLessEqual(len(step["title"]), _MAX_TITLE)
                self.assertIsInstance(step["description"], str)
                self.assertLessEqual(len(step["description"]), _MAX_DESC)

    def test_response_carries_no_ids_or_persistence_fields(self) -> None:
        blob = json.dumps(self._draft())
        for banned in (
            '"id"', '"userId"', '"user_id"', '"goalId"', '"goal_id"', '"status"',
            '"createdAt"', '"updatedAt"', '"source"',
        ):
            self.assertNotIn(banned, blob)

    def test_descriptions_are_strings_never_null(self) -> None:
        # Better You's validateRoadmapDraft() throws on a null description
        # (it only accepts an omitted key or a string).
        for milestone in self._draft()["milestones"]:
            self.assertIsInstance(milestone["description"], str)
            for step in milestone["actionSteps"]:
                self.assertIsInstance(step["description"], str)

    def test_a_very_long_goal_title_stays_within_the_title_cap(self) -> None:
        r = _client().post(
            "/roadmaps/generate", json={"goalCategory": "career", "goalTitle": "y" * 200}
        )
        self.assertEqual(r.status_code, 200)
        for milestone in r.json()["milestones"]:
            self.assertLessEqual(len(milestone["title"]), _MAX_TITLE)

    def test_generation_is_deterministic(self) -> None:
        self.assertEqual(self._draft(), self._draft())

    def test_goal_title_influences_the_output(self) -> None:
        client = _client()
        one = client.post(
            "/roadmaps/generate", json={"goalCategory": "career", "goalTitle": "Alpha goal"}
        ).json()
        two = client.post(
            "/roadmaps/generate", json={"goalCategory": "career", "goalTitle": "Beta goal"}
        ).json()
        self.assertNotEqual(one, two)


class StatelessTests(unittest.TestCase):
    def test_no_database_is_required_or_created(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.sqlite3")
            client = TestClient(create_app(missing, init_db=False))
            r = client.post("/roadmaps/generate", json=_VALID)
            self.assertEqual(r.status_code, 200, r.text)  # not 503, unlike the DB-backed routes
            self.assertFalse(Path(missing).exists())

    def test_a_call_does_not_mutate_an_existing_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            Repository.at_path(db).close()
            before = Path(db).read_bytes()

            client = TestClient(create_app(db, init_db=True))
            for _ in range(3):
                self.assertEqual(
                    client.post("/roadmaps/generate", json=_VALID).status_code, 200
                )

            self.assertEqual(Path(db).read_bytes(), before)

    def test_endpoint_creates_no_beliefs_evidence_recommendations_or_outcomes(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = TestClient(create_app(db, init_db=True))
            for _ in range(3):
                client.post("/roadmaps/generate", json=_VALID)

            repo = Repository.readonly_at_path(db)
            try:
                self.assertEqual(repo.list_latest_beliefs(), [])
                self.assertEqual(repo.list_all_evidence(), [])
                self.assertEqual(repo.list_recommendations(), [])
                self.assertEqual(repo.list_recommendation_outcomes(), [])
                self.assertEqual(repo.list_outcome_learning_signals(), [])
                self.assertEqual(repo.list_events(user_id=None), [])
            finally:
                repo.close()


class RouteVisibilityTests(unittest.TestCase):
    def test_route_is_registered_as_post_only(self) -> None:
        app = create_app(":memory:", init_db=False)
        methods = {
            method
            for route in app.routes
            if getattr(route, "path", None) == "/roadmaps/generate"
            for method in getattr(route, "methods", set())
        }
        self.assertEqual(methods, {"POST"})

    def test_other_methods_on_the_route_are_405(self) -> None:
        client = _client()
        self.assertEqual(client.get("/roadmaps/generate").status_code, 405)
        self.assertEqual(client.put("/roadmaps/generate", json=_VALID).status_code, 405)
        self.assertEqual(client.delete("/roadmaps/generate").status_code, 405)

    def test_the_route_does_not_widen_the_promotion_surface(self) -> None:
        app = create_app(":memory:", init_db=False)
        for route in app.routes:
            path = getattr(route, "path", "")
            self.assertNotIn("promote", path)
            self.assertNotIn("approve", path)
            self.assertNotIn("reject", path)


if __name__ == "__main__":
    unittest.main()

"""Tests for the local internal-alpha API (src/api/app.py + service.py).

Uses fastapi.testclient (in-process; no socket). Proves the safety
boundaries the API must not weaken:

- GET /health works and reports DB presence.
- Read endpoints never mutate the database file.
- POST /events persists a valid, fully-versioned event.
- Invalid / extra-field payloads return 4xx (never a silent accept).
- The recommendation endpoint goes through the real context/risk policy and
  never uses a locked belief.
- No route promotes an outcome-learning signal or resolves a review, and
  recording outcomes never triggers learning.
- No endpoint can set a backend-owned field (version tags, evidence
  aggregation state, belief confidence/status/lock).
- A missing database is a 503, never silently created (unless init_db).
"""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

# fastapi.testclient emits a Starlette deprecation notice about httpx on import;
# it is cosmetic and unrelated to what these tests check.
warnings.filterwarnings("ignore", message=r".*httpx.*starlette\.testclient.*")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.storage.repository import Repository  # noqa: E402


def _client(db_path: str, *, init_db: bool = True) -> TestClient:
    return TestClient(create_app(db_path, init_db=init_db))


def _seed_user_with_belief(
    client: TestClient, *, user_id: str, belief_id: str, suffix: str, belief_key: str
) -> None:
    """event -> observation -> belief_evidence -> recompute, all via the API,
    so the belief exists as ``candidate`` for ``user_id``."""
    assert client.post(
        "/events",
        json={"user_id": user_id, "event_id": f"evt_{suffix}", "event_type": "goal_completed",
              "source": "app", "timestamp": "2026-01-01T12:00:00Z"},
    ).status_code == 201
    assert client.post(
        "/observations",
        json={"observation_id": f"obs_{suffix}", "user_id": user_id,
              "event_ids": [f"evt_{suffix}"], "category": "routine", "observation": "did a workout",
              "importance": 0.6, "confidence": 0.6, "created_at": "2026-01-01T12:00:00Z"},
    ).status_code == 201
    assert client.post(
        "/belief-evidence",
        json={"evidence_id": f"bev_{suffix}", "observation_id": f"obs_{suffix}",
              "belief_id": belief_id, "belief_type": "behavioral_tendency", "direction": "support",
              "source_type": "recorded_event", "context_key": "fitness", "strength": 0.9,
              "model_version": "api-test", "created_at": "2026-01-01T12:00:00Z"},
    ).status_code == 201
    assert client.post(
        f"/beliefs/{belief_id}/recompute",
        json={"user_id": user_id, "belief_type": "behavioral_tendency", "belief_key": belief_key,
              "belief_value": True, "as_of": "2026-02-01T12:00:00Z"},
    ).status_code == 201


class HealthTests(unittest.TestCase):
    def test_health_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            r = _client(db).get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["db_exists"])
        self.assertIn("api_version", body)

    def test_health_reports_a_missing_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "missing.sqlite3")
            r = _client(db, init_db=False).get("/health")
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json()["db_exists"])
            self.assertFalse(Path(db).exists())  # health never creates it


class MissingDatabaseTests(unittest.TestCase):
    def test_read_on_missing_db_is_503_and_creates_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "missing.sqlite3")
            client = _client(db, init_db=False)
            self.assertEqual(client.get("/users/u1/model").status_code, 503)
            self.assertEqual(client.get("/users/u1/reviews").status_code, 503)
            self.assertFalse(Path(db).exists())

    def test_write_on_missing_db_is_503_and_creates_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "missing.sqlite3")
            client = _client(db, init_db=False)
            r = client.post(
                "/events",
                json={"user_id": "u", "event_id": "e", "event_type": "t", "source": "s"},
            )
            self.assertEqual(r.status_code, 503)
            self.assertFalse(Path(db).exists())

    def test_init_db_creates_the_file_once(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "fresh.sqlite3")
            _client(db, init_db=True)
            self.assertTrue(Path(db).is_file())


class ReadEndpointsAreReadOnlyTests(unittest.TestCase):
    def test_reads_do_not_mutate_the_database_file(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(
                client, user_id="u1", belief_id="b1", suffix="1",
                belief_key="higher_adherence_after_work",
            )
            before = Path(db).read_bytes()

            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/users/u1/model").status_code, 200)
            self.assertEqual(client.get("/users/u1/reviews").status_code, 200)
            self.assertEqual(client.get("/evals").status_code, 200)

            self.assertEqual(Path(db).read_bytes(), before)

    def test_user_model_is_scoped_to_the_user(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            _seed_user_with_belief(client, user_id="u2", belief_id="b2", suffix="2",
                                   belief_key="prefers_evening_exercise_sessions")

            model = client.get("/users/u1/model").json()
        event_ids = {e["event_id"] for e in model["events"]}
        self.assertEqual(event_ids, {"evt_1"})


class PostEventsTests(unittest.TestCase):
    def test_valid_event_persists_with_backend_version_tags(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            r = client.post(
                "/events",
                json={"user_id": "u1", "event_id": "e1", "event_type": "goal_completed",
                      "source": "app"},
            )
            self.assertEqual(r.status_code, 201)
            body = r.json()
            self.assertEqual(body["event_id"], "e1")
            # the backend injected the version tags the client cannot set
            self.assertEqual(body["schema_version"], "6")
            self.assertEqual(body["scoring_version"], "belief-score-0.6")

            repo = Repository.readonly_at_path(db)
            try:
                stored = repo.get_event("e1")
            finally:
                repo.close()
            self.assertIsNotNone(stored)
            self.assertEqual(stored.user_id, "u1")

    def test_missing_required_field_is_422(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            r = _client(db).post("/events", json={"user_id": "u1", "event_id": "e1"})
        self.assertEqual(r.status_code, 422)

    def test_empty_string_id_is_422(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            r = _client(db).post(
                "/events",
                json={"user_id": "u1", "event_id": "", "event_type": "t", "source": "s"},
            )
        self.assertEqual(r.status_code, 422)

    def test_duplicate_event_id_is_409(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            payload = {"user_id": "u1", "event_id": "e1", "event_type": "t", "source": "s"}
            self.assertEqual(client.post("/events", json=payload).status_code, 201)
            self.assertEqual(client.post("/events", json=payload).status_code, 409)

    def test_unknown_event_for_observation_is_404(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            r = _client(db).post(
                "/observations",
                json={"observation_id": "o1", "user_id": "u1", "event_ids": ["ghost"],
                      "category": "routine", "observation": "x", "importance": 0.5, "confidence": 0.5},
            )
        self.assertEqual(r.status_code, 404)


class BackendOwnedFieldsAreRejectedTests(unittest.TestCase):
    def test_event_payload_cannot_carry_a_version_tag(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            r = _client(db).post(
                "/events",
                json={"user_id": "u1", "event_id": "e1", "event_type": "t", "source": "s",
                      "schema_version": "HACKED"},
            )
        self.assertEqual(r.status_code, 422)

    def test_belief_evidence_payload_cannot_set_aggregation_or_independence_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            for banned in ("authorized_aggregation_mode", "independence_group",
                           "aggregation_authorized_by", "backend_validation_passed",
                           "source_reliability"):
                r = client.post(
                    "/belief-evidence",
                    json={"evidence_id": "bevX", "observation_id": "obs_1", "belief_id": "b1",
                          "belief_type": "behavioral_tendency", "direction": "support",
                          "source_type": "recorded_event", "context_key": "fitness",
                          "strength": 0.9, "model_version": "api-test", banned: "x"},
                )
                self.assertEqual(r.status_code, 422, f"{banned} should be rejected")

    def test_authorized_evidence_stays_leaf_default(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            repo = Repository.readonly_at_path(db)
            try:
                evidence = repo.list_evidence(user_id="u1", belief_id="b1")
            finally:
                repo.close()
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].authorized_aggregation_mode.value, "leaf_default")

    def test_recompute_payload_cannot_set_confidence_or_lock(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            for banned in ("confidence", "status", "locked_until_recompute"):
                r = client.post(
                    "/beliefs/b1/recompute",
                    json={"user_id": "u1", "belief_type": "behavioral_tendency",
                          "belief_key": "higher_adherence_after_work", "belief_value": True,
                          banned: 0.99 if banned == "confidence" else "x"},
                )
                self.assertEqual(r.status_code, 422, f"{banned} should be rejected")


class RecommendationPolicyTests(unittest.TestCase):
    def _seed_two_beliefs_one_locked(self, client: TestClient) -> None:
        _seed_user_with_belief(client, user_id="u1", belief_id="b_ok", suffix="ok",
                               belief_key="higher_adherence_after_work")
        _seed_user_with_belief(client, user_id="u1", belief_id="b_locked", suffix="lk",
                               belief_key="prefers_evening_exercise_sessions")

    def test_locked_belief_is_excluded_and_recorded_as_blocked(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            self._seed_two_beliefs_one_locked(client)

            repo = Repository.at_path(db)
            try:
                self.assertTrue(
                    repo.lock_belief_until_recompute(user_id="u1", belief_id="b_locked")
                )
            finally:
                repo.close()

            body = client.post(
                "/recommendations",
                json={"recommendation_id": "r1", "user_id": "u1",
                      "context_key": "fitness_scheduling"},
            ).json()

        self.assertNotIn("b_locked", body["belief_ids_used"])
        self.assertIn("b_ok", body["belief_ids_used"])
        blocked = {b["belief_id"]: b["reason"] for b in body["blocked_beliefs"]}
        self.assertEqual(blocked.get("b_locked"), "blocked_locked_until_recompute")

    def test_unknown_context_resolves_to_high_risk_and_blocks_all_beliefs(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b_ok", suffix="ok",
                                   belief_key="higher_adherence_after_work")

            body = client.post(
                "/recommendations",
                json={"recommendation_id": "r1", "user_id": "u1",
                      "context_key": "totally_unknown_context_zzz"},
            ).json()

        self.assertEqual(body["risk_tier"], "high")
        self.assertTrue(body["review_required"])
        self.assertEqual(body["belief_ids_used"], [])


class NoAutomaticPromotionTests(unittest.TestCase):
    def test_no_route_promotes_or_resolves_reviews(self) -> None:
        app = create_app(":memory:", init_db=False)
        paths = {route.path for route in app.routes}
        for path in paths:
            self.assertNotIn("promote", path)
            self.assertNotIn("approve", path)
            self.assertNotIn("reject", path)
        # the write surface is exactly the six documented endpoints
        write_methods = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
            if method == "POST"
        }
        self.assertEqual(
            {p for p, _ in write_methods},
            {"/events", "/observations", "/belief-evidence",
             "/beliefs/{belief_id}/recompute", "/recommendations",
             "/recommendation-outcomes"},
        )

    def test_recording_outcomes_never_creates_a_learning_signal(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")

            for i in range(1, 6):
                self.assertEqual(
                    client.post(
                        "/recommendations",
                        json={"recommendation_id": f"rc{i}", "user_id": "u1",
                              "context_key": "fitness_scheduling"},
                    ).status_code, 201,
                )
                self.assertEqual(
                    client.post(
                        "/recommendation-outcomes",
                        json={"outcome_id": f"o{i}", "recommendation_id": f"rc{i}",
                              "followed": "followed", "result": "successful", "source": "app_event"},
                    ).status_code, 201,
                )

            repo = Repository.readonly_at_path(db)
            try:
                signals = repo.list_outcome_learning_signals()
                rps = [
                    e for e in repo.list_all_evidence(user_id="u1")
                    if e.source_type.value == "repeated_pattern_summary"
                ]
            finally:
                repo.close()

        self.assertEqual(signals, [])
        self.assertEqual(rps, [])

    def test_reviews_endpoint_is_read_only_and_says_so(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            body = client.get("/users/u1/reviews").json()
        self.assertIn("note", body)
        self.assertIn("CLI", body["note"])
        # there is no PUT/PATCH/DELETE on the reviews path
        app = create_app(db)
        methods = {
            method
            for route in app.routes
            if getattr(route, "path", None) == "/users/{user_id}/reviews"
            for method in getattr(route, "methods", set())
        }
        self.assertEqual(methods, {"GET"})


class EvalsEndpointTests(unittest.TestCase):
    def test_evals_runs_the_harness_without_touching_the_db(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "api.sqlite3")
            client = _client(db)
            _seed_user_with_belief(client, user_id="u1", belief_id="b1", suffix="1",
                                   belief_key="higher_adherence_after_work")
            before = Path(db).read_bytes()
            body = client.get("/evals").json()
            self.assertEqual(Path(db).read_bytes(), before)
        self.assertTrue(body["passed"])
        self.assertEqual(body["summary"]["failed"], 0)
        self.assertGreaterEqual(body["summary"]["scenarios"], 8)


if __name__ == "__main__":
    unittest.main()

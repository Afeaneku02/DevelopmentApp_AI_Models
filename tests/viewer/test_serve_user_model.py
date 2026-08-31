"""Tests for tools/serve_user_model.py -- the read-only local web server for
the adaptive user model viewer.

Runs a real ``ThreadingHTTPServer`` on an ephemeral port in a background
thread and drives it with ``urllib``. Deliberately minimal (no framework, no
UI assertions): that HTML is served from a real DB, that a missing DB fails
clearly without crashing the server, that requests never modify the DB, and
that the optional query params scope the page.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.events.models import UserEvent
from src.storage.repository import Repository
from tools.serve_user_model import render_evals_page, render_page, serve

_QUICK_SCENARIO = {
    "name": "quick_smoke",
    "description": "trivial supporting-evidence scenario for fast route tests",
    "steps": [
        {"op": "evidence", "id": "s1", "belief": "b", "direction": "support", "strength": 0.9,
         "source_type": "recorded_event", "days_before_as_of": 10},
        {"op": "recompute", "belief": "b", "belief_key": "higher_adherence_after_work",
         "belief_value": True},
    ],
    "expect": [{"check": "belief_confidence", "belief": "b", "gt": 0.0}],
}

VERSION_FIELDS = dict(
    schema_version="6", scoring_version="belief-score-0.6",
    canonicalizer_version="canon-0.6", policy_version="policy-0.6",
)
AS_OF = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)


def _event(event_id: str, user_id: str) -> UserEvent:
    return UserEvent(
        event_id=event_id, user_id=user_id, event_type="goal_completed",
        timestamp=AS_OF - timedelta(days=1), source="app", **VERSION_FIELDS,
    )


def _seed_db(path: str, users: dict[str, str]) -> None:
    repo = Repository.at_path(path)
    try:
        for event_id, user_id in users.items():
            repo.insert_event(_event(event_id, user_id))
    finally:
        repo.close()


@contextmanager
def _running_server(db_path: str, *, evals_dir: str | None = None):
    httpd = serve(db_path, host="127.0.0.1", port=0, evals_dir=evals_dir)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(url: str, *, method: str = "GET", data: bytes | None = None) -> tuple[int, str]:
    """One HTTP request, retrying transient socket aborts (WinError 10053
    under load is not a real failure). Returns ``(status_code, body)``;
    an HTTP error status (e.g. 405) is a normal return, not an exception."""
    request = urllib.request.Request(url, method=method, data=data)
    last_error: OSError | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            exc.close()
            return exc.code, body
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError(f"{method} {url} failed after retries: {last_error!r}")


def _get(url: str) -> tuple[int, str]:
    return _request(url)


class ServesHtmlFromARealDatabaseTests(unittest.TestCase):
    def test_root_returns_the_rendered_viewer_page(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})

            with _running_server(db_path) as base:
                status, body = _get(base + "/")

        self.assertEqual(status, 200)
        self.assertIn("<!doctype html>", body.lstrip().lower())
        self.assertIn("READ-ONLY", body)
        self.assertIn("evt_1", body)

    def test_every_request_re_reads_the_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})

            with _running_server(db_path) as base:
                _, first = _get(base + "/")
                self.assertNotIn("evt_2", first)

                # Add a row through a separate writable connection while the
                # server is running; the next request must reflect it.
                _seed_db(db_path, {"evt_2": "usr_1"})
                _, second = _get(base + "/")

        self.assertIn("evt_2", second)


class MissingDatabaseTests(unittest.TestCase):
    def test_render_page_reports_a_missing_db_as_503_without_raising(self) -> None:
        status, content_type, body = render_page("does_not_exist_anywhere.sqlite3")
        self.assertEqual(status, 503)
        self.assertIn("text/plain", content_type)
        self.assertIn("does_not_exist_anywhere.sqlite3", body)

    def test_server_stays_up_and_returns_503_for_a_missing_db(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.sqlite3")
            with _running_server(missing) as base:
                status, body = _get(base + "/")
                self.assertEqual(status, 503)
                self.assertIn("unavailable", body)
                # still responsive on a second request
                status_again, _ = _get(base + "/")
                self.assertEqual(status_again, 503)
            self.assertFalse(Path(missing).exists())


class ReadOnlyRouteTests(unittest.TestCase):
    def test_a_request_does_not_modify_the_database_file(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            before = Path(db_path).read_bytes()

            with _running_server(db_path) as base:
                self.assertEqual(_get(base + "/")[0], 200)
                self.assertEqual(_get(base + "/?user_id=usr_1")[0], 200)

            self.assertEqual(Path(db_path).read_bytes(), before)

    def test_write_methods_are_rejected_with_405(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            before = Path(db_path).read_bytes()

            with _running_server(db_path) as base:
                status, _ = _request(base + "/", method="POST", data=b"{}")

            self.assertEqual(status, 405)
            self.assertEqual(Path(db_path).read_bytes(), before)


class QueryParamScopingTests(unittest.TestCase):
    def test_user_id_query_param_scopes_the_page(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_a": "usr_1", "evt_b": "usr_2"})

            with _running_server(db_path) as base:
                _, all_users = _get(base + "/")
                _, scoped = _get(base + "/?user_id=usr_1")

        self.assertIn("evt_a", all_users)
        self.assertIn("evt_b", all_users)
        self.assertIn("evt_a", scoped)
        self.assertNotIn("evt_b", scoped)

    def test_belief_id_query_param_is_accepted(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_a": "usr_1"})
            with _running_server(db_path) as base:
                status, _ = _get(base + "/?belief_id=bel_1")
        self.assertEqual(status, 200)


class EvalsRouteTests(unittest.TestCase):
    def _evals_dir(self, tmp: str, scenarios: dict[str, dict]) -> str:
        d = Path(tmp) / "evals"
        d.mkdir()
        for name, manifest in scenarios.items():
            (d / f"{name}.json").write_text(json.dumps(manifest), encoding="utf-8")
        return str(d)

    def test_render_evals_page_returns_html_without_touching_a_database(self) -> None:
        with TemporaryDirectory() as tmp:
            evals_dir = self._evals_dir(tmp, {"quick": _QUICK_SCENARIO})
            status, content_type, body = render_evals_page(evals_dir)
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("evaluation scorecard", body.lower())
        self.assertIn("quick_smoke", body)
        self.assertIn("ALL PASS", body)

    def test_evals_route_is_served_and_does_not_modify_the_database(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            before = Path(db_path).read_bytes()
            evals_dir = self._evals_dir(tmp, {"quick": _QUICK_SCENARIO})

            with _running_server(db_path, evals_dir=evals_dir) as base:
                status, body = _get(base + "/evals")
                # the normal viewer is still there and unchanged
                root_status, root_body = _get(base + "/")

            self.assertEqual(status, 200)
            self.assertIn("quick_smoke", body)
            self.assertIn("scenarios passed", body)
            self.assertEqual(root_status, 200)
            self.assertIn("READ-ONLY", root_body)
            self.assertNotIn("quick_smoke", root_body)
            self.assertEqual(Path(db_path).read_bytes(), before)

    def test_evals_route_survives_a_missing_manifest_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            missing = str(Path(tmp) / "no_such_evals_dir")

            with _running_server(db_path, evals_dir=missing) as base:
                status, body = _get(base + "/evals")
                # server still responsive afterwards
                again, _ = _get(base + "/evals")

        self.assertEqual(status, 200)
        self.assertIn("No evaluation manifests were found", body)
        self.assertEqual(again, 200)

    def test_evals_route_survives_a_broken_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            evals_dir = Path(tmp) / "evals"
            evals_dir.mkdir()
            (evals_dir / "broken.json").write_text("{ not valid json", encoding="utf-8")
            (evals_dir / "bad_step.json").write_text(
                json.dumps({"name": "bad", "steps": [{"op": "nope"}],
                            "expect": [{"check": "signal_exists", "context": "x"}]}),
                encoding="utf-8",
            )

            with _running_server(db_path, evals_dir=str(evals_dir)) as base:
                status, body = _get(base + "/evals")
                again, _ = _get(base + "/")

        self.assertEqual(status, 200)
        self.assertIn("FAIL", body)
        self.assertIn("could not load manifest", body)
        self.assertIn("could not run this scenario", body)
        self.assertEqual(again, 200)

    def test_both_routes_serve_cross_linking_nav(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            evals_dir = self._evals_dir(tmp, {"quick": _QUICK_SCENARIO})

            with _running_server(db_path, evals_dir=evals_dir) as base:
                _, root = _get(base + "/")
                _, evals = _get(base + "/evals")

        for body in (root, evals):
            self.assertIn('<nav class="nav">', body)
            self.assertIn(">User Model</a>", body)
            self.assertIn(">Eval Scorecard</a>", body)
            self.assertIn('href="/"', body)
            self.assertIn('href="/evals"', body)
        self.assertIn('<a href="/" class="active">User Model</a>', root)
        self.assertIn('<a href="/evals" class="active">Eval Scorecard</a>', evals)

    def test_post_to_evals_is_rejected_with_405(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "events.sqlite3")
            _seed_db(db_path, {"evt_1": "usr_1"})
            with _running_server(db_path) as base:
                status, _ = _request(base + "/evals", method="POST", data=b"{}")
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()

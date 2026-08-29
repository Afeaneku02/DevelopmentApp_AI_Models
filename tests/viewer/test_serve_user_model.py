"""Tests for tools/serve_user_model.py -- the read-only local web server for
the adaptive user model viewer.

Runs a real ``ThreadingHTTPServer`` on an ephemeral port in a background
thread and drives it with ``urllib``. Deliberately minimal (no framework, no
UI assertions): that HTML is served from a real DB, that a missing DB fails
clearly without crashing the server, that requests never modify the DB, and
that the optional query params scope the page.
"""
from __future__ import annotations

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
from tools.serve_user_model import render_page, serve

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
def _running_server(db_path: str):
    httpd = serve(db_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, str]:
    # A transient socket abort (WinError 10053 under load) is not a real
    # failure of a read-only GET -- retry a couple of times before giving up.
    last_error: OSError | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            exc.close()
            return exc.code, body
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError(f"GET {url} failed after retries: {last_error!r}")


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
                request = urllib.request.Request(base + "/", method="POST", data=b"{}")
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        status = response.status
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    exc.read()
                    exc.close()

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


if __name__ == "__main__":
    unittest.main()

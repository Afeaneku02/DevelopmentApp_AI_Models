"""Regression: on-disk Repository connections must be usable from a thread
other than the one that opened them. FastAPI opens the per-request
Repository in a sync generator dependency (one threadpool thread) and runs
the sync endpoint in another; with sqlite3's default ``check_same_thread``
that intermittently 500'd POST /events, /users/{id}/process and
/users/{id}/mentor-guidance when Better You drove the API for real.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from src.storage.repository import Repository


def _use_in_other_thread(fn) -> BaseException | None:
    errors: list[BaseException] = []

    def run():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return errors[0] if errors else None


class RepositoryCrossThreadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "t.sqlite3")
        Repository.at_path(self.db_path).close()

    def tearDown(self):
        self._tmp.cleanup()

    def test_writable_repository_used_from_another_thread(self):
        repo = Repository.at_path(self.db_path)
        try:
            self.assertIsNone(_use_in_other_thread(lambda: repo.list_events(user_id="u1")))
        finally:
            repo.close()

    def test_readonly_repository_used_from_another_thread(self):
        repo = Repository.readonly_at_path(self.db_path)
        try:
            self.assertIsNone(_use_in_other_thread(lambda: repo.list_latest_beliefs(user_id="u1")))
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()

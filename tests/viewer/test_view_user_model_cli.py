"""Thin smoke tests for tools/view_user_model.py -- the read-only viewer CLI.

Deliberately minimal (requirement: don't overbuild UI tests): just that the
CLI writes an HTML file, that ``--demo`` seeds and views a throwaway
database, and that it never writes to the database it is pointed at.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.storage.repository import Repository

_VIEW_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "view_user_model.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_VIEW_SCRIPT), *args], capture_output=True, text=True)


class ViewUserModelCliTests(unittest.TestCase):
    def test_demo_flag_seeds_and_writes_a_page_without_opening_a_browser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "view.html"
            result = _run(["--demo", "--no-open", "--out", str(out)])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file())
            page = out.read_text(encoding="utf-8")
            self.assertIn("READ-ONLY", page)
            self.assertIn("higher_adherence_after_work", page)

    def test_missing_database_fails_and_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.sqlite3"
            result = _run(["--db", str(missing), "--no-open"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(missing.exists())

    def test_viewing_a_real_database_does_not_modify_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "events.sqlite3"
            repo = Repository.at_path(str(db_path))
            repo.close()
            before = db_path.read_bytes()

            out = Path(tmp) / "view.html"
            result = _run(["--db", str(db_path), "--no-open", "--out", str(out)])
            self.assertEqual(result.returncode, 0, result.stderr)

            self.assertEqual(db_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

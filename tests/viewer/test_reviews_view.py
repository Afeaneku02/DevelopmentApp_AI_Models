"""Tests for src/viewer/reviews_view.py -- the read-only manual review queue
page (the server's ``/reviews`` route).

Covers: an empty database renders every section empty; a pending
outcome-learning signal appears in "Pending review" with its proposed
evidence; a signal with an approved/rejected review is not listed as pending
and shows a status badge; the review trail is rendered; user scoping;
collecting the queue mutates nothing; manifest/DB text is HTML-escaped.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.recommendations.review import review_outcome_learning_signal
from src.storage.repository import Repository
from src.viewer.reviews_view import collect_review_queue, render_reviews_html
from tests.recommendations.test_promotion import _seed_model

REVIEWED_AT = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _seeded_repo() -> tuple[Repository, str, str]:
    """One pending signal (usr_pending) and one rejected-review signal
    (usr_reviewed). Returns ``(repo, pending_signal_id, reviewed_signal_id)``."""
    repo = Repository.in_memory()
    _, pending = _seed_model(repo, user_id="usr_pending", belief_id="bel_p", event_id="evt_p")
    _, reviewed = _seed_model(repo, user_id="usr_reviewed", belief_id="bel_r", event_id="evt_r")
    review_outcome_learning_signal(
        repo, review_id="rev_r", signal_id=reviewed.signal_id, reviewer_id="alice",
        decision="rejected", as_of=REVIEWED_AT, notes="self-report only",
    )
    return repo, pending.signal_id, reviewed.signal_id


class EmptyQueueTests(unittest.TestCase):
    def test_empty_database_renders_every_section_empty(self) -> None:
        repo = Repository.in_memory()
        try:
            queue = collect_review_queue(repo, db_path="empty.sqlite3")
        finally:
            repo.close()
        self.assertEqual(queue.pending_signals, [])
        self.assertEqual(queue.reviewed_signals, [])
        self.assertEqual(queue.reviews, [])

        page = render_reviews_html(queue)
        self.assertTrue(page.lstrip().startswith("<!doctype html>"))
        self.assertIn("READ-ONLY", page)
        self.assertIn("No outcome-learning signals are waiting for review.", page)
        self.assertIn("No signals have been reviewed yet.", page)
        self.assertIn("No review decisions recorded.", page)


class PendingSignalTests(unittest.TestCase):
    def test_pending_signal_appears_with_its_proposed_evidence(self) -> None:
        repo, pending_id, reviewed_id = _seeded_repo()
        try:
            queue = collect_review_queue(repo, db_path="demo.sqlite3")
        finally:
            repo.close()

        self.assertEqual([s["signal_id"] for s in queue.pending_signals], [pending_id])
        pending_row = queue.pending_signals[0]
        self.assertEqual(pending_row["review_status"], "pending")
        self.assertTrue(pending_row["proposed_evidence"])

        page = render_reviews_html(queue)
        self.assertIn("<h2>Pending review</h2>", page)
        self.assertIn("<h2>Proposed evidence (pending signals)</h2>", page)
        self.assertIn(pending_id, page)
        # the pending signal's proposed evidence names its belief
        self.assertIn("bel_p", page.split("Reviewed signals")[0])

    def test_pending_count_in_summary(self) -> None:
        repo, _, _ = _seeded_repo()
        try:
            queue = collect_review_queue(repo, db_path="demo.sqlite3")
        finally:
            repo.close()
        summary = queue.summary()
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["signals"], 2)


class ReviewedSignalTests(unittest.TestCase):
    def test_reviewed_signal_is_not_pending_and_carries_a_status_badge(self) -> None:
        repo, pending_id, reviewed_id = _seeded_repo()
        try:
            queue = collect_review_queue(repo, db_path="demo.sqlite3")
        finally:
            repo.close()

        pending_ids = {s["signal_id"] for s in queue.pending_signals}
        self.assertNotIn(reviewed_id, pending_ids)

        reviewed_ids = {s["signal_id"]: s["review_status"] for s in queue.reviewed_signals}
        self.assertEqual(reviewed_ids.get(reviewed_id), "rejected")

        page = render_reviews_html(queue)
        self.assertIn("<h2>Reviewed signals</h2>", page)
        self.assertIn('<span class="tag rejected">rejected</span>', page)
        # the reviewed signal must not appear in the pending block
        self.assertNotIn(reviewed_id, page.split("Reviewed signals")[0])
        # the review trail row is shown
        self.assertIn("rev_r", page)
        self.assertIn("<h2>Review decisions</h2>", page)

    def test_approved_review_shows_as_approved(self) -> None:
        repo = Repository.in_memory()
        try:
            _, signal = _seed_model(repo, user_id="u", belief_id="b", event_id="e")
            review_outcome_learning_signal(
                repo, review_id="rv1", signal_id=signal.signal_id, reviewer_id="bob",
                decision="approved", as_of=REVIEWED_AT,
            )
            queue = collect_review_queue(repo, db_path="demo.sqlite3")
        finally:
            repo.close()
        self.assertEqual(queue.pending_signals, [])
        self.assertEqual(queue.reviewed_signals[0]["review_status"], "approved")
        page = render_reviews_html(queue)
        self.assertIn('<span class="tag approved">approved</span>', page)


class ScopingAndSafetyTests(unittest.TestCase):
    def test_user_scope_limits_the_signals_shown(self) -> None:
        repo, pending_id, reviewed_id = _seeded_repo()
        try:
            queue = collect_review_queue(repo, db_path="demo.sqlite3", user_id="usr_pending")
        finally:
            repo.close()
        ids = {s["signal_id"] for s in queue.pending_signals + queue.reviewed_signals}
        self.assertEqual(ids, {pending_id})
        # a review for another user's signal must not leak in
        self.assertEqual(queue.reviews, [])

    def test_collecting_the_queue_mutates_nothing(self) -> None:
        repo, pending_id, reviewed_id = _seeded_repo()
        try:
            before_signals = len(repo.list_outcome_learning_signals())
            before_reviews = len(repo.list_outcome_learning_signal_reviews())
            before_evidence = len(repo.list_all_evidence())

            collect_review_queue(repo, db_path="demo.sqlite3")
            collect_review_queue(repo, db_path="demo.sqlite3", user_id="usr_pending")

            self.assertEqual(len(repo.list_outcome_learning_signals()), before_signals)
            self.assertEqual(len(repo.list_outcome_learning_signal_reviews()), before_reviews)
            self.assertEqual(len(repo.list_all_evidence()), before_evidence)
        finally:
            repo.close()

    def test_the_page_has_no_write_controls(self) -> None:
        repo, _, _ = _seeded_repo()
        try:
            page = render_reviews_html(collect_review_queue(repo, db_path="demo.sqlite3"))
        finally:
            repo.close()
        lowered = page.lower()
        for token in ("<form", "<button", "<input", "<script", "onclick=", 'method="post"'):
            self.assertNotIn(token, lowered)

    def test_nav_links_to_all_three_viewer_pages(self) -> None:
        repo = Repository.in_memory()
        try:
            page = render_reviews_html(collect_review_queue(repo, db_path="demo.sqlite3"))
        finally:
            repo.close()
        self.assertIn('<nav class="nav">', page)
        self.assertIn("User Model", page)
        self.assertIn("Eval Scorecard", page)
        self.assertIn("Review Queue", page)
        self.assertIn('href="/"', page)
        self.assertIn('href="/evals"', page)
        self.assertIn('href="/reviews"', page)
        self.assertIn('<a href="/reviews" class="active">Review Queue</a>', page)

    def test_db_path_is_escaped(self) -> None:
        repo = Repository.in_memory()
        try:
            queue = collect_review_queue(
                repo, db_path="<script>alert(1)</script>", user_id="<b>x</b>"
            )
        finally:
            repo.close()
        page = render_reviews_html(queue)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)


if __name__ == "__main__":
    unittest.main()

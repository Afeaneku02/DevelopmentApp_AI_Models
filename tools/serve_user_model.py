#!/usr/bin/env python3
"""Read-only local/dev web server for the adaptive user model viewer.

Serves the same page ``tools/view_user_model.py`` writes to a file, but live:
open http://localhost:8000 and every request re-reads the SQLite database and
re-renders, so there is no HTML file to regenerate by hand after running a
CLI or manifest.

Strictly read-only, the same three ways as ``view_user_model.py``:

- The database is opened per request via ``Repository.readonly_at_path(db)``
  -- SQLite URI ``mode=ro`` -- so any accidental write fails at the SQLite
  layer, and a missing/typoed path fails loudly rather than creating a file.
- Only ``src/viewer/user_model_view.collect_view_model`` +
  ``render_html`` are used; there is no code path here that recomputes,
  suppresses, invalidates, canonicalizes, or writes.
- Only ``GET`` / ``HEAD`` are handled. Every other method returns 405.

Run:
    python tools/serve_user_model.py --db events.sqlite3
    python tools/serve_user_model.py --db events.sqlite3 --host 0.0.0.0 --port 9001
    python tools/serve_user_model.py --demo

``--demo`` seeds one throwaway temp database from the shipped
``tools/manifests/canonicalized_after_work_workout.json`` (via the existing
``tools/run_manifest.py``) and serves that; the seeding runs once at startup,
not per request.

Optional query parameters on ``/``: ``?user_id=usr_17`` and
``?belief_id=bel_1`` scope the page exactly like the CLI's ``--user-id`` /
``--belief-id`` flags.

``/evals`` renders a read-only evaluation scorecard: the existing harness
(``src/evals/harness.py``) is run over the shipped scenario manifests
(``examples/evals/*.json``) -- each scenario in its own fresh in-memory
database -- and the page shows summary counts, every scenario's pass/fail,
and every check. This route never opens or touches the served database, and
a missing or broken manifest is shown as a failed scenario rather than
crashing the server. Pass ``--evals-dir`` to point at a different manifest
directory.

Stdlib only (``http.server``); the repo has no web framework.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.repository import Repository  # noqa: E402
from src.viewer.evals_view import (  # noqa: E402
    DEFAULT_MANIFEST_DIR,
    collect_eval_report,
    render_evals_html,
)
from src.viewer.user_model_view import collect_view_model, render_html  # noqa: E402
from tools.view_user_model import _seed_demo_database  # noqa: E402  (shared demo-seed helper)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the adaptive user model viewer over HTTP, re-reading the SQLite "
            "database on every request. Read-only: GET/HEAD only, no writes."
        ),
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to the SQLite database to serve. Required unless --demo is given.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Seed one throwaway temp database from the shipped demo manifest at startup, then serve it.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument(
        "--evals-dir", default=None, dest="evals_dir",
        help=(
            "Directory of scenario manifests for the /evals scorecard "
            f"(default: {DEFAULT_MANIFEST_DIR})."
        ),
    )
    return parser.parse_args(argv)


def render_page(
    db_path: str, *, user_id: str | None = None, belief_id: str | None = None
) -> tuple[int, str, str]:
    """Read the database read-only and render the viewer page.

    Returns ``(status_code, content_type, body)``. This never writes: it only
    opens ``Repository.readonly_at_path`` and calls the pure view helpers. A
    missing database is reported as ``503`` with a plain-text message naming
    the path, not raised, so the server stays up.
    """
    try:
        repo = Repository.readonly_at_path(db_path)
    except FileNotFoundError as exc:
        return 503, "text/plain; charset=utf-8", f"user model database unavailable: {exc}"

    try:
        view_model = collect_view_model(
            repo,
            db_path=db_path,
            user_id=user_id,
            belief_id=belief_id,
            generated_at=datetime.now(timezone.utc),
        )
    finally:
        repo.close()

    return 200, "text/html; charset=utf-8", render_html(view_model)


def render_evals_page(manifest_dir: str | Path | None = None) -> tuple[int, str, str]:
    """Run the evaluation harness over ``manifest_dir`` and render the
    scorecard page.

    Returns ``(status_code, content_type, body)``. Always ``200`` with an
    HTML body: a missing manifest directory renders an empty-state page and a
    broken manifest renders as a failed scenario -- neither raises, so the
    server stays up. This never opens the served database; every scenario
    runs in its own fresh in-memory ``Repository``.
    """
    report, resolved_dir = collect_eval_report(manifest_dir)
    body = render_evals_html(
        report, manifest_dir=resolved_dir, generated_at=datetime.now(timezone.utc)
    )
    return 200, "text/html; charset=utf-8", body


def build_handler(
    db_path: str, *, evals_dir: str | Path | None = None
) -> type[BaseHTTPRequestHandler]:
    """A ``BaseHTTPRequestHandler`` subclass bound to one database path.

    Serves the viewer at ``/`` (honouring ``?user_id=`` / ``?belief_id=``),
    the evaluation scorecard at ``/evals``, returns 204 for
    ``/favicon.ico``, 404 for anything else, and 405 for any method other
    than GET/HEAD. There is no route that writes.
    """
    resolved_evals_dir = Path(evals_dir) if evals_dir is not None else DEFAULT_MANIFEST_DIR

    class UserModelViewerHandler(BaseHTTPRequestHandler):
        server_version = "UserModelViewer/1.0"

        def _write(self, status: int, content_type: str, body: str, *, include_body: bool) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if include_body:
                self.wfile.write(encoded)

        def handle_one_request(self) -> None:
            # A client that hangs up mid-request/response (common under load
            # and in tests) raises a connection error here. That is not a
            # server fault and must not crash the worker thread -- just drop
            # the connection quietly.
            try:
                super().handle_one_request()
            except (ConnectionError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                self.close_connection = True

        def _handle(self, *, include_body: bool) -> None:
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"

            if route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            if route == "/evals":
                status, content_type, body = render_evals_page(resolved_evals_dir)
                self._write(status, content_type, body, include_body=include_body)
                return

            if route != "/":
                self._write(404, "text/plain; charset=utf-8", "not found", include_body=include_body)
                return

            params = parse_qs(parsed.query)
            user_id = params.get("user_id", [None])[0] or None
            belief_id = params.get("belief_id", [None])[0] or None

            status, content_type, body = render_page(db_path, user_id=user_id, belief_id=belief_id)
            self._write(status, content_type, body, include_body=include_body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            self._handle(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle(include_body=False)

        def _reject_write_method(self) -> None:
            self._write(
                405, "text/plain; charset=utf-8",
                "this viewer is read-only; only GET and HEAD are supported",
                include_body=True,
            )

        do_POST = do_PUT = do_PATCH = do_DELETE = _reject_write_method  # noqa: N815

        def log_message(self, fmt: str, *args) -> None:  # keep the default access log quiet-ish
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return UserModelViewerHandler


class _ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # A dropped client connection is not a server error worth a
        # traceback; anything else keeps the default behaviour.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(
    db_path: str, *, host: str, port: int, evals_dir: str | Path | None = None
) -> ThreadingHTTPServer:
    """Build (but do not block on) a server for ``db_path``. The caller runs
    ``serve_forever()``; tests bind port 0 and drive it from a thread."""
    handler = build_handler(db_path, evals_dir=evals_dir)
    return _ViewerHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.demo:
        if args.db is not None:
            print("--demo seeds its own throwaway database; do not also pass --db.", file=sys.stderr)
            return 1
        try:
            db_path = _seed_demo_database()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Seeded demo database: {db_path}", file=sys.stderr)
    else:
        if not args.db:
            print("--db is required (or pass --demo).", file=sys.stderr)
            return 1
        db_path = args.db
        if not Path(db_path).is_file():
            print(f"no such database file: {db_path!r}", file=sys.stderr)
            return 1

    httpd = serve(db_path, host=args.host, port=args.port, evals_dir=args.evals_dir)
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"Serving read-only user model viewer for {db_path!r} at http://{host}:{port}/  (Ctrl+C to stop)")
    print(f"  evaluation scorecard: http://{host}:{port}/evals", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

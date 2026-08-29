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


def build_handler(db_path: str) -> type[BaseHTTPRequestHandler]:
    """A ``BaseHTTPRequestHandler`` subclass bound to one database path.

    Serves the viewer at ``/`` (honouring ``?user_id=`` / ``?belief_id=``),
    returns 204 for ``/favicon.ico``, 404 for anything else, and 405 for any
    method other than GET/HEAD. There is no route that writes.
    """

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

        def _handle(self, *, include_body: bool) -> None:
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"

            if route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
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


def serve(db_path: str, *, host: str, port: int) -> ThreadingHTTPServer:
    """Build (but do not block on) a server for ``db_path``. The caller runs
    ``serve_forever()``; tests bind port 0 and drive it from a thread."""
    handler = build_handler(db_path)
    return ThreadingHTTPServer((host, port), handler)


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

    httpd = serve(db_path, host=args.host, port=args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"Serving read-only user model viewer for {db_path!r} at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

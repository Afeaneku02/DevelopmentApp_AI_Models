#!/usr/bin/env python3
"""Run the local internal-alpha adaptive-user-model HTTP API.

Wraps the existing repository and model functions (``src/api/app.py``);
re-implements no scoring, recompute, recommendation, outcome-learning, or
promotion logic. Read + controlled-write endpoints only -- there is no
endpoint that promotes an outcome-learning signal or resolves a manual
review (those stay CLI-only).

Database path is explicit: ``--db PATH`` or the ``BETTER_YOU_API_DB``
environment variable. A missing database is an error unless ``--init-db`` is
given, which creates the file and schema once at startup.

Auth is a TODO (``src.api.app.require_alpha_access``): bind to localhost
(the default) and do not expose this port.

Run:
    python tools/serve_api.py --db canonical.sqlite3
    python tools/serve_api.py --db fresh.sqlite3 --init-db
    BETTER_YOU_API_DB=canonical.sqlite3 python tools/serve_api.py --port 8100

Interactive docs are served at http://127.0.0.1:8100/docs .
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DB_ENV_VAR = "BETTER_YOU_API_DB"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the local internal-alpha adaptive-user-model API. Read + "
            "controlled-write endpoints; no promotion, no review-resolution."
        ),
    )
    parser.add_argument(
        "--db", default=None,
        help=f"Path to the SQLite database. Falls back to ${_DB_ENV_VAR}.",
    )
    parser.add_argument(
        "--init-db", action="store_true", dest="init_db",
        help="Create the database file and schema if it does not exist (otherwise a missing DB is an error).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind (default: 8100).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    db_path = args.db or os.environ.get(_DB_ENV_VAR)
    if not db_path:
        print(f"--db is required (or set ${_DB_ENV_VAR}).", file=sys.stderr)
        return 2

    if not args.init_db and not Path(db_path).is_file():
        print(
            f"no such database file: {db_path!r}; pass --init-db to create it.",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from src.api.app import create_app

    app = create_app(db_path, init_db=args.init_db)
    print(
        f"Serving adaptive user model API for {db_path!r} at "
        f"http://{args.host}:{args.port}/  (docs at /docs, Ctrl+C to stop)"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

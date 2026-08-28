#!/usr/bin/env python3
"""Read-only local/dev viewer for the adaptive user model.

Renders whatever is stored in a ``Repository`` SQLite database as one
self-contained HTML page -- events, observations, evidence, beliefs, and
belief-key canonicalization decisions -- so a demo user model can be
eyeballed instead of read as raw JSON.

Strictly read-only, in three independent ways:

- The database is opened via ``Repository.readonly_at_path(db)`` -- SQLite's
  own URI ``mode=ro`` -- so any accidental write call fails at the SQLite
  layer, and a typoed path fails loudly instead of creating an empty file.
- Only the repository's read/list helpers are used
  (``src/viewer/user_model_view.collect_view_model``); there is no code path
  here that recomputes, suppresses, invalidates, canonicalizes, or writes.
- The output is a static HTML file with inline CSS and no JavaScript.

Choosing the database:

    python tools/view_user_model.py --db events.sqlite3          # explicit
    python tools/view_user_model.py                              # prompts for a path
    python tools/view_user_model.py --demo                       # seed a throwaway
                                                                 # db from a shipped
                                                                 # manifest, then view

``--demo`` runs the already-built ``tools/run_manifest.py`` against a fresh
temporary database using the shipped
``tools/manifests/canonicalized_after_work_workout.json`` example, then
opens the viewer on it. The seeding is done by that existing manifest tool,
not by this viewer; the viewer itself still only ever reads.

Other flags: ``--out`` (where to write the HTML; default: a temp file),
``--user-id`` / ``--belief-id`` (scope the page), ``--no-open`` (don't launch
a browser).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.repository import Repository  # noqa: E402
from src.viewer.user_model_view import collect_view_model, render_html  # noqa: E402

_SHIPPED_DEMO_MANIFEST = (
    Path(__file__).resolve().parent / "manifests" / "canonicalized_after_work_workout.json"
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Repository SQLite database as one read-only HTML page "
            "(events, observations, evidence, beliefs, canonicalization decisions)."
        ),
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to the SQLite database. If omitted (and --demo is not set), you are prompted for one.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Seed a throwaway temp database from the shipped demo manifest, then view it.",
    )
    parser.add_argument("--out", default=None, help="Where to write the HTML file (default: a temp file).")
    parser.add_argument("--user-id", default=None, dest="user_id", help="Restrict the page to this user.")
    parser.add_argument(
        "--belief-id", default=None, dest="belief_id",
        help="Restrict evidence and beliefs to this belief_id.",
    )
    parser.add_argument("--no-open", action="store_true", dest="no_open", help="Do not open a browser.")
    return parser.parse_args(argv)


def _prompt_for_db() -> str | None:
    try:
        answer = input("Path to SQLite user-model database: ").strip()
    except EOFError:
        return None
    return answer or None


def _seed_demo_database() -> str:
    """Run the shipped demo manifest into a fresh temp database via the
    existing manifest runner, and return that database's path."""
    import tools.run_manifest as run_manifest

    tmp_dir = Path(tempfile.mkdtemp(prefix="user_model_demo_"))
    db_path = tmp_dir / "demo_user_model.sqlite3"
    exit_code = run_manifest.main(["--db", str(db_path), "--manifest", str(_SHIPPED_DEMO_MANIFEST)])
    if exit_code != 0:
        raise RuntimeError(f"demo manifest failed with exit code {exit_code}")
    return str(db_path)


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
        db_path = args.db or _prompt_for_db()
        if not db_path:
            print("No database path given.", file=sys.stderr)
            return 1

    try:
        repo = Repository.readonly_at_path(db_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        view_model = collect_view_model(
            repo,
            db_path=db_path,
            user_id=args.user_id,
            belief_id=args.belief_id,
            generated_at=datetime.now(timezone.utc),
        )
    finally:
        repo.close()

    html_text = render_html(view_model)

    if args.out is not None:
        out_path = Path(args.out)
        out_path.write_text(html_text, encoding="utf-8")
    else:
        handle = tempfile.NamedTemporaryFile(
            prefix="user_model_view_", suffix=".html", delete=False, mode="w", encoding="utf-8"
        )
        handle.write(html_text)
        handle.close()
        out_path = Path(handle.name)

    summary = view_model.summary()
    print(
        f"Wrote {out_path} "
        f"({summary['events']} events, {summary['observations']} observations, "
        f"{summary['evidence']} evidence, {summary['beliefs']} beliefs, "
        f"{summary['canonicalizations']} canonicalization decisions)."
    )

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

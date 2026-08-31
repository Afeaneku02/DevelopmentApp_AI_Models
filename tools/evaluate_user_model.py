#!/usr/bin/env python3
"""Evaluation-harness CLI for the Better You adaptive user model.

Runs one or more *scenario manifests* -- each a JSON file describing a
sequence of lifecycle steps plus the expectations that should hold
afterwards -- through the real model lifecycle (the same sanctioned
functions the single-purpose CLIs use; no core logic is re-implemented) and
prints a scorecard.

Each scenario runs against its own fresh in-memory database, so scenarios
are independent and the harness writes nothing to disk.

Exit code:
    0  every expectation in every scenario held
    1  at least one expectation failed (or a scenario could not be executed)
    2  usage error (no manifests found, unreadable file)

Run:
    python tools/evaluate_user_model.py --manifest examples/evals
    python tools/evaluate_user_model.py --manifest examples/evals/clean_support_raises_confidence.json
    python tools/evaluate_user_model.py --manifest examples/evals --format json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evals.harness import discover_manifests, format_report, run_manifests  # noqa: E402

_DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "examples" / "evals"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive-user-model evaluation scenarios and print a scorecard. "
            "Exits nonzero if any expectation fails."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        metavar="PATH",
        help=(
            "A scenario manifest .json file or a directory of them. Repeatable. "
            f"Defaults to {_DEFAULT_MANIFEST_DIR}."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Scorecard format on stdout (default: text).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In text format, also list checks that passed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    manifest_paths = args.manifests or [str(_DEFAULT_MANIFEST_DIR)]
    discovered = discover_manifests(manifest_paths)
    if not discovered:
        print(
            f"No scenario manifests found under: {', '.join(manifest_paths)}",
            file=sys.stderr,
        )
        return 2

    missing = [str(p) for p in discovered if not p.is_file()]
    if missing:
        print(f"Manifest file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    report = run_manifests(manifest_paths)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    # A short machine-parseable line on stderr regardless of format.
    summary = report.to_dict()["summary"]
    print(
        f"evaluation: {summary['passed']}/{summary['scenarios']} scenarios passed",
        file=sys.stderr,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

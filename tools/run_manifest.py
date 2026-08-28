#!/usr/bin/env python3
"""Batch import / demo-manifest CLI for the Better You adaptive user model.

Reads a JSON manifest describing an ordered sequence of steps and runs each
one through the *same* already-built, already-tested single-record CLI it
names -- ``tools/add_user_event.py``, ``tools/add_user_observation.py``,
``tools/add_belief_evidence.py``, ``tools/recompute_belief.py``, and
``tools/invalidate_belief_evidence.py`` -- by calling each one's own
``main()`` function in-process with an argv built from the step. This is
pure orchestration: it never re-implements any of those CLIs' parsing,
validation, or persistence logic, so a manifest step behaves identically to
typing the equivalent command by hand.

Manifest format (JSON):
    {
      "steps": [
        {"type": "add_user_event", "user_id": "usr_17", "event_id": "evt_1",
         "event_type": "goal_completed", "source": "app"},
        {"type": "add_user_observation", "observation_id": "obs_1",
         "user_id": "usr_17", "event_id": "evt_1", "category": "routine",
         "observation": "...", "importance": 0.6, "confidence": 0.6},
        ...
      ]
    }

Each step's ``"type"`` selects which CLI runs it (one of the five keys in
``STEP_HANDLERS`` below); every other key becomes that CLI's own flag
(snake_case key -> ``--kebab-case`` flag):

- A list value repeats the flag, for a repeatable CLI flag like
  ``--event-id``/``--replaces-evidence-id``
  (``"event_id": ["evt_1", "evt_2"]`` -> ``--event-id evt_1 --event-id evt_2``).
- ``true``/``false`` selects a ``store_true`` flag like
  ``--allow-no-evidence``/``--backend-validation-passed``
  (``true`` -> the bare flag, ``false`` -> omitted).
- ``null`` omits the flag entirely, so the target CLI's own default applies.
- ``"structured_data"`` and ``"belief_value_json"`` may be given as a real
  JSON value (object, array, string, number, bool, or null) rather than a
  pre-encoded string -- this runner JSON-encodes it for you, since those two
  flags themselves expect a JSON *string* argument.
- A step must not set ``"db"``: every step runs against the one database
  named by this CLI's own ``--db``.

Execution stops at the first step that fails (nonzero exit from that step's
own CLI) -- no later step runs. Whatever earlier steps already wrote stays
committed; this runner performs no cleanup or rollback of its own, the same
as stopping partway through typing a sequence of commands by hand.

Run:
    python tools/run_manifest.py --db events.sqlite3 --manifest manifest.json

A worked example reproducing tools/demo_user_model.py's own scenario as a
manifest lives at tools/manifests/demo_after_work_workout.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.add_belief_evidence as add_belief_evidence  # noqa: E402
import tools.add_user_event as add_user_event  # noqa: E402
import tools.add_user_observation as add_user_observation  # noqa: E402
import tools.invalidate_belief_evidence as invalidate_belief_evidence  # noqa: E402
import tools.recompute_belief as recompute_belief  # noqa: E402

STEP_HANDLERS: dict[str, Callable[[list[str]], int]] = {
    "add_user_event": add_user_event.main,
    "add_user_observation": add_user_observation.main,
    "add_belief_evidence": add_belief_evidence.main,
    "recompute_belief": recompute_belief.main,
    "invalidate_belief_evidence": invalidate_belief_evidence.main,
}

# Flags whose value is itself a JSON-encoded string, so a manifest may supply
# the real JSON value directly instead of a pre-escaped string.
_JSON_VALUE_KEYS = {"structured_data", "belief_value_json"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ordered sequence of steps from a JSON manifest, each through the "
            "matching already-built single-record CLI's own main(). Stops at the first "
            "step that fails."
        ),
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file every step runs against.")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest file.")
    return parser.parse_args(argv)


def _step_to_argv(step: dict[str, Any], *, db: str) -> list[str]:
    argv = ["--db", db]
    for key, value in step.items():
        if key == "type":
            continue
        flag = "--" + key.replace("_", "-")
        if key in _JSON_VALUE_KEYS and not isinstance(value, str):
            argv += [flag, json.dumps(value)]
        elif isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    return argv


def _run_step(index: int, step: dict[str, Any], *, db: str) -> int:
    if "db" in step:
        print(
            f'Step {index}: must not set "db" -- every step runs against this CLI\'s own --db.',
            file=sys.stderr,
        )
        return 1
    step_type = step.get("type")
    if step_type not in STEP_HANDLERS:
        print(
            f"Step {index}: unknown step type {step_type!r}; expected one of {sorted(STEP_HANDLERS)}.",
            file=sys.stderr,
        )
        return 1

    handler = STEP_HANDLERS[step_type]
    step_argv = _step_to_argv(step, db=db)
    print(f"=== Step {index}: {step_type} ===", file=sys.stderr)
    try:
        return handler(step_argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        manifest_text = Path(args.manifest).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Cannot read --manifest {args.manifest!r}: {exc}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in --manifest {args.manifest!r}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(manifest, dict) or not isinstance(manifest.get("steps"), list):
        print(f'--manifest {args.manifest!r} must be a JSON object with a "steps" array.', file=sys.stderr)
        return 1
    steps = manifest["steps"]
    if not steps:
        print(f"--manifest {args.manifest!r} has no steps.", file=sys.stderr)
        return 1

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            print(f"Step {index}: must be a JSON object, got {type(step).__name__}.", file=sys.stderr)
            return 1
        exit_code = _run_step(index, step, db=args.db)
        if exit_code != 0:
            print(
                f"Manifest stopped at step {index} ({step.get('type')!r}): exit code {exit_code}.",
                file=sys.stderr,
            )
            return exit_code

    print(f"All {len(steps)} step(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

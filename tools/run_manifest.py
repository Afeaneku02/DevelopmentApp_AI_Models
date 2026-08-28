#!/usr/bin/env python3
"""Batch import / demo-manifest CLI for the Better You adaptive user model.

Reads a JSON manifest describing an ordered sequence of steps and runs each
one through the *same* already-built, already-tested single-record CLI it
names -- ``tools/add_user_event.py``, ``tools/add_user_observation.py``,
``tools/add_belief_evidence.py``, ``tools/recompute_belief.py``,
``tools/invalidate_belief_evidence.py``, and
``tools/resolve_belief_key.py`` -- by calling each one's own ``main()``
function in-process with an argv built from the step. This is pure
orchestration: it never re-implements any of those CLIs' parsing,
validation, or persistence logic, so a manifest step behaves identically to
typing the equivalent command by hand.

In particular, ``resolve_belief_key`` here is *only* wired up as another
step type -- this runner contains no canonicalization logic of its own, no
belief_key alias table, and no proposal/LLM authority. The single
sanctioned decision path is still
``src.beliefs.canonicalization.authorize_belief_key_canonicalization()``
inside that CLI, which always checks the backend belief_key registry first;
this runner just passes the step's flags through and, if asked, hands the
authorized ``canonical_key`` it printed to a later step.

Manifest format (JSON):
    {
      "steps": [
        {"type": "add_user_event", "user_id": "usr_17", "event_id": "evt_1",
         "event_type": "goal_completed", "source": "app"},
        {"type": "add_user_observation", "observation_id": "obs_1",
         "user_id": "usr_17", "event_id": "evt_1", "category": "routine",
         "observation": "...", "importance": 0.6, "confidence": 0.6},
        {"type": "resolve_belief_key", "id": "can_1",
         "canonicalization_id": "can_evt_1", "user_id": "usr_17",
         "belief_type": "behavioral_tendency",
         "proposed_key": "prefers_evening_exercise_sessions"},
        {"type": "recompute_belief", "belief_id": "bel_1", "user_id": "usr_17",
         "belief_type": "behavioral_tendency",
         "belief_key": "$steps.can_1.canonical_key", "belief_value_json": true},
        ...
      ]
    }

Each step's ``"type"`` selects which CLI runs it (one of the keys in
``STEP_HANDLERS`` below); every other key -- except the two reserved keys
``"type"`` and ``"id"`` -- becomes that CLI's own flag (snake_case key ->
``--kebab-case`` flag):

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

Referencing an earlier step's output:

- A step may carry an ``"id"`` (a short, manifest-unique label). When it
  does, that step's stdout is captured and, if it is a single JSON object
  (as ``resolve_belief_key`` prints), stored under the id.
- A later step may then use the string ``"$steps.<id>.<field>"`` as any
  flag value; the runner substitutes the named field from that stored
  object before building the step's argv. This is a literal, deterministic
  lookup -- no expressions, no arithmetic, no nested paths.
- This is how an authorized ``canonical_key`` reaches ``recompute_belief``:
  the ``resolve_belief_key`` step decides it (registry-backed), prints it,
  and the ``recompute_belief`` step references
  ``"$steps.<id>.canonical_key"`` for its ``--belief-key``. A
  ``keep_separate`` decision prints ``canonical_key`` equal to the raw
  ``proposed_key``, so the same reference still works and simply passes the
  original key through.
- A reference to an unknown id, to a step that printed no JSON object, or
  to a missing field is a hard error that stops the manifest -- the runner
  never falls back to a raw proposed key.

Execution stops at the first step that fails (nonzero exit from that step's
own CLI) -- no later step runs. Whatever earlier steps already wrote stays
committed; this runner performs no cleanup or rollback of its own, the same
as stopping partway through typing a sequence of commands by hand.

Run:
    python tools/run_manifest.py --db events.sqlite3 --manifest manifest.json

Worked examples live in tools/manifests/: demo_after_work_workout.json
reproduces tools/demo_user_model.py's own scenario, and
canonicalized_after_work_workout.json shows a raw proposed belief_key being
resolved to its canonical key before recompute.
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
import tools.resolve_belief_key as resolve_belief_key  # noqa: E402

STEP_HANDLERS: dict[str, Callable[[list[str]], int]] = {
    "add_user_event": add_user_event.main,
    "add_user_observation": add_user_observation.main,
    "add_belief_evidence": add_belief_evidence.main,
    "recompute_belief": recompute_belief.main,
    "invalidate_belief_evidence": invalidate_belief_evidence.main,
    "resolve_belief_key": resolve_belief_key.main,
}

# Keys that configure the runner itself rather than becoming a target-CLI flag.
_RESERVED_KEYS = {"type", "id"}

# Flags whose value is itself a JSON-encoded string, so a manifest may supply
# the real JSON value directly instead of a pre-escaped string.
_JSON_VALUE_KEYS = {"structured_data", "belief_value_json"}

# A step value of exactly this shape -- "$steps.<step_id>.<field>" -- is
# replaced with <field> from the JSON object an earlier step (with that
# "id") printed. Deliberately the only substitution syntax there is.
_REFERENCE_PREFIX = "$steps."


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


def _resolve_reference(
    raw: str, results: dict[str, dict[str, Any] | None], *, index: int
) -> tuple[Any, str | None]:
    """Resolve a single ``"$steps.<id>.<field>"`` string against the outputs
    of earlier steps. Returns ``(value, None)`` on success or
    ``(None, error_message)`` on any failure -- an unknown id, a step that
    produced no JSON object, or a missing field."""
    body = raw[len(_REFERENCE_PREFIX):]
    parts = body.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, (
            f'Step {index}: malformed reference {raw!r}; expected "$steps.<step_id>.<field>".'
        )
    step_id, field = parts
    if step_id not in results:
        return None, (
            f"Step {index}: reference {raw!r} names step id {step_id!r}, which is not an earlier "
            f'step with an "id" (known ids: {sorted(results) or "none"}).'
        )
    output = results[step_id]
    if output is None:
        return None, (
            f"Step {index}: reference {raw!r} points at step {step_id!r}, which did not print a "
            "single JSON object to stdout, so its output cannot be referenced."
        )
    if field not in output:
        return None, (
            f"Step {index}: reference {raw!r} has no field {field!r} in step {step_id!r}'s output "
            f"(available fields: {sorted(output)})."
        )
    return output[field], None


def _resolve_step_references(
    step: dict[str, Any], results: dict[str, dict[str, Any] | None], *, index: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a copy of ``step`` with every ``"$steps..."`` string value --
    top level or inside a list value -- replaced by the referenced output.
    Returns ``(None, error)`` if any reference cannot be resolved."""
    resolved: dict[str, Any] = {}
    for key, value in step.items():
        if isinstance(value, str) and value.startswith(_REFERENCE_PREFIX):
            substituted, error = _resolve_reference(value, results, index=index)
            if error is not None:
                return None, error
            resolved[key] = substituted
        elif isinstance(value, list):
            new_items: list[Any] = []
            for item in value:
                if isinstance(item, str) and item.startswith(_REFERENCE_PREFIX):
                    substituted, error = _resolve_reference(item, results, index=index)
                    if error is not None:
                        return None, error
                    new_items.append(substituted)
                else:
                    new_items.append(item)
            resolved[key] = new_items
        else:
            resolved[key] = value
    return resolved, None


def _step_to_argv(step: dict[str, Any], *, db: str) -> list[str]:
    argv = ["--db", db]
    for key, value in step.items():
        if key in _RESERVED_KEYS:
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


def _run_step(
    index: int, step: dict[str, Any], *, db: str, results: dict[str, dict[str, Any] | None]
) -> tuple[int, dict[str, Any] | None]:
    if "db" in step:
        print(
            f'Step {index}: must not set "db" -- every step runs against this CLI\'s own --db.',
            file=sys.stderr,
        )
        return 1, None
    step_type = step.get("type")
    if step_type not in STEP_HANDLERS:
        print(
            f"Step {index}: unknown step type {step_type!r}; expected one of {sorted(STEP_HANDLERS)}.",
            file=sys.stderr,
        )
        return 1, None

    step_id = step.get("id")
    if step_id is not None:
        if not isinstance(step_id, str) or not step_id:
            print(f'Step {index}: "id" must be a non-empty string.', file=sys.stderr)
            return 1, None
        if step_id in results:
            print(f"Step {index}: duplicate step id {step_id!r}.", file=sys.stderr)
            return 1, None

    resolved_step, error = _resolve_step_references(step, results, index=index)
    if error is not None:
        print(error, file=sys.stderr)
        return 1, None

    handler = STEP_HANDLERS[step_type]
    step_argv = _step_to_argv(resolved_step, db=db)
    print(f"=== Step {index}: {step_type} ===", file=sys.stderr)

    capture = io.StringIO() if step_id is not None else None
    try:
        if capture is not None:
            with contextlib.redirect_stdout(capture):
                exit_code = handler(step_argv)
        else:
            exit_code = handler(step_argv)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1

    output: dict[str, Any] | None = None
    if capture is not None:
        printed = capture.getvalue()
        sys.stdout.write(printed)  # still show the step's own output to the user
        if exit_code == 0:
            try:
                parsed = json.loads(printed)
            except json.JSONDecodeError:
                parsed = None
            output = parsed if isinstance(parsed, dict) else None
        results[step_id] = output
    return exit_code, output


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

    results: dict[str, dict[str, Any] | None] = {}
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            print(f"Step {index}: must be a JSON object, got {type(step).__name__}.", file=sys.stderr)
            return 1
        exit_code, _ = _run_step(index, step, db=args.db, results=results)
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

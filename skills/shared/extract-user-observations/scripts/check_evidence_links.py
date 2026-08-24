#!/usr/bin/env python3
"""Deterministic completion check for extract-user-observations.

Validates that every user_observation in a batch has at least one
observation_events row with link_role="primary", and that observation_events
only reference observation_ids that actually exist in the batch. This is a
structural check only -- it does not judge whether the observation text is a
fair reading of the events, which stays a human/LLM judgment call.

Usage:
    python check_evidence_links.py <batch.json>

<batch.json> shape:
    {
      "observations": [ {"observation_id": ..., ...}, ... ],
      "observation_events": [ {"observation_id": ..., "event_id": ..., "link_role": ...}, ... ]
    }

Exit code 0 means the batch passes; non-zero means it does not, with the
reasons printed to stderr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VALID_LINK_ROLES = {"primary", "supporting"}


def check_batch(batch: dict) -> list[str]:
    errors: list[str] = []
    observations = batch.get("observations", [])
    links = batch.get("observation_events", [])

    observation_ids = {obs.get("observation_id") for obs in observations}
    if len(observation_ids) != len(observations):
        errors.append("Duplicate observation_id values found in observations.")

    primary_count: dict[str, int] = {oid: 0 for oid in observation_ids}
    for link in links:
        obs_id = link.get("observation_id")
        role = link.get("link_role")
        if obs_id not in observation_ids:
            errors.append(f"observation_events references unknown observation_id: {obs_id!r}")
            continue
        if role not in VALID_LINK_ROLES:
            errors.append(f"observation_events row for {obs_id!r} has invalid link_role: {role!r}")
            continue
        if role == "primary":
            primary_count[obs_id] += 1

    for obs in observations:
        obs_id = obs.get("observation_id")
        if primary_count.get(obs_id, 0) == 0:
            errors.append(f"observation {obs_id!r} has no primary observation_events link.")
        if not (isinstance(obs.get("importance"), (int, float)) and 0 <= obs["importance"] <= 1):
            errors.append(f"observation {obs_id!r} has an out-of-range or missing importance.")
        if not (isinstance(obs.get("confidence"), (int, float)) and 0 <= obs["confidence"] <= 1):
            errors.append(f"observation {obs_id!r} has an out-of-range or missing confidence.")
        for field in ("schema_version", "scoring_version", "canonicalizer_version", "policy_version"):
            if not obs.get(field):
                errors.append(f"observation {obs_id!r} is missing required version field: {field}")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: check_evidence_links.py <batch.json>", file=sys.stderr)
        return 2
    batch = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    errors = check_batch(batch)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("OK: all observations have primary evidence links and valid fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

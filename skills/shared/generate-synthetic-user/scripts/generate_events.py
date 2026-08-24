#!/usr/bin/env python3
"""Starting-point generator for the three canonical Better You evaluation
personas (Blueprint section 13). Extend this rather than writing a parallel
generator from scratch.

Usage:
    python generate_events.py <persona> [--seed N] [--output path.json]

<persona> is one of: schedule-shift, intention-vs-behavior, temporary-disruption

Each persona function returns a list of dicts matching the user_event
contract (Blueprint section 5.3). Version fields are left as placeholders
("6", "belief-score-0.6", "canon-0.6", "policy-0.6") -- replace them with
whatever the caller's active configuration actually uses.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone

VERSION_FIELDS = {
    "schema_version": "6",
    "scoring_version": "belief-score-0.6",
    "canonicalizer_version": "canon-0.6",
    "policy_version": "policy-0.6",
}


def _event(event_id: str, user_id: str, event_type: str, timestamp: datetime, structured_data: dict, goal_id: str, raw_content: str | None = None) -> dict:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "raw_content": raw_content,
        "structured_data": structured_data,
        "source": "app",
        "goal_id": goal_id,
        "session_id": f"sess_{event_id}",
        **VERSION_FIELDS,
    }


def generate_schedule_shift(seed: int = 0, weeks: int = 32) -> list[dict]:
    """Persona A: evening workouts succeed for the first half of `weeks`, then
    a schedule shift makes mornings succeed instead. See references/policy.md
    for the full hidden-pattern description this must reproduce. Default
    `weeks=32` yields ~32 events, meeting the 30-100 event Definition of Done
    (Blueprint section 18)."""
    rng = random.Random(seed)
    user_id = "usr_synth_scheduleshift_01"
    goal_id = "goal_synth_workout"
    events = []
    shift_week = weeks // 2
    start = datetime(2026, 6, 1, tzinfo=timezone(timedelta(hours=-5)))
    for week in range(weeks):
        day = start + timedelta(weeks=week, days=rng.choice([0, 2, 4]))
        if week < shift_week:
            completed_time = day.replace(hour=17, minute=rng.randint(0, 20))
            events.append(_event(
                f"evt_synth_scheduleshift_{len(events):03d}", user_id, "goal_completed", completed_time,
                {"goal": "workout", "scheduled_time": "17:00", "completed_time": completed_time.strftime("%H:%M")}, goal_id,
            ))
        elif week == shift_week:
            missed_time = day.replace(hour=17, minute=0)
            events.append(_event(
                f"evt_synth_scheduleshift_{len(events):03d}", user_id, "goal_missed", missed_time,
                {"goal": "workout", "scheduled_time": "17:00", "reason_hint": "new job start date"}, goal_id,
            ))
        else:
            completed_time = day.replace(hour=6, minute=30 + rng.randint(0, 10))
            events.append(_event(
                f"evt_synth_scheduleshift_{len(events):03d}", user_id, "goal_completed", completed_time,
                {"goal": "workout", "scheduled_time": "06:30", "completed_time": completed_time.strftime("%H:%M")}, goal_id,
            ))
    return events


def generate_intention_vs_behavior(seed: int = 0, weeks: int = 15) -> list[dict]:
    """Persona B: user states a nightly-study goal, but successful study
    events actually cluster on Saturday mornings. Two events/week plus one
    stated-goal event; default `weeks=15` yields ~31 events, meeting the
    30-100 event Definition of Done (Blueprint section 18)."""
    rng = random.Random(seed)
    user_id = "usr_synth_intentionvsbehavior_01"
    goal_id = "goal_synth_study"
    events = []
    start = datetime(2026, 6, 1, tzinfo=timezone(timedelta(hours=-5)))
    events.append(_event(
        f"evt_synth_intentionvsbehavior_{len(events):03d}", user_id, "goal_stated", start,
        {"goal": "study", "stated_frequency": "nightly", "stated_time": "21:00"}, goal_id,
        raw_content="I want to study every night after dinner.",
    ))
    for week in range(weeks):
        weeknight_attempt = start + timedelta(weeks=week, days=rng.choice([1, 3]))
        events.append(_event(
            f"evt_synth_intentionvsbehavior_{len(events):03d}", user_id, "goal_missed", weeknight_attempt.replace(hour=21),
            {"goal": "study", "scheduled_time": "21:00"}, goal_id,
        ))
        saturday = start + timedelta(weeks=week, days=5)
        completed_time = saturday.replace(hour=9, minute=rng.randint(0, 30))
        events.append(_event(
            f"evt_synth_intentionvsbehavior_{len(events):03d}", user_id, "goal_completed", completed_time,
            {"goal": "study", "scheduled_time": "09:00", "completed_time": completed_time.strftime("%H:%M")}, goal_id,
        ))
    return events


def generate_temporary_disruption(seed: int = 0, weeks: int = 32) -> list[dict]:
    """Persona C: a normally consistent user becomes inconsistent during a
    clearly-contextualized disruption period (travel/poor sleep/new workload)
    roughly in the middle of the timeline, then returns to the baseline
    pattern. Default `weeks=32` yields ~32 events, meeting the 30-100 event
    Definition of Done (Blueprint section 18)."""
    rng = random.Random(seed)
    user_id = "usr_synth_temporarydisruption_01"
    goal_id = "goal_synth_workout"
    events = []
    disruption_start = weeks // 2 - 1
    disruption_end = weeks // 2
    start = datetime(2026, 6, 1, tzinfo=timezone(timedelta(hours=-5)))
    for week in range(weeks):
        day = start + timedelta(weeks=week, days=rng.choice([0, 2, 4]))
        disrupted = disruption_start <= week <= disruption_end
        if disrupted:
            events.append(_event(
                f"evt_synth_temporarydisruption_{len(events):03d}", user_id, "goal_missed", day.replace(hour=17),
                {"goal": "workout", "scheduled_time": "17:00"}, goal_id,
                raw_content="Traveling for work this week, barely sleeping, skipping the gym.",
            ))
        else:
            completed_time = day.replace(hour=17, minute=rng.randint(0, 20))
            events.append(_event(
                f"evt_synth_temporarydisruption_{len(events):03d}", user_id, "goal_completed", completed_time,
                {"goal": "workout", "scheduled_time": "17:00", "completed_time": completed_time.strftime("%H:%M")}, goal_id,
            ))
    return events


PERSONAS = {
    "schedule-shift": generate_schedule_shift,
    "intention-vs-behavior": generate_intention_vs_behavior,
    "temporary-disruption": generate_temporary_disruption,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("persona", choices=sorted(PERSONAS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weeks", type=int, help="Override the default timeline length (affects event count).")
    parser.add_argument("--output", help="Optional output path; defaults to stdout.")
    args = parser.parse_args(argv)

    kwargs = {"seed": args.seed}
    if args.weeks is not None:
        kwargs["weeks"] = args.weeks
    events = PERSONAS[args.persona](**kwargs)
    output = json.dumps(events, indent=2, ensure_ascii=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

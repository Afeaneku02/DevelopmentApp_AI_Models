"""Mock by default. --live makes bounded API calls with fictional data only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.base import MockMentorProvider, ProviderUnavailable
from src.llm.config import ROOT, Settings
from src.llm.models import ApprovedBelief, MentorContext
from src.llm.openai_provider import OpenAIMentorProvider
from src.llm.validation import validate_guidance


def synthetic_context() -> MentorContext:
    return MentorContext(
        snapshot_id="synthetic-smoke-v1", context_key="fitness_scheduling",
        goal="Build a consistent short walking habit", question="What small step could I try this week?",
        risk_policy_version="synthetic", risk_domain_policy_version="synthetic",
        constraints=["Fictional fixture, not a real user", "Only small reversible scheduling suggestions"],
        beliefs=[ApprovedBelief(
            belief_id="synthetic-after-work", key="higher_adherence_after_work", value="true",
            confidence=0.72, status="provisional", evidence_ids=["synthetic-evidence-1"],
            source_event_ids=["synthetic-event-1"], versions={"schema_version": "synthetic"},
        )],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--calls", type=int, choices=range(1, 11), default=1)
    args = parser.parse_args(argv)
    try:
        if args.live:
            settings = Settings.from_environment(ROOT / ".env")
            # Smoke tool cannot raise the experiment's configured limits.
            settings = settings.model_copy(update={"budget_usd": min(settings.budget_usd, 0.05)})
            provider = OpenAIMentorProvider(settings)
        else:
            provider = MockMentorProvider()
        for index in range(args.calls):
            context = synthetic_context()
            result = provider.generate_guidance(context)
            validate_guidance(result.guidance, context)
            # Only sanitized metadata; never dump prompts, raw responses or keys.
            print(json.dumps({"call": index + 1, "validated": True, "telemetry": result.telemetry.model_dump()}))
        return 0
    except ProviderUnavailable as exc:
        print(json.dumps({"validated": False, "reason_code": exc.code,
                          "telemetry": exc.telemetry.model_dump() if exc.telemetry else None}))
        return 1
    except Exception:
        print(json.dumps({"validated": False, "reason_code": "configuration_or_validation_error"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

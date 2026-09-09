"""Stateless roadmap-draft bridge for the Better You app.

This is the implementation behind ``POST /roadmaps/generate``. It is
deliberately **not** part of the adaptive user model: it opens no database,
reads no belief / evidence / recommendation / outcome state, and writes
nothing. It turns the sanitized ``{goalCategory, goalTitle}`` Better You is
willing to send into a ``RoadmapDraft``-compatible structure -- whose shape
is defined by Better You's own ``RoadmapGenerator`` contract and validated
on Better You's side (``validateRoadmapDraft``) before it persists anything.

This is the *bridge contract*, not the final intelligence layer:
compatibility and safety first. A later iteration can replace
``generate_roadmap_draft()`` with model-informed reasoning without the route
or the request/response contract changing.

Two constraints Better You's validator imposes that the output here must
respect:

- 1-10 milestones, 1-10 action steps per milestone; titles <= 200 chars,
  descriptions <= 1000 chars.
- ``description`` must be an actual string or omitted entirely -- a JSON
  ``null`` makes ``validateRoadmapDraft`` throw. This module always emits a
  real description string.
"""
from __future__ import annotations

from src.api.models import ActionStepDraftOut, MilestoneDraftOut, RoadmapDraftOut

# Better You's own five categories (packages/contracts/src/goal.ts). The
# bridge contract types ``goalCategory`` as a plain string, so an unknown
# value (a category Better You adds later) must fall back to a generic frame
# rather than 422 or 500.
_CATEGORY_FLAVOR: dict[str, str] = {
    "career": "the role you want, the skills it needs, and the people who can open doors",
    "fitness": "training sessions, recovery, and a rhythm you can keep",
    "finances": "what you spend, what you save, and a target you can measure",
    "education": "a study plan and steady, deliberate practice",
    "personal_development": "the small habits and the honest reflection this takes",
}
_GENERIC_FLAVOR = "the concrete, repeatable habits this goal needs"

# Keep an embedded goal title short inside a generated milestone title so the
# whole string stays well under Better You's 200-char cap.
_MAX_EMBED = 80


def _embed(goal_title: str) -> str:
    title = goal_title.strip()
    if len(title) <= _MAX_EMBED:
        return title
    return title[: _MAX_EMBED - 3].rstrip() + "..."


def generate_roadmap_draft(goal_category: str, goal_title: str) -> RoadmapDraftOut:
    """A deterministic three-phase ``RoadmapDraft`` for one goal.

    Pure: no I/O, no randomness. Given the same ``(goal_category,
    goal_title)`` it always returns the same draft.
    """
    short = _embed(goal_title)
    flavor = _CATEGORY_FLAVOR.get(goal_category.strip().lower(), _GENERIC_FLAVOR)

    milestones = [
        MilestoneDraftOut(
            title=f'Get clear on "{short}"',
            description=(
                "Turn the goal into something concrete: what reaching it actually "
                f"looks like, why it matters to you, and where {flavor} fit in."
            ),
            actionSteps=[
                ActionStepDraftOut(
                    title="Write one sentence describing what reaching this goal looks like",
                    description="A specific, checkable outcome -- not a vague aspiration.",
                ),
                ActionStepDraftOut(
                    title="Pick the single first action you can take this week",
                    description="Small enough that starting is easy; real enough that it counts.",
                ),
                ActionStepDraftOut(
                    title="Name the obstacle most likely to stop you",
                    description="Knowing it in advance is how you plan around it instead of into it.",
                ),
            ],
        ),
        MilestoneDraftOut(
            title="Build a routine you can repeat",
            description=(
                "Momentum comes from consistency, not intensity. Put a regular, "
                "low-friction rhythm in place and let it carry the goal for a few weeks."
            ),
            actionSteps=[
                ActionStepDraftOut(
                    title="Schedule a recurring time for this goal and protect it",
                    description="The same slot each week is easier to keep than an open-ended plan.",
                ),
                ActionStepDraftOut(
                    title="Track each time you follow through",
                    description="A simple streak or count you can see makes the habit visible.",
                ),
                ActionStepDraftOut(
                    title="After two weeks, keep what is working and change what is not",
                    description="Adjust the part that keeps slipping rather than starting over.",
                ),
            ],
        ),
        MilestoneDraftOut(
            title=f'Close the gap to "{short}"',
            description=(
                "With a routine running, put your effort on the part that actually "
                "moves you toward the goal, then decide what comes after it."
            ),
            actionSteps=[
                ActionStepDraftOut(
                    title="Review your progress against the outcome you wrote down",
                    description="An honest check: closer, stalled, or drifting off track?",
                ),
                ActionStepDraftOut(
                    title="Tackle the hardest remaining piece on purpose",
                    description="The step you have been avoiding is usually the one that matters most.",
                ),
                ActionStepDraftOut(
                    title="Decide your next goal, or how to go deeper on this one",
                    description="Finishing well includes choosing what you build on it.",
                ),
            ],
        ),
    ]
    return RoadmapDraftOut(milestones=milestones)

from src.llm.models import MentorContext, MentorGuidance
from src.mentor.feedback import _is_high_stakes


def validate_guidance(guidance: MentorGuidance, context: MentorContext) -> MentorGuidance:
    # Revalidate even injected providers rather than trusting model_construct/copy.
    guidance = MentorGuidance.model_validate_json(guidance.model_dump_json())
    allowed = {b.belief_id for b in context.beliefs}
    for item in guidance.recommendations:
        if not set(item.grounded_in_belief_ids) <= allowed:
            raise ValueError("unauthorized grounding")
    # This conservative filter supplements policy and schema validation. It is
    # not a proof of semantic grounding; synthetic adversarial evals are required.
    if _is_high_stakes(guidance.model_dump_json()):
        raise ValueError("high stakes output")
    return guidance

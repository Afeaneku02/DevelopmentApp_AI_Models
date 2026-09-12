"""Mentor-facing feedback generation (local/internal alpha).

``src.mentor.feedback.generate_mentor_feedback`` is a deterministic,
read-only function: given a user's stored beliefs and evidence it returns
structured, cautious mentor feedback grounded in specific belief ids, or
``needs_more_data`` when the model does not yet know enough to say anything
responsibly. No LLM, no persistence, no invented facts.
"""

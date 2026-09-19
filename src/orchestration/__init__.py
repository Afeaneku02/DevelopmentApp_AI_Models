"""Local orchestration layer: chains the already-built pipeline steps
(``src.observations.process_events`` -> ``src.beliefs.process_observations``)
into one per-user operation. See ``process_user_pipeline`` for the entry
point.
"""

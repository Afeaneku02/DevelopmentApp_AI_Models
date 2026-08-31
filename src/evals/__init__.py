"""Evaluation harness for the adaptive user model.

``src.evals.harness`` loads scenario manifests, replays the full model
lifecycle through the existing sanctioned functions (no core logic is
re-implemented), and scores the resulting database against the manifest's
expectations. See ``tools/evaluate_user_model.py`` for the CLI.
"""

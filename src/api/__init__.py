"""Local internal-alpha HTTP API for the Better You adaptive user model.

``src.api.app.create_app(db_path, *, init_db=False)`` builds a FastAPI app
that wraps the *existing* repository and model functions -- it re-implements
no scoring, recompute, recommendation, outcome-learning, or promotion logic.
See ``tools/serve_api.py`` for the runner and ``README.md`` for how to run it.

Boundaries this layer keeps intact:

- Every write endpoint constructs a real domain model
  (``UserEvent``/``UserObservation``/``BeliefEvidence`` via the sanctioned
  ``authorize_evidence``/``UserBelief`` via ``recompute_belief``/
  ``UserRecommendation`` via ``generate_recommendation``/
  ``RecommendationOutcome``) and persists it through ``Repository``.
- Request models (``src.api.models``) expose only client-fillable fields and
  set ``extra="forbid"``, so a payload cannot smuggle in a backend-owned
  field (version tags, ``evidence_id`` authorization state,
  ``authorized_aggregation_mode``, belief ``confidence``/``status``/lock,
  ...).
- There is no endpoint that promotes an outcome-learning signal or resolves
  a manual review; those stay CLI-only
  (``tools/promote_outcome_learning_signal.py`` /
  ``tools/review_outcome_learning_signal.py``).
- Auth is a TODO: ``src.api.app.require_alpha_access`` is the single
  chokepoint and is currently a no-op. Bind to localhost; do not expose.
"""

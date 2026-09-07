"""FastAPI app factory for the local internal-alpha adaptive-user-model API.

``create_app(db_path, *, init_db=False)`` returns a fresh ``FastAPI`` bound
to one SQLite database path. A new ``Repository`` is opened per request
(read-only for ``GET``, writable for ``POST``) and closed afterwards.

Missing database: a request returns ``503`` -- it is never silently created.
Pass ``init_db=True`` (``tools/serve_api.py --init-db``) to create the file
and schema once at startup.

Endpoints
---------
Reads (read-only ``Repository``, never mutate the file):
    GET  /health
    GET  /users/{user_id}/model
    GET  /users/{user_id}/reviews
    GET  /evals

Controlled writes (each validates through a real domain model):
    POST /events
    POST /observations
    POST /belief-evidence
    POST /beliefs/{belief_id}/recompute
    POST /recommendations
    POST /recommendation-outcomes

Deliberately absent: any endpoint that promotes an outcome-learning signal
or resolves a manual review. Those stay CLI-only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import ValidationError

from src.api import service
from src.api.models import (
    BeliefEvidenceIn,
    EventIn,
    ObservationIn,
    RecommendationIn,
    RecommendationOutcomeIn,
    RecomputeIn,
)
from src.storage.repository import Repository

API_VERSION = "0.1.0-alpha"


def require_alpha_access() -> None:
    """TODO(auth): the single chokepoint for future authn/authz on write
    endpoints.

    For the local internal alpha this is intentionally a **no-op** -- bind
    the API to localhost and do not expose it. Before any non-local
    deployment, replace this with a real check (API key / session token) and
    it will apply to every write route without touching the handlers.
    """
    return None


def _map_domain_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=422, detail=f"record validation failed: {exc}")
    if isinstance(exc, sqlite3.IntegrityError):
        return HTTPException(status_code=409, detail=f"id already used: {exc}")
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc  # pragma: no cover - unexpected, let FastAPI 500 it


def create_app(db_path: str | Path, *, init_db: bool = False) -> FastAPI:
    db_path = str(db_path)

    if init_db and not Path(db_path).is_file():
        # Explicit, opt-in creation: build the file + schema once, now.
        Repository.at_path(db_path).close()

    app = FastAPI(
        title="Better You adaptive user model API",
        version=API_VERSION,
        description=(
            "Local internal-alpha wrapper around the existing repository and model "
            "functions. Read + controlled-write endpoints only. No outcome-learning "
            "promotion; manual review approval/rejection stays CLI-only "
            "(tools/review_outcome_learning_signal.py). Auth: TODO -- see "
            "require_alpha_access; bind to localhost and do not expose."
        ),
    )
    app.state.db_path = db_path
    app.state.init_db = init_db

    def _require_db_file() -> None:
        if not Path(db_path).is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"database {db_path!r} does not exist; restart the API with "
                    "--init-db to create it"
                ),
            )

    def read_repo() -> Iterator[Repository]:
        _require_db_file()
        repo = Repository.readonly_at_path(db_path)
        try:
            yield repo
        finally:
            repo.close()

    def write_repo(_: None = Depends(require_alpha_access)) -> Iterator[Repository]:
        _require_db_file()
        repo = Repository.at_path(db_path)
        try:
            yield repo
        finally:
            repo.close()

    def _do(fn, *args) -> Any:
        try:
            return fn(*args)
        except (ValidationError, sqlite3.IntegrityError, LookupError, ValueError) as exc:
            raise _map_domain_error(exc) from exc

    # ----------------------------------------------------------- reads --

    @app.get("/health")
    def health() -> dict[str, Any]:
        path = Path(db_path)
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "db_path": db_path,
            "db_exists": path.is_file(),
            "init_db": init_db,
        }

    @app.get("/users/{user_id}/model")
    def get_user_model(user_id: str, repo: Repository = Depends(read_repo)) -> dict[str, Any]:
        return service.read_user_model(repo, db_path=db_path, user_id=user_id)

    @app.get("/users/{user_id}/reviews")
    def get_user_reviews(user_id: str, repo: Repository = Depends(read_repo)) -> dict[str, Any]:
        return service.read_user_reviews(repo, db_path=db_path, user_id=user_id)

    @app.get("/evals")
    def get_evals() -> dict[str, Any]:
        # Runs the harness in fresh in-memory repos; never touches db_path.
        return service.read_evals()

    # ---------------------------------------------------------- writes --

    @app.post("/events", status_code=201)
    def post_event(payload: EventIn, repo: Repository = Depends(write_repo)) -> dict[str, Any]:
        return _do(service.create_event, repo, payload)

    @app.post("/observations", status_code=201)
    def post_observation(
        payload: ObservationIn, repo: Repository = Depends(write_repo)
    ) -> dict[str, Any]:
        return _do(service.create_observation, repo, payload)

    @app.post("/belief-evidence", status_code=201)
    def post_belief_evidence(
        payload: BeliefEvidenceIn, repo: Repository = Depends(write_repo)
    ) -> dict[str, Any]:
        return _do(service.create_belief_evidence, repo, payload)

    @app.post("/beliefs/{belief_id}/recompute", status_code=201)
    def post_recompute(
        belief_id: str, payload: RecomputeIn, repo: Repository = Depends(write_repo)
    ) -> dict[str, Any]:
        return _do(service.recompute, repo, belief_id, payload)

    @app.post("/recommendations", status_code=201)
    def post_recommendation(
        payload: RecommendationIn, repo: Repository = Depends(write_repo)
    ) -> dict[str, Any]:
        return _do(service.make_recommendation, repo, payload)

    @app.post("/recommendation-outcomes", status_code=201)
    def post_recommendation_outcome(
        payload: RecommendationOutcomeIn, repo: Repository = Depends(write_repo)
    ) -> dict[str, Any]:
        return _do(service.record_recommendation_outcome, repo, payload)

    return app

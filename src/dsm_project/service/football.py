"""Historical matchup routes for the versioned reference demonstration."""

import logging
from datetime import date
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from dsm_project.artifacts.bundle import ModelBundle
from dsm_project.artifacts.football import (
    FootballReference,
    archive_hashes,
    text_fingerprint,
)
from dsm_project.config import load_config
from dsm_project.data import football_features
from dsm_project.data.football_features import (
    FEATURES,
    OUTCOMES,
    matchup_features,
    replay,
)
from dsm_project.data.openfootball import load_matches

LOGGER = logging.getLogger(__name__)


class MatchupRequest(BaseModel):
    """Identify two teams and a date in the supported historical archive."""

    model_config = ConfigDict(extra="forbid")
    home: str = Field(min_length=1, max_length=80)
    away: str = Field(min_length=1, max_length=80)
    date: date


class MatchupResponse(BaseModel):
    """Report probabilities, signed goal difference, and two historical profiles."""

    request_id: str
    artifact_version: str
    probabilities: dict[str, float]
    goal_difference: float
    home_profile: dict[str, float]
    away_profile: dict[str, float]


def create_football_app(bundle: ModelBundle) -> FastAPI:
    """Validate the artifact against its data and expose bounded predictions."""
    model = bundle.estimator
    metadata = bundle.metadata
    config = load_config(metadata["configuration"])
    if (
        not isinstance(model, FootballReference)
        or tuple(metadata["feature_names"]) != FEATURES
        or tuple(model.classifier_.feature_names_in_) != FEATURES
        or tuple(model.regressor_.feature_names_in_) != FEATURES
        or tuple(model.classifier_.classes_) != OUTCOMES
        or tuple(metadata["outcome_order"]) != OUTCOMES
        or text_fingerprint(config.path) != metadata["configuration_sha256"]
        or text_fingerprint(football_features.__file__)
        != metadata["feature_code_sha256"]
        or archive_hashes(config) != metadata["data_sha256"]
    ):
        raise ValueError("Football model, configuration or data history mismatch")
    matches = load_matches(config)
    version = str(metadata["artifact_version"])
    trained_through = date.fromisoformat(metadata["train_through"])
    seasons = {}
    for season in metadata["validation_seasons"]:
        fixtures = matches[matches["season"] == season]
        first, last = fixtures["date"].min(), fixtures["date"].max()
        if fixtures.empty or first <= trained_through:
            raise ValueError("Prediction dates must follow model fitting")
        seasons[season] = {
            "first": first,
            "last": last,
            "teams": sorted(fixtures.home.unique()),
        }
    app = FastAPI(title="Football Intelligence Engine", version="0.1.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)
        request.state.request_id = str(uuid4())
        start = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            LOGGER.log(
                logging.WARNING if status >= 400 else logging.INFO,
                "matchup prediction request",
                extra={
                    "request_id": request.state.request_id,
                    "artifact_version": version,
                    "model_name": metadata["model_name"],
                    "status_code": status,
                    "validation_failure": status == 422,
                    "latency_ms": round((perf_counter() - start) * 1000, 2),
                },
            )

    @app.get("/", response_class=HTMLResponse)
    def analytical_client() -> str:
        page = Path(__file__).parent / "static" / "football.html"
        return page.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": metadata["model_name"],
            "artifact_version": version,
        }

    @app.get("/model")
    def model_metadata() -> dict:
        return {key: value for key, value in metadata.items() if key != "configuration"}

    @app.get("/teams")
    def supported_teams() -> dict:
        return seasons

    @app.post("/predict", response_model=MatchupResponse)
    def predict(match: MatchupRequest, request: Request) -> MatchupResponse:
        supported = next(
            (
                item
                for item in seasons.values()
                if item["first"] <= match.date <= item["last"]
            ),
            None,
        )
        if supported is None:
            raise HTTPException(
                422, "Date must fall within a supported historical season."
            )
        if match.home == match.away or not {match.home, match.away} <= set(
            supported["teams"]
        ):
            raise HTTPException(
                422, "Choose two different teams from the selected season."
            )
        _, histories = replay(matches, before=match.date)
        values = matchup_features(histories, match.home, match.away)
        features = pd.DataFrame([values], columns=FEATURES)
        profiles = {
            side: {
                key.removeprefix(f"{side}_"): value
                for key, value in values.items()
                if key.startswith(f"{side}_")
            }
            for side in ("home", "away")
        }
        probabilities = model.predict_proba(features)[0]
        return MatchupResponse(
            request_id=request.state.request_id,
            artifact_version=version,
            probabilities=dict(zip(OUTCOMES, probabilities, strict=True)),
            goal_difference=float(model.predict_difference(features)[0]),
            home_profile=profiles["home"],
            away_profile=profiles["away"],
        )

    return app

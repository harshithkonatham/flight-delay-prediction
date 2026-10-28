"""HTTP API that returns the delay probability for a scheduled flight."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flightdelay.features import FEATURES
from flightdelay.train import LoadedModel, load_model

log = logging.getLogger("flightdelay.api")

MODEL_DIR_ENV = "FLIGHTDELAY_MODEL_DIR"
DEFAULT_MODEL_DIR = Path("artifacts/models")

AirlineCode = Annotated[str, Field(pattern=r"^[A-Z0-9]{2}$", examples=["DL"])]
AirportCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$", examples=["ATL"])]
ClockTime = Annotated[int, Field(ge=0, le=2359, description="HHMM, local time", examples=[1730])]


class FlightRequest(BaseModel):
    """A scheduled flight. Everything here is known before departure."""

    model_config = ConfigDict(extra="forbid")  # reject unexpected fields

    flight_date: date
    airline: AirlineCode
    origin: AirportCode
    dest: AirportCode
    scheduled_departure: ClockTime
    scheduled_arrival: ClockTime
    scheduled_minutes: Annotated[int, Field(gt=0, le=1200)]
    distance_miles: Annotated[int, Field(gt=0, le=6000)]
    origin_hourly_departures: Annotated[int, Field(ge=1, le=500)] = 30

    @model_validator(mode="after")
    def check_route_and_times(self) -> FlightRequest:
        if self.origin == self.dest:
            raise ValueError("origin and dest must differ")
        for value in (self.scheduled_departure, self.scheduled_arrival):
            if value % 100 >= 60:
                raise ValueError(f"{value} is not a valid HHMM time")
        return self

    def to_frame(self) -> pd.DataFrame:
        row = {
            "airline": self.airline,
            "origin": self.origin,
            "dest": self.dest,
            "month": self.flight_date.month,
            "day_of_month": self.flight_date.day,
            "day_of_week": self.flight_date.isoweekday(),
            "dep_hour": self.scheduled_departure // 100,
            "arr_hour": self.scheduled_arrival // 100,
            "scheduled_minutes": self.scheduled_minutes,
            "distance": self.distance_miles,
            "origin_hourly_departures": self.origin_hourly_departures,
        }
        return pd.DataFrame([row], columns=FEATURES)


class Prediction(BaseModel):
    delay_probability: float
    likely_delayed: bool
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_dir = Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR))
    try:
        app.state.model = load_model(model_dir)
        log.info("Loaded model %s", app.state.model.version)
    except (OSError, ValueError) as exc:
        # Start anyway so /health works, but report not-ready until a model loads.
        app.state.model = None
        log.error("No usable model in %s: %s", model_dir, exc)
    yield


app = FastAPI(
    title="Flight Delay Prediction API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


@app.middleware("http")
async def log_and_harden(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    log.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    return response


def _model(request: Request) -> LoadedModel:
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    """Readiness: a verified model is loaded and requests can be served."""
    model = _model(request)
    return {"status": "ready", "model_version": model.version}


@app.post("/predict")
def predict(flight: FlightRequest, request: Request) -> Prediction:
    model = _model(request)
    probability = float(model.pipeline.predict_proba(flight.to_frame())[0, 1])
    return Prediction(
        delay_probability=round(probability, 4),
        likely_delayed=probability >= 0.5,
        model_version=model.version,
    )

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from flightdelay import api

VALID: dict[str, Any] = {
    "flight_date": "2024-03-15",
    "airline": "DL",
    "origin": "ATL",
    "dest": "LAX",
    "scheduled_departure": 1730,
    "scheduled_arrival": 1915,
    "scheduled_minutes": 285,
    "distance_miles": 1946,
}


@pytest.fixture
def client(trained_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(api.MODEL_DIR_ENV, str(trained_model_dir))
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert "@" in body["model_version"]


def test_predict_returns_probability(client: TestClient) -> None:
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["delay_probability"] <= 1
    assert body["likely_delayed"] == (body["delay_probability"] >= 0.5)


def test_security_headers_are_set(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_unseen_airport_still_gets_a_prediction(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID, "origin": "ZZZ"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("airline", "delta"),
        ("origin", "atl"),
        ("dest", "ATL"),  # same as origin
        ("scheduled_departure", 2460),
        ("scheduled_arrival", 1275),  # 75 minutes past the hour
        ("scheduled_minutes", 0),
        ("distance_miles", 999_999),
        ("flight_date", "not-a-date"),
    ],
)
def test_invalid_input_is_rejected(client: TestClient, field: str, value: Any) -> None:
    response = client.post("/predict", json={**VALID, field: value})
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID, "ArrDelayMinutes": 0})
    assert response.status_code == 422


def test_without_model_service_is_alive_but_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(api.MODEL_DIR_ENV, str(tmp_path))
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=VALID).status_code == 503

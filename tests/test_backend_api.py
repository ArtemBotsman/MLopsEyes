"""API tests for FastAPI backend."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.storage import init_db

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "eye_cnn_best_val_final.pth"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    monkeypatch.setattr("backend.app.storage.DEFAULT_DB_PATH", db_path)
    init_db(db_path)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Model weights are required")
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Model weights are required")
def test_predict_with_generated_image(client):
    image = Image.new("L", (24, 24), color=128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("tiny.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["label"] in {"opened", "closed"}
    assert isinstance(payload["is_anomaly"], bool)
    assert payload["model_version"] == "eye_cnn_best_val_final.pth"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Model weights are required")
def test_predictions_endpoint(client):
    image = Image.new("L", (24, 24), color=90)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    client.post("/predict", files={"file": ("tiny.png", buffer.getvalue(), "image/png")})

    response = client.get("/predictions")
    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert len(payload["predictions"]) >= 1


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlops_predictions_total" in response.text


def test_drift_latest_when_report_missing(client, monkeypatch):
    monkeypatch.setattr("backend.app.main.get_latest_drift_report", lambda: None)
    response = client.get("/drift/latest")
    assert response.status_code == 200
    assert response.json() == {
        "status": "not_available",
        "message": "drift report not generated yet",
        "report": None,
    }

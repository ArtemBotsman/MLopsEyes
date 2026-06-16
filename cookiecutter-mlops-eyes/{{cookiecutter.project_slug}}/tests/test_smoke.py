"""Smoke tests for the generated MLOps skeleton."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project"] == "{{cookiecutter.project_slug}}"

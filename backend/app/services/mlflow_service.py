"""MLflow integration helpers for backend API."""

from __future__ import annotations

import os
from typing import Any

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def list_experiments() -> dict[str, Any]:
    try:
        client = _client()
        experiments = client.search_experiments()
        payload = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
        return {"status": "ok", "experiments": payload}
    except Exception:  # noqa: BLE001
        return {
            "status": "not_available",
            "message": "MLflow tracking server is not available",
            "experiments": None,
        }


def list_registered_models() -> dict[str, Any]:
    try:
        client = _client()
        models = client.search_registered_models()
        payload = [
            {
                "name": model.name,
                "latest_versions": [
                    {
                        "version": version.version,
                        "current_stage": version.current_stage,
                        "status": version.status,
                    }
                    for version in (model.latest_versions or [])
                ],
            }
            for model in models
        ]
        return {"status": "ok", "models": payload}
    except Exception:  # noqa: BLE001
        return {
            "status": "not_available",
            "message": "MLflow tracking server is not available",
            "models": None,
        }

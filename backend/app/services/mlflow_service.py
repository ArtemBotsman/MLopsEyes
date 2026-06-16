"""MLflow integration helpers for backend API."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_REQUEST_TIMEOUT = float(os.getenv("MLFLOW_REQUEST_TIMEOUT", "5"))

T = TypeVar("T")


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def _call_with_timeout(func: Callable[[], T]) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=MLFLOW_REQUEST_TIMEOUT)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _not_available(kind: str) -> dict[str, Any]:
    return {
        "status": "not_available",
        "message": "MLflow tracking server is not available",
        kind: None,
    }


def list_experiments() -> dict[str, Any]:
    try:
        client = _client()
        experiments = _call_with_timeout(lambda: client.search_experiments())
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
    except (FuturesTimeoutError, Exception):  # noqa: BLE001
        return _not_available("experiments")


def list_registered_models() -> dict[str, Any]:
    try:
        client = _client()
        models = _call_with_timeout(lambda: client.search_registered_models())
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
    except (FuturesTimeoutError, Exception):  # noqa: BLE001
        return _not_available("models")

"""FastAPI backend for inference, drift, and monitoring."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from backend.app.schemas import (
    DriftLatestResponse,
    DriftRunResponse,
    ExperimentsResponse,
    HealthResponse,
    ModelsResponse,
    PredictionRecord,
    PredictionsListResponse,
    PredictResponse,
)
from backend.app.services.drift_service import get_latest_drift_report, run_drift_report
from backend.app.services.metrics_service import record_drift_report, record_prediction
from backend.app.services.mlflow_service import list_experiments, list_registered_models
from backend.app.services.predictor import MODEL_VERSION, predict_upload
from backend.app.storage import init_db, list_predictions, save_prediction

app = FastAPI(
    title="Open Eyes Classifier API",
    description="Inference, drift monitoring, and Prometheus metrics",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    started = time.perf_counter()
    score, label, is_anomaly, _ = predict_upload(file)
    created_at = datetime.now(timezone.utc).isoformat()
    latency = time.perf_counter() - started

    save_prediction(
        filename=file.filename or "upload.jpg",
        score=score,
        label=label,
        is_anomaly=is_anomaly,
        created_at=created_at,
        model_version=MODEL_VERSION,
    )
    record_prediction(score=score, label=label, is_anomaly=is_anomaly, latency_seconds=latency)

    return PredictResponse(
        score=round(score, 4),
        label=label,
        is_anomaly=is_anomaly,
        created_at=created_at,
        model_version=MODEL_VERSION,
    )


@app.get("/predictions", response_model=PredictionsListResponse)
def predictions(limit: int = 50) -> PredictionsListResponse:
    rows = list_predictions(limit=limit)
    return PredictionsListResponse(
        predictions=[PredictionRecord(**row) for row in rows],
    )


@app.post("/drift/run", response_model=DriftRunResponse)
def drift_run() -> DriftRunResponse:
    result = run_drift_report()
    summary = result.get("summary") or {}
    if summary:
        record_drift_report(summary)
    return DriftRunResponse(
        status=result["status"],
        message=result.get("message"),
        report_path=result.get("report_path"),
        summary=summary,
    )


@app.get("/drift/latest", response_model=DriftLatestResponse)
def drift_latest() -> DriftLatestResponse:
    report = get_latest_drift_report()
    if report is None:
        return DriftLatestResponse(
            status="not_available",
            message="drift report not generated yet",
        )
    return DriftLatestResponse(status="available", report=report)


@app.get("/experiments", response_model=ExperimentsResponse)
def experiments() -> ExperimentsResponse:
    result = list_experiments()
    return ExperimentsResponse(**result)


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    result = list_registered_models()
    return ModelsResponse(**result)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

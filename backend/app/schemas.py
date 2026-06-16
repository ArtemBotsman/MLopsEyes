"""Pydantic schemas for backend API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictResponse(BaseModel):
    score: float
    label: str
    is_anomaly: bool
    created_at: str
    model_version: str


class PredictionRecord(BaseModel):
    id: int
    filename: str
    score: float
    label: str
    is_anomaly: bool
    created_at: str
    model_version: str


class PredictionsListResponse(BaseModel):
    predictions: list[PredictionRecord] = Field(default_factory=list)


class DriftRunResponse(BaseModel):
    status: str
    message: str | None = None
    report_path: str | None = None
    summary: dict | None = None


class DriftLatestResponse(BaseModel):
    status: str
    message: str | None = None
    report: dict | None = None


class ExperimentsResponse(BaseModel):
    status: str
    message: str | None = None
    experiments: list[dict] | None = None


class ModelsResponse(BaseModel):
    status: str
    message: str | None = None
    models: list[dict] | None = None

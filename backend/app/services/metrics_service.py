"""Prometheus metrics for backend monitoring."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter(
    "mlops_predictions_total",
    "Total number of predictions served by the API",
)
OPENED_PREDICTIONS_TOTAL = Counter(
    "mlops_opened_predictions_total",
    "Total number of opened predictions",
)
CLOSED_PREDICTIONS_TOTAL = Counter(
    "mlops_closed_predictions_total",
    "Total number of closed predictions",
)
ANOMALY_PREDICTIONS_TOTAL = Counter(
    "mlops_anomaly_predictions_total",
    "Total number of anomaly predictions",
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "mlops_prediction_latency_seconds",
    "Prediction latency in seconds",
)
PREDICTION_SCORE = Histogram(
    "mlops_prediction_score",
    "Distribution of prediction scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
DRIFT_RUNS_TOTAL = Counter(
    "mlops_drift_runs_total",
    "Total number of drift report runs",
)
DRIFT_ALERTS_TOTAL = Counter(
    "mlops_drift_alerts_total",
    "Total number of drift alerts",
)
DATA_DRIFT_DETECTED = Gauge(
    "mlops_data_drift_detected",
    "Data drift detected flag (1=true, 0=false)",
)
TARGET_DRIFT_DETECTED = Gauge(
    "mlops_target_drift_detected",
    "Target drift detected flag (1=true, 0=false)",
)
CONCEPT_DRIFT_DETECTED = Gauge(
    "mlops_concept_drift_detected",
    "Concept drift detected flag (1=true, 0=false)",
)
RETRAIN_REQUESTS_TOTAL = Counter(
    "mlops_retrain_requests_total",
    "Total number of retraining requests",
)


def record_prediction(score: float, label: str, is_anomaly: bool, latency_seconds: float) -> None:
    PREDICTIONS_TOTAL.inc()
    if label == "opened":
        OPENED_PREDICTIONS_TOTAL.inc()
    else:
        CLOSED_PREDICTIONS_TOTAL.inc()
    if is_anomaly:
        ANOMALY_PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_seconds)
    PREDICTION_SCORE.observe(score)


def record_drift_report(report: dict) -> None:
    DRIFT_RUNS_TOTAL.inc()
    data_flag = bool(report.get("data_drift", {}).get("data_drift_detected", False))
    target_flag = bool(report.get("target_drift", {}).get("target_drift_detected", False))
    concept_flag = bool(report.get("concept_drift", {}).get("concept_drift_detected", False))
    DATA_DRIFT_DETECTED.set(1 if data_flag else 0)
    TARGET_DRIFT_DETECTED.set(1 if target_flag else 0)
    CONCEPT_DRIFT_DETECTED.set(1 if concept_flag else 0)
    if data_flag or target_flag or concept_flag or report.get("status") == "warning":
        DRIFT_ALERTS_TOTAL.inc()


def record_retrain_request() -> None:
    RETRAIN_REQUESTS_TOTAL.inc()

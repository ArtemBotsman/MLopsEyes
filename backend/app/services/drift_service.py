"""Drift orchestration service."""

from __future__ import annotations

import os
from pathlib import Path

from backend.src.drift import build_drift_report, load_latest_report

REFERENCE_DATASET = os.getenv("REFERENCE_DATASET", "EyesDataset")
CURRENT_DATASET = os.getenv("CURRENT_DATASET", "data/incoming")
REPORT_DIR = os.getenv("DRIFT_REPORT_DIR", "reports/drift")
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "eye_cnn_best_val_final.pth")


def run_drift_report() -> dict:
    current_path = Path(CURRENT_DATASET)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    has_images = any(
        p.suffix.lower() in image_ext for p in current_path.rglob("*") if p.is_file()
    ) if current_path.exists() else False

    if not has_images:
        report = build_drift_report(
            reference_path=REFERENCE_DATASET,
            current_path=CURRENT_DATASET,
            output_dir=REPORT_DIR,
            weights_path=WEIGHTS_PATH,
        )
        return {
            "status": "not_enough_data",
            "message": "No images found in current dataset path",
            "report_path": str(Path(REPORT_DIR) / "latest_drift_report.json"),
            "summary": report,
        }

    report = build_drift_report(
        reference_path=REFERENCE_DATASET,
        current_path=CURRENT_DATASET,
        output_dir=REPORT_DIR,
        weights_path=WEIGHTS_PATH,
    )
    return {
        "status": report.get("status", "ok"),
        "message": None,
        "report_path": str(Path(REPORT_DIR) / "latest_drift_report.json"),
        "summary": report,
    }


def get_latest_drift_report() -> dict | None:
    return load_latest_report(REPORT_DIR)

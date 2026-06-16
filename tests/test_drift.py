"""Drift detection and report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from backend.src.drift import build_drift_report
from backend.src.features import extract_dataset_features, extract_image_features

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "eye_cnn_best_val_final.pth"


def _save_gray_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (24, 24), color=value).save(path)


def test_feature_extraction(tmp_path):
    image_path = tmp_path / "sample.png"
    _save_gray_image(image_path, 120)
    features = extract_image_features(str(image_path))
    assert "mean_pixel" in features
    assert "hist_7" in features
    assert features["mean_pixel"] == 120.0


def test_extract_dataset_features_with_labels(tmp_path):
    _save_gray_image(tmp_path / "opened" / "a.jpg", 200)
    _save_gray_image(tmp_path / "closed" / "b.jpg", 40)
    df = extract_dataset_features(str(tmp_path))
    assert len(df) == 2
    assert set(df["label"]) == {"opened", "closed"}


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Model weights are required")
def test_drift_report_generation_with_temporary_folders(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    output = tmp_path / "reports"

    _save_gray_image(reference / "opened" / "r1.jpg", 180)
    _save_gray_image(reference / "closed" / "r2.jpg", 60)
    _save_gray_image(current / "opened" / "c1.jpg", 170)
    _save_gray_image(current / "closed" / "c2.jpg", 55)

    report = build_drift_report(
        reference_path=str(reference),
        current_path=str(current),
        output_dir=str(output),
        weights_path=str(WEIGHTS),
        max_samples_per_dataset=10,
    )
    assert "data_drift" in report

    json_path = output / "latest_drift_report.json"
    html_path = output / "latest_drift_report.html"
    assert json_path.exists()
    assert html_path.exists()

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert "data_drift" in saved
    assert "target_drift" in saved
    assert "concept_drift" in saved


def test_drift_report_when_current_empty(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    output = tmp_path / "reports"
    _save_gray_image(reference / "opened" / "r1.jpg", 180)
    current.mkdir()

    report = build_drift_report(
        reference_path=str(reference),
        current_path=str(current),
        output_dir=str(output),
        weights_path=str(WEIGHTS),
        max_samples_per_dataset=5,
    )
    assert report["status"] == "not_enough_data"
    assert (output / "latest_drift_report.json").exists()

"""Drift detection: data, target, and concept drift."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.src.features import extract_dataset_features
from backend.src.reports import save_drift_report

FEATURE_COLUMNS = [
    "mean_pixel",
    "std_pixel",
    "min_pixel",
    "max_pixel",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    *[f"hist_{i}" for i in range(8)],
]
TARGET_DRIFT_THRESHOLD = 0.2
CONCEPT_DRIFT_ACCURACY_THRESHOLD = 0.85
KS_ALPHA = 0.05


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_images(dataset_path: Path) -> list[Path]:
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    if not dataset_path.exists():
        return []
    return sorted(
        p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in image_ext
    )


def _has_current_data(current_path: Path) -> bool:
    return len(_list_images(current_path)) > 0


def _score_to_label(score: float) -> str:
    return "opened" if score >= 0.5 else "closed"


def _predict_dataset_scores(dataset_path: Path, weights_path: str) -> pd.DataFrame:
    from open_eyes_classifier import OpenEyesClassificator

    classifier = OpenEyesClassificator(weights_path=weights_path)
    rows: list[dict] = []
    root = dataset_path

    for image_path in _list_images(dataset_path):
        score = classifier.predict(str(image_path))
        row = {
            "filepath": str(image_path),
            "score": float(score),
            "predicted_label": _score_to_label(score),
        }
        try:
            rel_parts = image_path.relative_to(root).parts
        except ValueError:
            rel_parts = ()
        if rel_parts and rel_parts[0] in {"opened", "closed"}:
            row["true_label"] = rel_parts[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, Any]:
    feature_results: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref_values = reference_df[feature].dropna().to_numpy()
        cur_values = current_df[feature].dropna().to_numpy()
        if len(ref_values) == 0 or len(cur_values) == 0:
            continue
        statistic, p_value = stats.ks_2samp(ref_values, cur_values, method="auto")
        drift_detected = bool(p_value < KS_ALPHA)
        feature_results.append(
            {
                "feature": feature,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
            }
        )

    data_drift_detected = any(item["drift_detected"] for item in feature_results)
    return {
        "data_drift_detected": data_drift_detected,
        "features": feature_results,
    }


def compute_target_drift(
    reference_preds: pd.DataFrame,
    current_preds: pd.DataFrame,
) -> dict[str, Any]:
    if reference_preds.empty or current_preds.empty:
        return {
            "reference_opened_ratio": None,
            "current_opened_ratio": None,
            "delta": None,
            "target_drift_detected": False,
        }

    ref_ratio = float((reference_preds["predicted_label"] == "opened").mean())
    cur_ratio = float((current_preds["predicted_label"] == "opened").mean())
    delta = abs(ref_ratio - cur_ratio)
    return {
        "reference_opened_ratio": ref_ratio,
        "current_opened_ratio": cur_ratio,
        "delta": delta,
        "target_drift_detected": bool(delta > TARGET_DRIFT_THRESHOLD),
    }


def compute_concept_drift(current_preds: pd.DataFrame) -> dict[str, Any]:
    if "true_label" not in current_preds.columns or current_preds["true_label"].isna().all():
        return {
            "concept_drift_status": "not_available",
            "reason": "true labels are not available for current data",
            "concept_drift_detected": False,
        }

    labeled = current_preds.dropna(subset=["true_label"])
    if labeled.empty:
        return {
            "concept_drift_status": "not_available",
            "reason": "true labels are not available for current data",
            "concept_drift_detected": False,
        }

    y_true = labeled["true_label"].tolist()
    y_pred = labeled["predicted_label"].tolist()
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, pos_label="opened", zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label="opened", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label="opened", zero_division=0))
    concept_drift_detected = accuracy < CONCEPT_DRIFT_ACCURACY_THRESHOLD

    return {
        "concept_drift_status": "available",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "concept_drift_detected": concept_drift_detected,
    }


def build_drift_report(
    reference_path: str,
    current_path: str,
    output_dir: str,
    weights_path: str = "eye_cnn_best_val_final.pth",
    max_samples_per_dataset: int | None = 200,
) -> dict[str, Any]:
    reference = Path(reference_path)
    current = Path(current_path)
    output = Path(output_dir)

    if not _has_current_data(current):
        report = {
            "generated_at": _utc_now(),
            "status": "not_enough_data",
            "message": "No images found in current dataset path",
            "reference_dataset": str(reference),
            "current_dataset": str(current),
            "data_drift": {"data_drift_detected": False, "features": []},
            "target_drift": {
                "reference_opened_ratio": None,
                "current_opened_ratio": None,
                "delta": None,
                "target_drift_detected": False,
            },
            "concept_drift": {
                "concept_drift_status": "not_available",
                "reason": "current data is empty",
                "concept_drift_detected": False,
            },
            "overall_status": "not_enough_data",
        }
        save_drift_report(report, output)
        return report

    reference_df = extract_dataset_features(str(reference))
    current_df = extract_dataset_features(str(current))
    if max_samples_per_dataset and len(reference_df) > max_samples_per_dataset:
        reference_df = reference_df.sample(max_samples_per_dataset, random_state=42)
    if max_samples_per_dataset and len(current_df) > max_samples_per_dataset:
        current_df = current_df.sample(max_samples_per_dataset, random_state=42)

    data_drift = compute_data_drift(reference_df, current_df)

    reference_preds = _predict_dataset_scores(reference, weights_path)
    current_preds = _predict_dataset_scores(current, weights_path)
    if max_samples_per_dataset and len(reference_preds) > max_samples_per_dataset:
        reference_preds = reference_preds.sample(max_samples_per_dataset, random_state=42)
    if max_samples_per_dataset and len(current_preds) > max_samples_per_dataset:
        current_preds = current_preds.sample(max_samples_per_dataset, random_state=42)

    target_drift = compute_target_drift(reference_preds, current_preds)
    concept_drift = compute_concept_drift(current_preds)

    any_drift = (
        data_drift["data_drift_detected"]
        or target_drift["target_drift_detected"]
        or concept_drift.get("concept_drift_detected", False)
    )
    report = {
        "generated_at": _utc_now(),
        "status": "warning" if any_drift else "ok",
        "reference_dataset": str(reference),
        "current_dataset": str(current),
        "reference_samples": int(len(reference_df)),
        "current_samples": int(len(current_df)),
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "overall_status": "WARNING: drift detected" if any_drift else "OK",
    }
    save_drift_report(report, output)
    return report


def load_latest_report(output_dir: str = "reports/drift") -> dict[str, Any] | None:
    report_path = Path(output_dir) / "latest_drift_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift detection report")
    parser.add_argument("--reference", default="EyesDataset")
    parser.add_argument("--current", default="data/incoming")
    parser.add_argument("--output", default="reports/drift")
    parser.add_argument("--weights", default="eye_cnn_best_val_final.pth")
    args = parser.parse_args()
    report = build_drift_report(args.reference, args.current, args.output, args.weights)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

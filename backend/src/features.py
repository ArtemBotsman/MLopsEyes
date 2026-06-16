"""Image feature extraction for drift detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
HIST_BINS = 8


def _load_grayscale_24x24(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((24, 24))
    return np.asarray(img, dtype=np.float32)


def extract_image_features(image_path: str) -> dict:
    """Extract numeric features from a grayscale 24x24 image."""
    pixels = _load_grayscale_24x24(image_path)
    flat = pixels.flatten()
    hist, _ = np.histogram(flat, bins=HIST_BINS, range=(0, 255))

    features = {
        "mean_pixel": float(flat.mean()),
        "std_pixel": float(flat.std()),
        "min_pixel": float(flat.min()),
        "max_pixel": float(flat.max()),
        "dark_pixel_ratio": float((flat < 50).mean()),
        "bright_pixel_ratio": float((flat > 200).mean()),
    }
    for idx, value in enumerate(hist):
        features[f"hist_{idx}"] = float(value)
    return features


def _infer_label_from_path(path: Path, dataset_root: Path) -> str | None:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    if parts[0] in {"opened", "closed"}:
        return parts[0]
    return None


def _iter_image_files(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path)
    if not root.exists():
        return []

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def extract_dataset_features(dataset_path: str) -> pd.DataFrame:
    """Build a feature dataframe for all images under a dataset path."""
    root = Path(dataset_path)
    rows: list[dict] = []
    for image_path in _iter_image_files(root):
        row = extract_image_features(str(image_path))
        row["filepath"] = str(image_path)
        label = _infer_label_from_path(image_path, root)
        if label is not None:
            row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)

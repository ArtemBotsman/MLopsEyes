"""Model inference service."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import UploadFile
from PIL import Image

from open_eyes_classifier import OpenEyesClassificator

MODEL_VERSION = "eye_cnn_best_val_final.pth"
DEFAULT_WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", MODEL_VERSION)
LOW_BRIGHTNESS_THRESHOLD = 30.0
HIGH_BRIGHTNESS_THRESHOLD = 225.0
UNCERTAIN_LOW = 0.4
UNCERTAIN_HIGH = 0.6


@lru_cache(maxsize=1)
def get_classifier() -> OpenEyesClassificator:
    return OpenEyesClassificator(weights_path=DEFAULT_WEIGHTS)


def score_to_label(score: float) -> str:
    return "opened" if score >= 0.5 else "closed"


def _mean_brightness(image_path: str) -> float:
    img = Image.open(image_path).convert("L").resize((24, 24))
    return float(np.asarray(img, dtype=np.float32).mean())


def detect_anomaly(score: float, image_path: str) -> bool:
    uncertain = UNCERTAIN_LOW <= score <= UNCERTAIN_HIGH
    brightness = _mean_brightness(image_path)
    bad_brightness = brightness < LOW_BRIGHTNESS_THRESHOLD or brightness > HIGH_BRIGHTNESS_THRESHOLD
    return uncertain or bad_brightness


def predict_upload(file: UploadFile) -> tuple[float, str, bool, str]:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        temp_path = tmp.name

    try:
        classifier = get_classifier()
        score = classifier.predict(temp_path)
        label = score_to_label(score)
        is_anomaly = detect_anomaly(score, temp_path)
        return score, label, is_anomaly, temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)

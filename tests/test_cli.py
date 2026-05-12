"""Smoke tests for open_eyes_classifier CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "open_eyes_classifier.py"


def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "image_path" in result.stdout or "--weights" in result.stdout


def test_predict_on_temp_grayscale_png(tmp_path: Path) -> None:
    img_path = tmp_path / "tiny.png"
    Image.new("L", (24, 24), color=128).save(img_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(img_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    first_line = result.stdout.strip().splitlines()[0]
    score = float(first_line)
    assert 0.0 <= score <= 1.0

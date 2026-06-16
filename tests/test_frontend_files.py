"""Static checks for frontend files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_app_exists():
    assert (ROOT / "frontend" / "streamlit_app.py").is_file()


def test_frontend_dockerfile_exists():
    assert (ROOT / "docker" / "Dockerfile.frontend").is_file()


def test_streamlit_app_uses_api_url():
    content = (ROOT / "frontend" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "API_URL" in content


def test_streamlit_app_has_required_pages():
    content = (ROOT / "frontend" / "streamlit_app.py").read_text(encoding="utf-8")
    for page in [
        "Inference",
        "Predictions",
        "Drift",
        "Experiments",
        "Retraining",
        "System status",
    ]:
        assert page in content

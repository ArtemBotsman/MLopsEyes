"""Tests for the Cookiecutter MLOps template."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "cookiecutter-mlops-eyes"
COOKIECUTTER_JSON = TEMPLATE_DIR / "cookiecutter.json"
PROJECT_TEMPLATE_DIR = TEMPLATE_DIR / "{{cookiecutter.project_slug}}"

REQUIRED_KEYS = ("project_slug", "project_name", "model_name")

REQUIRED_TEMPLATE_FILES = (
    "backend/app/main.py",
    "frontend/streamlit_app.py",
    "docker-compose.yml",
    "README.md",
)


def _load_cookiecutter_defaults() -> dict:
    return json.loads(COOKIECUTTER_JSON.read_text(encoding="utf-8"))


def test_cookiecutter_json_exists():
    assert COOKIECUTTER_JSON.is_file()


def test_project_slug_directory_exists():
    assert PROJECT_TEMPLATE_DIR.is_dir()


@pytest.mark.parametrize("relative_path", REQUIRED_TEMPLATE_FILES)
def test_required_template_files_exist(relative_path: str):
    assert (PROJECT_TEMPLATE_DIR / relative_path).is_file()


def test_cookiecutter_json_contains_required_keys():
    data = _load_cookiecutter_defaults()
    for key in REQUIRED_KEYS:
        assert key in data


def test_cookiecutter_generation(tmp_path: Path):
    cookiecutter = shutil.which("cookiecutter")
    if cookiecutter is None:
        pytest.skip("cookiecutter CLI not installed")

    result = subprocess.run(
        [
            cookiecutter,
            "--no-input",
            str(TEMPLATE_DIR),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    defaults = _load_cookiecutter_defaults()
    project_dir = tmp_path / defaults["project_slug"]
    assert project_dir.is_dir()

    for relative_path in REQUIRED_TEMPLATE_FILES:
        assert (project_dir / relative_path).is_file()

    health_source = (project_dir / "backend/app/main.py").read_text(encoding="utf-8")
    assert defaults["project_slug"] in health_source
    assert defaults["model_name"] in health_source

    smoke_test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-q"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_test.returncode == 0, smoke_test.stderr or smoke_test.stdout

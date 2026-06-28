"""Tests for quick retrain script and config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRAIN_CONFIG = PROJECT_ROOT / "configs" / "retrain.yaml"
RETRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "retrain.py"


def test_retrain_config_exists():
    assert RETRAIN_CONFIG.is_file()


def test_retrain_script_exists():
    assert RETRAIN_SCRIPT.is_file()


def test_retrain_config_epochs_in_range():
    config = yaml.safe_load(RETRAIN_CONFIG.read_text(encoding="utf-8"))
    assert 1 <= int(config["epochs"]) <= 3


@pytest.mark.parametrize("epochs", [1, 2, 3])
def test_retrain_script_accepts_epochs(epochs, monkeypatch):
    from scripts import retrain

    monkeypatch.setattr(sys, "argv", ["retrain.py", "--epochs", str(epochs)])
    args = retrain.parse_args()
    assert args.epochs == epochs


def test_retrain_script_rejects_invalid_epochs(monkeypatch):
    from scripts import retrain

    monkeypatch.setattr(sys, "argv", ["retrain.py", "--epochs", "4"])
    with pytest.raises(SystemExit):
        retrain.parse_args()

"""Tests for MLflow training pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_train_module_exists():
    assert (ROOT / "backend" / "src" / "train.py").is_file()


def test_train_config_exists():
    assert (ROOT / "configs" / "train.yaml").is_file()


def test_train_config_parse():
    config = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    assert config["experiment_name"] == "open-eyes-classifier"
    assert config["registered_model_name"] == "open-eyes-cnn"
    assert config["epochs"] >= 1


def test_build_train_config_from_yaml():
    from backend.src.train import TrainConfig, build_train_config

    class Args:
        config = str(ROOT / "configs" / "train.yaml")
        data = None
        epochs = None
        batch_size = None
        learning_rate = None
        experiment_name = None
        registered_model_name = None
        fast_dev_run = True

    config = build_train_config(Args())
    assert isinstance(config, TrainConfig)
    assert config.fast_dev_run is True
    assert config.epochs == 1
    assert config.experiment_name == "open-eyes-classifier"


@pytest.mark.skipif(not (ROOT / "EyesDataset" / "opened").exists(), reason="EyesDataset required")
def test_fast_dev_training_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    from backend.src.train import TrainConfig, train_model

    config = TrainConfig(
        data=str(ROOT / "EyesDataset"),
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        fast_dev_run=True,
        model_output_path=str(tmp_path / "model.pth"),
    )
    summary = train_model(config)
    assert summary["run_id"]
    assert "accuracy" in summary["metrics"]
    assert Path(summary["weights_path"]).exists()

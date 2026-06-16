"""Training pipeline with MLflow tracking and model registry."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from open_eyes_classifier import MediumEyeCNN

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class TrainConfig:
    data: str = "EyesDataset"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    experiment_name: str = "open-eyes-classifier"
    registered_model_name: str = "open-eyes-cnn"
    model_output_path: str = "artifacts/models/open_eyes_cnn_mlflow.pth"
    fast_dev_run: bool = False


class EyesDatasetLoader(Dataset):
    def __init__(self, root: str, transform: transforms.Compose, max_per_class: int | None = None):
        self.transform = transform
        self.samples: list[tuple[Path, float]] = []
        root_path = Path(root)
        for label_name, label_value in (("opened", 1.0), ("closed", 0.0)):
            class_dir = root_path / label_name
            if not class_dir.exists():
                continue
            images = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_per_class is not None:
                images = images[:max_per_class]
            self.samples.extend((path, label_value) for path in images)
        if not self.samples:
            raise FileNotFoundError(f"No training images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path)
        tensor = self.transform(image)
        return tensor, torch.tensor([label], dtype=torch.float32)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if args.config:
        file_config = load_config(args.config)
        if file_config.get("dataset_path"):
            config.data = str(file_config["dataset_path"])
        if file_config.get("epochs") is not None:
            config.epochs = int(file_config["epochs"])
        if file_config.get("batch_size") is not None:
            config.batch_size = int(file_config["batch_size"])
        if file_config.get("learning_rate") is not None:
            config.learning_rate = float(file_config["learning_rate"])
        if file_config.get("experiment_name"):
            config.experiment_name = str(file_config["experiment_name"])
        if file_config.get("registered_model_name"):
            config.registered_model_name = str(file_config["registered_model_name"])
        if file_config.get("model_output_path"):
            config.model_output_path = str(file_config["model_output_path"])

    if args.data:
        config.data = args.data
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.registered_model_name:
        config.registered_model_name = args.registered_model_name
    if args.fast_dev_run:
        config.fast_dev_run = True
        config.epochs = 1
        config.batch_size = min(config.batch_size, 8)
    return config


def split_dataset(
    dataset: EyesDatasetLoader,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batches = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy().reshape(-1)
            preds = (outputs >= 0.5).astype(int)
            truths = labels.numpy().reshape(-1).astype(int)
            y_true.extend(truths.tolist())
            y_pred.extend(preds.tolist())
    return y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["closed", "opened"])
    ax.set_yticks([0, 1], labels=["closed", "opened"])
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, int(value), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def register_model(run_id: str, model_uri: str, registered_model_name: str) -> str | None:
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        return result.version
    except Exception as exc:  # noqa: BLE001
        print(f"Model registry registration skipped: {exc}")
        return None


def train_model(config: TrainConfig) -> dict[str, Any]:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((24, 24)),
            transforms.ToTensor(),
        ]
    )

    max_per_class = 16 if config.fast_dev_run else None
    full_dataset = EyesDatasetLoader(config.data, transform, max_per_class=max_per_class)
    train_dataset, val_dataset = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = MediumEyeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    output_dir = Path("artifacts/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(config.model_output_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.png"

    summary: dict[str, Any] = {}
    with mlflow.start_run(run_name="fast-dev-run" if config.fast_dev_run else None) as run:
        mlflow.log_params(
            {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "fast_dev_run": config.fast_dev_run,
                "dataset_path": config.data,
            }
        )

        last_train_loss = 0.0
        last_val_loss = 0.0
        for epoch in range(config.epochs):
            last_train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            last_val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
            mlflow.log_metric("train_loss", last_train_loss, step=epoch)
            mlflow.log_metric("val_loss", last_val_loss, step=epoch)

        y_true, y_pred = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        save_confusion_matrix(y_true, y_pred, confusion_path)
        torch.save(model.state_dict(), weights_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(weights_path))
        mlflow.log_artifact(str(metrics_path))

        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=config.registered_model_name,
            )
            registered_version = config.registered_model_name
        except Exception as exc:  # noqa: BLE001
            print(f"Model registry registration skipped: {exc}")
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
            )
            registered_version = register_model(
                run.info.run_id,
                model_info.model_uri,
                config.registered_model_name,
            )

        summary = {
            "run_id": run.info.run_id,
            "experiment_name": config.experiment_name,
            "metrics": metrics,
            "train_loss": last_train_loss,
            "val_loss": last_val_loss,
            "weights_path": str(weights_path),
            "registered_model_name": config.registered_model_name,
            "registered_model_version": registered_version,
            "tracking_uri": tracking_uri,
        }

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Open Eyes Classifier with MLflow")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data", type=str, default=None, help="Dataset root path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--registered-model-name", type=str, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_train_config(args)
    train_model(config)


if __name__ == "__main__":
    main()

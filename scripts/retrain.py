#!/usr/bin/env python3
"""Quick retrain CLI with short config (1–3 epochs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrain.yaml"
MIN_EPOCHS = 1
MAX_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick retrain for Open Eyes Classifier (1–3 epochs)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        choices=range(MIN_EPOCHS, MAX_EPOCHS + 1),
        default=None,
        help=f"Number of training epochs ({MIN_EPOCHS}–{MAX_EPOCHS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from backend.src.train import build_train_config, train_model

    namespace = argparse.Namespace(
        config=str(args.config),
        data=None,
        epochs=args.epochs,
        batch_size=None,
        learning_rate=None,
        experiment_name=None,
        registered_model_name=None,
        fast_dev_run=False,
    )
    config = build_train_config(namespace)
    if not MIN_EPOCHS <= config.epochs <= MAX_EPOCHS:
        print(
            f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}, got {config.epochs}",
            file=sys.stderr,
        )
        return 1

    print(f"Starting quick retrain: epochs={config.epochs}, dataset={config.data}")
    train_model(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

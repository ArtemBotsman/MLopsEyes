"""Training skeleton for {{cookiecutter.model_name}}."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train {{cookiecutter.model_name}}")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[{{cookiecutter.project_slug}}] training placeholder "
        f"for {{cookiecutter.model_name}} (epochs={args.epochs})"
    )


if __name__ == "__main__":
    main()

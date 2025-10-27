#!/usr/bin/env python
"""CLI for training the disease segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from sustainable_agriculture.config import TrainingConfig
from sustainable_agriculture.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.expanduser().resolve()
    config = TrainingConfig(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    train_model(config)


if __name__ == "__main__":
    main()

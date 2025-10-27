#!/usr/bin/env python
"""CLI for generating a synthetic precision agriculture dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sustainable_agriculture.config import DatasetConfig, ensure_directory
from sustainable_agriculture.dataset import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory where the dataset will be stored")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help="Ground sampling distance in meters for each pixel",
    )
    parser.add_argument("--width", type=float, default=500.0, help="Field width in meters")
    parser.add_argument("--height", type=float, default=500.0, help="Field height in meters")
    parser.add_argument(
        "--disease-fraction",
        type=float,
        default=0.2,
        help="Approximate fraction of the field affected by disease",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output)

    config = DatasetConfig(
        output_dir=output_dir,
        sample_count=args.samples,
    )
    config.field.width_m = args.width
    config.field.height_m = args.height
    config.field.resolution_m = args.resolution
    config.disease.diseased_fraction = args.disease_fraction

    generate_dataset(config)
    print(f"Dataset generated at {output_dir}")


if __name__ == "__main__":
    main()

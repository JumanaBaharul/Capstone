#!/usr/bin/env python
"""End-to-end pipeline for sustainable precision agriculture simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sustainable_agriculture.config import (
    DatasetConfig,
    InferenceConfig,
    TrainingConfig,
    VisualizationConfig,
    ensure_directory,
)
from sustainable_agriculture.dataset import generate_dataset
from sustainable_agriculture.inference import run_inference
from sustainable_agriculture.route_planner import compute_route
from sustainable_agriculture.train import train_model
from sustainable_agriculture.visualization import create_treatment_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Directory to store datasets and outputs")
    parser.add_argument("--samples", type=int, default=50, help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument("--skip-generation", action="store_true", help="Skip dataset generation step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-visualisation", action="store_true", help="Skip folium map export")
    parser.add_argument("--origin-lat", type=float, default=0.0, help="Latitude for map origin")
    parser.add_argument("--origin-lon", type=float, default=0.0, help="Longitude for map origin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = ensure_directory(args.workspace)
    dataset_dir = ensure_directory(workspace / "dataset")

    if not args.skip_generation:
        dataset_config = DatasetConfig(output_dir=dataset_dir, sample_count=args.samples)
        generate_dataset(dataset_config)

    if not args.skip_training:
        training_config = TrainingConfig(
            dataset_dir=dataset_dir,
            num_epochs=args.epochs,
            device=args.device,
        )
        train_model(training_config)

    inference_config = InferenceConfig(
        model_path=dataset_dir / "model.pt",
        dataset_dir=dataset_dir,
        device=args.device,
    )
    predictions = run_inference(inference_config)

    all_polygons = [polygon for result in predictions for polygon in result.polygons]
    route = compute_route(all_polygons)

    route_path = workspace / "treatment_route.json"
    route_summary = {
        "order": route.order,
        "waypoints": route.waypoints,
        "total_length_m": route.total_length,
    }
    route_path.write_text(json.dumps(route_summary, indent=2))

    if not args.skip_visualisation:
        map_config = VisualizationConfig(map_output=workspace / "treatment_map.html")
        create_treatment_map(predictions, route, map_config, origin_lat=args.origin_lat, origin_lon=args.origin_lon)

    print(f"Pipeline complete. Outputs stored in {workspace}")


if __name__ == "__main__":
    main()

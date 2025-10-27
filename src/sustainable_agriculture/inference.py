"""Inference routines for disease localisation and polygon extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from shapely.geometry import Polygon, mapping
from skimage import measure

from .config import InferenceConfig
from .dataset import load_sample
from .model import UNet


@dataclass
class PredictionResult:
    sample_id: str
    probability_map: np.ndarray
    mask: np.ndarray
    polygons: List[Polygon]


def _mask_to_polygons(mask: np.ndarray, resolution_m: float) -> List[Polygon]:
    """Converts a binary mask into a list of shapely polygons."""

    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons: List[Polygon] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        contour *= resolution_m
        polygon = Polygon(contour)
        if polygon.area > 1e-2:
            polygons.append(polygon)
    return polygons


def _load_model(model_path: Path, input_channels: int, device: torch.device) -> UNet:
    model = UNet(n_channels=input_channels)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _prepare_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


def infer_sample(
    sample_dir: Path,
    model: UNet,
    device: torch.device,
    threshold: float,
    resolution_m: float,
    min_area_m2: float,
) -> PredictionResult:
    sample_id = sample_dir.name
    image, _, _, metadata = load_sample(sample_dir)
    tensor = _prepare_tensor(image).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0, 0]
    mask = (probabilities >= threshold).astype(np.uint8)
    pixel_size = float(metadata.get("pixel_size_m", resolution_m))
    polygons = [poly for poly in _mask_to_polygons(mask, pixel_size) if poly.area >= min_area_m2]
    return PredictionResult(sample_id, probabilities, mask, polygons)


def run_inference(config: InferenceConfig) -> List[PredictionResult]:
    """Runs inference on a dataset directory and exports geojson results."""

    dataset_dir = Path(config.dataset_dir)
    sample_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No samples found in {dataset_dir}")

    example_image, _, _, _ = load_sample(sample_dirs[0])
    input_channels = example_image.shape[0]
    device = torch.device(getattr(config, "device", "cpu"))
    model = _load_model(config.model_path, input_channels, device)

    results: List[PredictionResult] = []
    for sample_dir in sample_dirs:
        result = infer_sample(
            sample_dir,
            model,
            device,
            config.probability_threshold,
            config.route_graph_resolution_m,
            config.min_polygon_area_m2,
        )
        results.append(result)

    export_path = dataset_dir / "predictions.geojson"
    features = []
    for result in results:
        for polygon in result.polygons:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"sample_id": result.sample_id},
                    "geometry": mapping(polygon),
                }
            )
    geojson = {"type": "FeatureCollection", "features": features}
    export_path.write_text(json.dumps(geojson, indent=2))

    return results

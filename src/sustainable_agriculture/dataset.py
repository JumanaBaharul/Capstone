"""Synthetic dataset generation for sustainable precision agriculture."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import DatasetConfig, DiseaseSimulationConfig, FieldConfig, SpectralBandConfig, WeatherConfig


def _create_mesh(field: FieldConfig) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Returns meshgrid coordinates representing the center of each pixel."""

    x_coords = np.linspace(0, field.width_m, int(field.width_m / field.resolution_m), endpoint=False, dtype=np.float32)
    y_coords = np.linspace(0, field.height_m, int(field.height_m / field.resolution_m), endpoint=False, dtype=np.float32)
    xv, yv = np.meshgrid(x_coords, y_coords)
    return xv, yv


def _generate_disease_clusters(
    xv: NDArray[np.float32],
    yv: NDArray[np.float32],
    disease_cfg: DiseaseSimulationConfig,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Generates a smooth disease probability mask using radial basis functions."""

    h, w = xv.shape
    probability = np.zeros((h, w), dtype=np.float32)
    cluster_count = rng.integers(1, disease_cfg.max_cluster_count + 1)

    for _ in range(cluster_count):
        center_x = rng.uniform(0, xv.max())
        center_y = rng.uniform(0, yv.max())
        radius = rng.uniform(disease_cfg.min_cluster_radius, disease_cfg.max_cluster_radius)
        influence = np.exp(-(((xv - center_x) ** 2 + (yv - center_y) ** 2) / (2 * radius**2)))
        probability += influence

    probability /= probability.max(initial=1.0)
    probability *= disease_cfg.diseased_fraction * rng.uniform(0.8, 1.2)
    return probability.clip(0.0, 1.0)


def _generate_spectral_cube(
    probability: NDArray[np.float32],
    bands: Tuple[SpectralBandConfig, ...],
    rng: np.random.Generator,
    disease_cfg: DiseaseSimulationConfig,
) -> NDArray[np.float32]:
    """Creates a multi-band spectral cube with disease-induced reflectance changes."""

    height, width = probability.shape
    cube = np.zeros((len(bands), height, width), dtype=np.float32)

    base_reflectance = rng.uniform(0.15, 0.45, size=(len(bands), 1, 1)).astype(np.float32)
    healthy_variability = rng.normal(0.0, 0.02, size=(len(bands), height, width)).astype(np.float32)
    disease_effect = -probability * disease_cfg.spectral_drop

    for band_idx, band in enumerate(bands):
        noise = rng.normal(0.0, band.noise_std, size=(height, width)).astype(np.float32)
        cube[band_idx] = base_reflectance[band_idx] + healthy_variability[band_idx] + disease_effect + noise

    return cube.clip(0.0, 1.0)


def _generate_weather_features(
    weather_cfg: WeatherConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Samples weather covariates that can act as auxiliary inputs for the model."""

    temperature = rng.uniform(*weather_cfg.temperature_range_c)
    humidity = rng.uniform(*weather_cfg.humidity_range)
    rainfall = rng.uniform(*weather_cfg.rainfall_range_mm)

    return {
        "temperature_c": float(np.round(temperature, 2)),
        "humidity": float(np.round(humidity, 3)),
        "rainfall_mm": float(np.round(rainfall, 2)),
    }


def generate_sample(
    sample_id: int,
    config: DatasetConfig,
    rng: np.random.Generator | None = None,
) -> Dict[str, NDArray[np.float32]]:
    """Generates a single synthetic field sample and returns its components."""

    rng = np.random.default_rng() if rng is None else rng
    xv, yv = _create_mesh(config.field)
    probability = _generate_disease_clusters(xv, yv, config.disease, rng)
    spectral_cube = _generate_spectral_cube(probability, tuple(config.bands), rng, config.disease)
    weather = _generate_weather_features(config.weather, rng)
    mask = (probability > config.disease.diseased_fraction / 2).astype(np.float32)

    return {
        "image": spectral_cube,
        "mask": mask,
        "probability": probability,
        "weather": weather,
    }


def save_sample(
    sample: Dict[str, NDArray[np.float32]],
    output_dir: Path,
    sample_id: int,
    bands: Tuple[SpectralBandConfig, ...],
    field: FieldConfig,
) -> None:
    """Persists a sample to disk as NPZ arrays and JSON metadata."""

    sample_dir = output_dir / f"sample_{sample_id:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sample_dir / "data.npz",
        image=sample["image"],
        mask=sample["mask"],
        probability=sample["probability"],
    )

    metadata = {
        "weather": sample["weather"],
        "bands": [asdict(band) for band in bands],
        "field_dimensions_m": {
            "width": float(field.width_m),
            "height": float(field.height_m),
        },
        "pixel_size_m": float(field.resolution_m),
        "image_shape": sample["image"].shape,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def generate_dataset(config: DatasetConfig) -> None:
    """Generates a full dataset according to the provided configuration."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for idx in range(config.sample_count):
        sample = generate_sample(idx, config, rng)
        save_sample(sample, output_dir, idx, tuple(config.bands), config.field)


def load_sample(sample_dir: Path) -> Tuple[NDArray[np.float32], NDArray[np.float32], Dict[str, float], Dict[str, object]]:
    """Loads a previously generated sample from disk along with metadata."""

    npz_path = sample_dir / "data.npz"
    with np.load(npz_path) as data:
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

    metadata = json.loads((sample_dir / "metadata.json").read_text())
    weather = metadata.get("weather", {})
    return image, mask, weather, metadata

"""Configuration data structures for the Sustainable Precision Agriculture pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass(slots=True)
class SpectralBandConfig:
    """Defines metadata for a spectral band in the simulated dataset."""

    name: str
    central_wavelength_nm: float
    noise_std: float = 0.01


@dataclass(slots=True)
class DiseaseSimulationConfig:
    """Parameters controlling the generation of synthetic disease patterns."""

    diseased_fraction: float = 0.2
    max_cluster_count: int = 5
    min_cluster_radius: float = 10.0
    max_cluster_radius: float = 40.0
    spectral_drop: float = 0.2


@dataclass(slots=True)
class WeatherConfig:
    """Parameters for the simulated weather covariates."""

    temperature_range_c: tuple[float, float] = (15.0, 35.0)
    humidity_range: tuple[float, float] = (0.45, 0.95)
    rainfall_range_mm: tuple[float, float] = (0.0, 20.0)


@dataclass(slots=True)
class FieldConfig:
    """Describes the spatial configuration of a simulated field tile."""

    width_m: float = 500.0
    height_m: float = 500.0
    resolution_m: float = 10.0


@dataclass(slots=True)
class DatasetConfig:
    """Aggregates dataset-level parameters for synthetic data generation."""

    output_dir: Path
    sample_count: int = 100
    bands: Sequence[SpectralBandConfig] = field(
        default_factory=lambda: [
            SpectralBandConfig("B02", 490.0),
            SpectralBandConfig("B03", 560.0),
            SpectralBandConfig("B04", 665.0),
            SpectralBandConfig("B08", 842.0),
        ]
    )
    disease: DiseaseSimulationConfig = field(default_factory=DiseaseSimulationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    field: FieldConfig = field(default_factory=FieldConfig)


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters for model training."""

    dataset_dir: Path
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    device: str = "cpu"


@dataclass(slots=True)
class InferenceConfig:
    """Configuration for inference and treatment planning."""

    model_path: Path
    dataset_dir: Path
    probability_threshold: float = 0.5
    min_polygon_area_m2: float = 50.0
    route_graph_resolution_m: float = 20.0
    device: str = "cpu"


@dataclass(slots=True)
class VisualizationConfig:
    """Parameters for exporting interactive treatment maps."""

    map_output: Path
    colormap: str = "YlOrRd_09"
    zoom_start: int = 17


def ensure_directory(path: Path) -> Path:
    """Creates a directory if it does not exist and returns the resolved path."""

    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_spectral_band_names(bands: Sequence[SpectralBandConfig]) -> List[str]:
    """Returns the names of the configured spectral bands."""

    return [band.name for band in bands]

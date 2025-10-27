"""Training utilities for the Sustainable Precision Agriculture project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

from .config import TrainingConfig, default_spectral_band_names
from .dataset import DatasetConfig, load_sample
from .model import dice_coefficient, make_model


class FieldSampleDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset wrapping the generated synthetic samples."""

    def __init__(self, root_dir: Path, include_weather: bool = True) -> None:
        self.root_dir = Path(root_dir)
        self.sample_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.include_weather = include_weather

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image, mask, weather, _ = load_sample(sample_dir)
        if self.include_weather and weather:
            weather_values = torch.tensor([weather[key] for key in sorted(weather.keys())], dtype=torch.float32)
            weather_channels = weather_values.view(-1, 1, 1).repeat(1, image.shape[1], image.shape[2])
            tensor_image = torch.from_numpy(image)
            tensor_image = torch.cat([tensor_image, weather_channels], dim=0)
        else:
            tensor_image = torch.from_numpy(image)
        tensor_mask = torch.from_numpy(mask).unsqueeze(0)
        return tensor_image, tensor_mask


@dataclass
class TrainingStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_dice: float


def _split_dataset(dataset: Dataset, validation_split: float) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def train_model(config: TrainingConfig, bands: List[str] | None = None) -> Dict[str, List[float]]:
    """Trains the segmentation model and saves weights to disk."""

    dataset = FieldSampleDataset(config.dataset_dir)
    train_dataset, val_dataset = _split_dataset(dataset, config.validation_split)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    sample_channels = train_dataset[0][0].shape[0]
    device = torch.device(config.device)
    model, criterion = make_model(sample_channels, device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_loss = float("inf")
    weights_path = config.dataset_dir / "model.pt"

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(torch.tensor(train_losses).mean().item())
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_losses: List[float] = []
        val_dice_scores: List[float] = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_losses.append(loss.item())
                val_dice_scores.append(float(dice_coefficient(outputs, masks).item()))

        avg_val_loss = float(torch.tensor(val_losses).mean().item()) if val_losses else avg_train_loss
        avg_val_dice = float(torch.tensor(val_dice_scores).mean().item()) if val_dice_scores else 0.0
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), weights_path)

        print(
            f"Epoch {epoch}/{config.num_epochs} - "
            f"train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - val_dice: {avg_val_dice:.4f}"
        )

    history_path = config.dataset_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    metadata = {
        "bands": bands or [],
        "config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
        },
        "weights_path": str(weights_path),
    }
    (config.dataset_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return history


def load_training_config(dataset_dir: Path, dataset_config: DatasetConfig | None = None) -> TrainingConfig:
    """Utility to create a TrainingConfig based on dataset metadata."""

    bands = default_spectral_band_names(dataset_config.bands) if dataset_config else []
    return TrainingConfig(dataset_dir=dataset_dir, device="cpu")

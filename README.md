# Sustainable Precision Agriculture via Disease Localization and Optimized Treatment Pathways

This repository contains a fully reproducible simulation of a precision agriculture system that:

1. Generates synthetic multi-modal remote sensing datasets.
2. Trains a deep learning segmentation network to localise diseased vegetation.
3. Derives geospatial polygons representing infected zones and plans an optimised spraying route.
4. Visualises the recommended treatment plan on an interactive map.

The codebase is intentionally modular so each stage can be run independently or via the provided end-to-end pipeline script.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate a synthetic dataset**

   ```bash
   python scripts/generate_dataset.py data/generated/dataset --samples 50
   ```

3. **Train the segmentation model**

   ```bash
   python scripts/train_model.py data/generated/dataset --epochs 5
   ```

4. **Run inference and produce the full pipeline outputs**

   ```bash
   python scripts/run_pipeline.py workspace --samples 50 --epochs 5
   ```

   The pipeline will generate:

   - `workspace/dataset` containing the simulated imagery and masks.
   - `workspace/dataset/model.pt` with trained U-Net weights.
   - `workspace/dataset/predictions.geojson` describing diseased regions.
   - `workspace/treatment_route.json` summarising the optimised spraying path.
   - `workspace/treatment_map.html` visualising predictions and routes (unless `--skip-visualisation` is provided).

## Project Structure

- `src/sustainable_agriculture/`
  - `config.py`: Dataclasses encapsulating configuration for dataset generation, training, inference, and visualisation.
  - `dataset.py`: Synthetic data generator leveraging stochastic disease patterns and spectral perturbations.
  - `model.py`: Lightweight U-Net architecture with Dice metric helper.
  - `train.py`: PyTorch training loop and dataset wrapper.
  - `inference.py`: Model inference routines, mask to polygon conversion, and GeoJSON export.
  - `route_planner.py`: Graph-based route optimisation using a Travelling Salesperson approximation.
  - `visualization.py`: Folium-based mapping helper for interactive treatment plans.
- `scripts/`: Command-line entry points for each pipeline stage and a comprehensive end-to-end runner.

## Notes

- The dataset generator uses fixed random seeds for reproducibility.
- Weather covariates are encoded as additional channels during training to emulate multi-modal fusion.
- The geospatial conversion assumes small-area planar approximations when converting metres to latitude/longitude for mapping purposes.

## License

This project is provided for academic and research purposes.

# Solar GHI Forecasting with Disentangled Uncertainty Quantification

Short-term Global Horizontal Irradiance (GHI) forecasting at horizons of 1, 3, and 6 hours,
driven by GOES-16 satellite imagery and ground-station measurements.
The project evaluates two spatial-temporal deep learning architectures (ResNet+LSTM,
GraphSAGE+LSTM, including a fixed-vs-learned graph-construction ablation for GraphSAGE)
and a flat MLP baseline against SARIMA and persistence, then layers a disentangled
aleatoric/epistemic uncertainty-quantification pipeline (a Gamma-likelihood variance
network + SGLD posterior sampling) on top of the best backbone, so forecasts carry not just an
interval but a diagnosis of *why* they are uncertain. Split Conformal Prediction is also
implemented in code as an alternative post-hoc UQ method.
Benchmarked on two Colombian stations with contrasting tropical climates: El Paso (semi-arid,
Caribbean lowlands) and Uniandes (equatorial Andean, Bogotá).

---

## Study sites

| Site | Coordinates | Elevation | Climate | Data period |
|------|------------|-----------|---------|-------------|
| **El Paso** (César, Colombia) | 9.737° N, 73.695° W | ~50 m | Tropical semi-arid | Mar 2022 – Mar 2024 |
| **Uniandes** (Bogotá, Colombia) | 4.602° N, 74.066° W | ~2 600 m | Equatorial mountain | Sep 2023 – Mar 2025 |

---

## Repository structure

```
.
├── configs/          # Experiment configuration (paths, splits, patch size)
├── data/             # Processed data: ground_aligned, datasets, patches, metadata — not in git
├── data_raw/         # Raw ground-station CSVs — not in git
├── data_processed/   # Raw GOES-16 NetCDF and processed satellite data — not in git
├── docs/             # LaTeX manuscript, slides, and progress notes
├── notebooks/        # Data-preparation and exploratory notebooks
├── results/          # Aggregated metrics and publication figures
├── runs/             # Per-run summaries (summary.json); checkpoints not in git
├── scripts/          # Numbered pipeline entrypoints (data → training → HPO → UQ → figures)
├── src/solar_uq/     # Core Python package: datasets, models, training, metrics, UQ
├── tests/            # Unit tests
├── pyproject.toml    # Package build config (pip install -e .)
├── requirements.txt  # Pinned dependencies
└── run_sequential.sh # Sequential experiment runner (skip-if-done)
```

---

## Setup

```bash
git clone <repo-url>
cd Proyecto_e_ladino

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt   # pins CUDA 13 PyTorch build
pip install -e .                   # installs solar_uq package
```

> **PyTorch note:** `requirements.txt` pins a CUDA 13 build.
> For a different CUDA version, install the matching wheel from
> [pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`.

Verify CUDA availability:
```bash
.venv/bin/python scripts/gpu_check.py
```

---

## Key results (test set)

| Arch | Site | Horizon | skill_day |
|------|------|---------|-----------|
| GraphSAGE+LSTM (tuned k-NN) | El Paso | 6 h | **0.604 ± 0.007** |
| ResNet+LSTM | El Paso | 6 h | 0.588 ± 0.030 |
| GraphSAGE+LSTM (fixed graph) | Uniandes | 6 h | 0.417 ± 0.008 |
| ResNet+LSTM | Uniandes | 1 h | 0.161 ± 0.014 |

El Paso GraphSAGE+LSTM uses the tuned, distance-weighted k-NN graph (2 seeds {42, 1});
Uniandes uses the fixed unweighted graph (4 seeds). Full table: `results/summary.md`.

---

## Citation

```bibtex
@article{ladino2026solar,
  title   = {[Title — to be added]},
  author  = {Ladino, Esteban and others},
  journal = {[Venue]},
  year    = {2026}
}
```

---

## License

To be specified upon publication.

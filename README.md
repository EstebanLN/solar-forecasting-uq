# Short-Term Solar Irradiance Forecasting at Contrasting Colombian Sites

Short-term Global Horizontal Irradiance (GHI) forecasting at horizons of 1, 3, and
6 hours, from GOES-16 geostationary satellite imagery and ground-station
measurements, benchmarked at two climatically contrasting Colombian sites.

The project compares graph-based and convolutional spatio-temporal deep learning —
**GraphSAGE-LSTM**, which treats each satellite patch as a graph of pixels (with a
fixed vs. tuned distance-weighted graph-construction ablation), and **ResNet-LSTM** —
against a **FlatMLP** spatial-ablation baseline, **SARIMA**, and **persistence**. A
**multisource residual-fusion** variant augments the satellite encoder with co-located
surface features. On top of the best backbone, an **SGLD posterior-sampling layer**
provides epistemic uncertainty intended to flag forecasts produced under data-scarce
conditions (under validation).

This repository accompanies the manuscript *"Short-Term Solar Irradiance Forecasting
at Contrasting Colombian Sites: Graph and Convolutional Deep Learning with Multisource
Fusion and SGLD Epistemic Uncertainty"* (Ladino Nieto & Villarraga Florez, in
preparation). See [`CITATION.cff`](CITATION.cff).

---

## Study sites

| Site | Coordinates | Elevation | Climate | Data period |
|------|-------------|-----------|---------|-------------|
| **El Paso** (César, Colombia) | 9.737° N, 73.695° W | ~50 m | High-irradiance Caribbean lowland | Mar 2022 – Mar 2024 |
| **Uniandes** (Bogotá, Colombia) | 4.602° N, 74.066° W | ~2 600 m | Equatorial Andean mountain | Sep 2023 – Mar 2025 |

---

## Repository structure

```
.
├── configs/          # Experiment configuration (paths, splits, patch size)
├── data/             # Processed data: ground_aligned, datasets, patches — not in git
├── data_raw/         # Raw ground-station CSVs — not in git
├── data_processed/   # Raw GOES-16 NetCDF and processed satellite data — not in git
├── notebooks/        # Data-preparation and exploratory notebooks
├── results/          # Aggregated metrics (summary.md, summary.csv) and figures
├── runs/             # Per-run summaries (summary.json); checkpoints not in git
├── scripts/          # Numbered pipeline entrypoints (data → training → HPO → UQ → figures)
├── src/solar_uq/     # Core Python package: datasets, models, training, metrics, UQ
├── tests/            # Unit tests
├── pyproject.toml    # Package build config (pip install -e .)
└── requirements.txt  # Pinned dependencies
```

---

## Setup

```bash
git clone <repo-url>
cd Proyecto_e_ladino

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt   # pins a CUDA 13 PyTorch build
pip install -e .                   # installs the solar_uq package
```

> **PyTorch note:** `requirements.txt` pins a CUDA 13 build. For a different CUDA
> version, install the matching wheel from [pytorch.org](https://pytorch.org)
> before running `pip install -r requirements.txt`.

Verify CUDA availability:
```bash
.venv/bin/python scripts/gpu_check.py
```

---

## Reproducing an experiment

The `scripts/` directory is a numbered pipeline (data preparation → training/HPO →
uncertainty → figures). Each training script runs an Optuna hyperparameter search
for one `(site, horizon, seed)` combination and writes a run directory under `runs/`.

```bash
# GraphSAGE-LSTM at Uniandes, 1-hour horizon, seed 42
.venv/bin/python scripts/06_graphsage_lstm_optuna.py \
  --site uniandes --hours_ahead 1 --seed 42 \
  --n_trials 75 --runs_root runs/graphsage_lstm_optuna_v2

# Multisource fusion variant (satellite + surface features)
.venv/bin/python scripts/06_resnet_lstm_optuna.py --fusion \
  --site elpaso --hours_ahead 6 --seed 42 \
  --n_trials 75 --runs_root runs/fusion_resnet_lstm

# SGLD epistemic-uncertainty layer, warm-started from the tuned backbone
.venv/bin/python scripts/08_sgld.py --arch resnet --site uniandes --hours_ahead 1 --seed 42
```

Aggregate all completed runs into the results table, and run the tests:
```bash
.venv/bin/python scripts/09_results_table.py   # → results/summary.{md,csv}
.venv/bin/python -m pytest -q tests/
```

---

## Key results (test set, skill vs. persistence)

| Model | Site | Horizon | skill_day |
|-------|------|---------|-----------|
| **Fusion ResNet-LSTM** (satellite + surface) | El Paso | 6 h | **0.725** *(1 seed, in progress)* |
| GraphSAGE-LSTM (tuned k-NN) | El Paso | 6 h | 0.604 ± 0.007 |
| ResNet-LSTM | El Paso | 6 h | 0.579 ± 0.018 |
| GraphSAGE-LSTM (fixed graph) | Uniandes | 6 h | 0.418 ± 0.011 |
| ResNet-LSTM | Uniandes | 1 h | 0.161 ± 0.014 |

Graph-based encoders are competitive with or better than convolutional ones at
multi-hour horizons; absolute skill is markedly lower at the high-altitude mountain
site, and multisource fusion delivers the largest gains where satellite and surface
signals are complementary. Full table: [`results/summary.md`](results/summary.md).

> The codebase also contains exploratory uncertainty components — a Gamma-likelihood
> variance network (`src/solar_uq/variance_net.py`) and split conformal prediction
> (`src/solar_uq/conformal.py`) — that are outside the current manuscript scope.

---

## Citation

If you use this code, please cite the repository via [`CITATION.cff`](CITATION.cff).
Once the manuscript is published this will be updated to the article reference:

```bibtex
@article{ladino2026solar,
  title   = {Short-Term Solar Irradiance Forecasting at Contrasting Colombian Sites:
             Graph and Convolutional Deep Learning with Multisource Fusion and SGLD
             Epistemic Uncertainty},
  author  = {Ladino Nieto, Esteban and Villarraga Florez, Daniel Francisco},
  journal = {(in preparation)},
  year    = {2026}
}
```

---

## License

To be specified upon publication.

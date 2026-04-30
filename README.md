# Solar GHI Forecasting with Deep Learning and Conformal Uncertainty Quantification

> **Paper:** *[Title — to be added upon publication]*
> **Authors:** Esteban Ladino · *[co-authors]*
> **Venue:** *[Conference / Journal, Year]*

---

## Overview

Short-term solar irradiance forecasting is a critical enabler for the reliable
integration of photovoltaic (PV) generation into electricity grids.
This repository accompanies the paper above and provides the full implementation
for reproducing all experiments, baselines, and uncertainty-quantification results.

The system produces **point forecasts** of Global Horizontal Irradiance (GHI)
at horizons of 1, 3, and 6 hours ahead, together with **conformal prediction
intervals** that carry finite-sample coverage guarantees.
It is evaluated on two measurement stations in Colombia with contrasting
climatic regimes — a semi-arid Caribbean lowland (El Paso, César) and a
high-altitude equatorial Andean site (Uniandes, Bogotá).

---

## Motivation and background

### The solar forecasting challenge

Solar irradiance at the surface is a function of the deterministic solar
geometry — well described by clear-sky models — and a stochastic cloud
attenuation component that is inherently difficult to predict.
At **intra-hour horizons**, persistence (the assumption that current irradiance
will not change) is a hard baseline to beat because cloud patterns evolve slowly.
At **multi-hour horizons** (3–6 h), cloud advection, formation, and dissipation
introduce large uncertainty, and spatial information becomes essential.

Grid operators need accurate multi-hour forecasts to schedule dispatchable backup
generation, manage energy storage, and trade on day-ahead markets.
Beyond point forecasts, **calibrated uncertainty estimates** are increasingly
required by operational decision tools that optimise under risk.

### Satellite-based approaches

Geostationary satellite imagery, particularly from the GOES-R series, provides
near-real-time cloud observations at high temporal (10-minute) and spatial
(~2 km at nadir) resolution across the Americas.
The ABI-L2-MCMIPF product delivers 16 spectral channels spanning visible,
near-infrared, and thermal-infrared, capturing cloud-top height, optical depth,
and microphysics simultaneously.
Integrating this spatial context into a forecasting model is the key
differentiator of deep-learning approaches over purely statistical time-series
methods.

### Graph-based spatial encoding

Convolutional neural networks (CNNs) are the standard choice for encoding
raster satellite patches, but they impose translational-equivariance assumptions
that may not hold for irregular cloud structures.
Graph Neural Networks (GNNs) offer a flexible alternative: pixels are treated
as nodes in an adjacency graph, and message-passing layers aggregate spatial
context without assuming a fixed kernel structure.
This work evaluates whether **GraphSAGE**-based encoding provides a meaningful
advantage over a compact **ResNet** encoder on tropical cloud regimes that
exhibit strong convective organisation.

### Uncertainty quantification

Most solar forecasting papers report only point-forecast accuracy.
Operational use, however, demands prediction *intervals* with known coverage
properties.
**Conformal prediction** provides a distribution-free post-hoc wrapper that
converts any trained point-forecast model into a calibrated interval predictor
with a finite-sample marginal coverage guarantee — no retraining or likelihood
assumptions required.
This makes it a practically attractive add-on to any existing deep-learning
forecasting pipeline.

---

## Contributions

1. **Two spatial-temporal encoder–decoder architectures** — ResNet+LSTM and
   GraphSAGE+LSTM — trained end-to-end on GOES-16 satellite patches and
   ground-station GHI series, evaluated across three forecast horizons.

2. **A paired tropical / sub-tropical benchmark** using two Colombian stations
   with substantially different cloud climatologies, enabling cross-site
   generalisation analysis.

3. **A rigorous SARIMA statistical baseline** on the clear-sky index, following
   the solar-forecasting literature convention, with documented parameter
   selection rationale.

4. **Hyperparameter optimisation via Optuna** for both architectures, with
   transparent reporting of the search space and convergence.

5. **Split Conformal Prediction intervals** on top of the trained DL models,
   with empirical coverage validation stratified by daytime hours.

---

## Study sites

| Site | Location | Elevation | Climate | GHI data period |
|------|----------|-----------|---------|-----------------|
| **El Paso** (César, Colombia) | 9.737° N, 73.695° W | ~50 m | Semi-arid tropical (BSh) | Mar 2022 – Mar 2024 |
| **Uniandes** (Bogotá, Colombia) | 4.602° N, 74.066° W | ~2 600 m | Andean equatorial (Cfb) | Sep 2023 – Mar 2025 |

Both stations record GHI at 10-minute resolution.
El Paso is located in the Caribbean lowlands and is characterised by high mean
irradiance and moderate cloudiness driven by trade-wind cumuli.
Uniandes sits on the Eastern Andes at 2 600 m and experiences strong
convective cloud development in the afternoon, making its irradiance
highly variable intraday.

---

## Data sources

### Ground station measurements

In-situ GHI time series at 10-minute UTC resolution.
The processed Parquet files used in this work are available upon reasonable
request to the authors. Raw data from the Uniandes station is owned by
Universidad de los Andes; El Paso data is from an in-situ meteorological station
at that location.

### GOES-16 ABI-L2-MCMIPF satellite imagery

Full-disk, 16-channel, multi-band cloud and moisture imagery from the
GOES-R Series archive, freely available via NOAA's Amazon S3 bucket:

```
s3://noaa-goes16/ABI-L2-MCMIPF/<year>/<doy>/<hour>/
```

For each station a **16×16-pixel patch** centred on the site is extracted
from the full-disk image at each timestamp, using the georeferencing procedure
in `notebooks/03_mcmipf_georeferencing_site_centers.ipynb`.
Pixel coordinates for each site are documented in that notebook.

---

## Methodology summary

```
GOES-16 patches (16×16×16 channels)
        │
        ▼
┌───────────────────────┐
│  Spatial encoder      │   ResNet  OR  GraphSAGE
│  (one frame at t−L…t) │   → embedding per timestep
└───────────┬───────────┘
            │  sequence of L=24 embeddings (4 h history)
            ▼
┌───────────────────────┐
│  Temporal decoder     │   LSTM
│                       │   → point forecast ŷ(t+H)
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Split Conformal CP   │   calibrated on val set
│                       │   → interval [ŷ ± q̂] at 1−α coverage
└───────────────────────┘
```

- **Input:** last L = 24 steps (4 h) of satellite patches + scalar GHI history
- **Output:** GHI point forecast at H ∈ {6, 18, 36} steps ahead (1, 3, 6 h)
- **Normalisation:** target GHI is z-score normalised using training-set statistics
- **Loss:** mean squared error (MSE) on normalised target
- **Optimiser:** Adam with weight decay; cosine annealing not used; early stopping on validation RMSE_day

---

## Repository structure

```
.
├── data/                          # Processed data (not tracked in git)
│   ├── ground_aligned/            #   10-min UTC GHI parquet files
│   ├── datasets/manifest_v1/      #   Supervised manifests per site and horizon
│   └── patches_v1/                #   Pre-extracted satellite patches (.npz)
├── notebooks/
│   ├── 01_ground_eda.ipynb                          # Ground EDA and UTC alignment
│   ├── 02_ground_splits_and_baselines.ipynb         # Temporal splits, persistence
│   ├── 03_mcmipf_georeferencing_site_centers.ipynb  # Satellite georeferencing
│   ├── 04_Manifest Build.ipynb                      # Supervised manifest builder
│   └── Support_build_patch_store.ipynb              # Patch extraction from GOES-16
├── scripts/
│   ├── 05_resnet_lstm_baseline.py       # Train ResNet+LSTM (fixed hparams)
│   ├── 05_graphsage_lstm_baseline.py    # Train GraphSAGE+LSTM (fixed hparams)
│   ├── 05_sarima_baseline.py            # SARIMA on clear-sky index
│   ├── 06_resnet_lstm_optuna.py         # Optuna HPO — ResNet+LSTM
│   ├── 06_graphsage_lstm_optuna.py      # Optuna HPO — GraphSAGE+LSTM
│   ├── 07_conformal_explore.py          # Split Conformal Prediction evaluation
│   └── 08_results_table.py             # Aggregate runs → CSV / LaTeX table
├── src/solar_uq/                  # Core Python package
│   ├── models/
│   │   ├── resnet_lstm.py         #   SmallResNet encoder + LSTM
│   │   └── graphsage_lstm.py      #   GraphSAGE encoder + LSTM (pure PyTorch)
│   ├── data.py                    #   Datasets, patch loading, normaliser
│   ├── train.py                   #   Training loop, evaluation helpers
│   ├── conformal.py               #   SplitCP calibration and evaluation
│   └── metrics.py                 #   RMSE, MAE, skill score, persistence
├── results/
│   ├── summary.csv                # Aggregated test-set metrics (all runs)
│   └── summary.md
├── run_sequential.sh              # GPU-safe sequential experiment runner
└── requirements.txt
```

---

## Requirements

| Dependency | Version tested |
|------------|---------------|
| Python | 3.12 |
| PyTorch | 2.11 (CUDA 13) |
| statsmodels | 0.14.6 |
| optuna | 4.7.0 |
| pvlib | 0.15.0 |
| pandas | 3.0 |
| pyarrow | 23.0 |
| numpy | 2.4 |
| scipy | 1.17 |

> GraphSAGE is implemented in **pure PyTorch** — no `torch_geometric` dependency.

A GPU with ≥ 12 GB VRAM is recommended for training.
All experiments in the paper were run on an NVIDIA RTX 5070.

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd Proyecto_e_ladino
```

### 2. Virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch note:** `requirements.txt` pins a CUDA 13 build.
> If your system uses a different CUDA version, install the matching
> PyTorch wheel from [pytorch.org](https://pytorch.org) before running the
> command above.

---

## Reproducing experiments

All commands are run **from the project root** with the virtual environment active.

### Step 0 — Prepare data

Follow the notebooks in order:

```
notebooks/01_ground_eda.ipynb                         # align and QC ground data
notebooks/03_mcmipf_georeferencing_site_centers.ipynb # identify site pixels
notebooks/Support_build_patch_store.ipynb             # extract satellite patches
notebooks/04_Manifest Build.ipynb                     # build supervised manifests
```

If you received the pre-processed data files from the authors, place them under
`data/` as described in §Data sources and skip directly to Step 1.

Expected directory layout after data preparation:

```
data/
├── ground_aligned/
│   ├── ground_10min_utc_elpaso.parquet
│   └── ground_10min_utc_uniandes.parquet
├── datasets/manifest_v1/
│   ├── elpaso/{h1,h3,h6}/{manifest_train,manifest_val,manifest_test}.parquet
│   └── uniandes/{h1,h3,h6}/...
└── patches_v1/
    ├── elpaso/P16/{year}/{mm}/{YYYYMMDD_HH_patch.npz}
    └── uniandes/P16/...
```

### Step 1 — Deep learning baselines

Trains both architectures with fixed hyperparameters across all
(site, horizon, seed) combinations (60 runs total).
The script skips any run that already has a `summary.json`.

```bash
bash run_sequential.sh baseline          # both architectures, both sites
bash run_sequential.sh baseline elpaso   # El Paso only
bash run_sequential.sh baseline uniandes # Uniandes only
```

To run a single experiment:

```bash
.venv/bin/python scripts/05_resnet_lstm_baseline.py \
    --site elpaso --hours_ahead 6 --seed 42

.venv/bin/python scripts/05_graphsage_lstm_baseline.py \
    --site uniandes --hours_ahead 3 --seed 7
```

Outputs are saved to `runs/resnet_lstm/<run_name>/` and
`runs/graphsage_lstm/<run_name>/`, each containing:
- `best_model.pt` — model checkpoint with architecture metadata
- `summary.json` — training curve, val metrics, and test metrics

### Step 2 — SARIMA baseline

```bash
.venv/bin/python scripts/05_sarima_baseline.py \
    --site elpaso --hours_ahead 1 3 6

.venv/bin/python scripts/05_sarima_baseline.py \
    --site uniandes --hours_ahead 1 3 6
```

The script fits SARIMAX(2,1,2)(1,1,1)₂₄ on the hourly clear-sky index
and evaluates all three horizons from a single model fit per site.
Output: `runs/sarima/<run_name>/summary.json`.

### Step 3 — Hyperparameter optimisation (Optuna)

Runs 20 Optuna trials per (architecture, site, horizon, seed), then retrains
a final model with the best configuration found (24 runs per architecture).

```bash
bash run_sequential.sh optuna            # both architectures, both sites
bash run_sequential.sh resnet_optuna     # ResNet+LSTM only
bash run_sequential.sh gsage_optuna      # GraphSAGE+LSTM only

bash run_sequential.sh optuna uniandes   # both architectures, Uniandes only
```

Individual run:

```bash
.venv/bin/python scripts/06_resnet_lstm_optuna.py \
    --site elpaso --hours_ahead 6 --seed 42 --n_trials 20

.venv/bin/python scripts/06_graphsage_lstm_optuna.py \
    --site uniandes --hours_ahead 6 --seed 1 --n_trials 20
```

> **Runtime note:** each Optuna run takes approximately 45–90 minutes on an RTX 5070.
> The full 48-run sweep requires 4–5 days when executed sequentially.

### Step 4 — Conformal prediction intervals

Applies Split CP calibration to any completed run's validation set, then
evaluates interval coverage and width on the test set.

```bash
.venv/bin/python scripts/07_conformal_explore.py \
    --run_dir runs/graphsage_lstm/<run_name> \
    --alphas 0.05 0.10 0.20
```

The script auto-detects the architecture from the checkpoint metadata,
reconstructs the model exactly, and saves results to
`<run_dir>/conformal_splitcp.json`.

### Step 5 — Aggregate results

```bash
.venv/bin/python scripts/08_results_table.py
```

Writes `results/summary.csv` and `results/summary.md` by scanning all
`runs/*/summary.json` files.

---

## Hyperparameter reference

**ResNet+LSTM — fixed baseline configuration**

| Parameter | Value |
|-----------|-------|
| Conv base channels | 32 |
| Embedding dimension | 128 |
| LSTM hidden size | 128 |
| Dropout | 0.10 |
| Learning rate | 2 × 10⁻³ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping patience | 8 |

**GraphSAGE+LSTM — fixed baseline configuration**

| Parameter | Value |
|-----------|-------|
| Graph hidden channels | 64 |
| GraphSAGE layers | 2 |
| LSTM layers | 1 |
| Dropout (head) | 0.10 |
| Input batch normalisation | enabled |
| Neighbourhood aggregation | concat (self ‖ mean of neighbours) |

**Optuna search space (both architectures)**

| Hyperparameter | Type | Range |
|----------------|------|-------|
| Base / hidden channels | categorical | {16, 24, 32, 48} |
| Embedding dimension | categorical | {64, 128, 192} |
| LSTM hidden size | categorical | {64, 128, 192} |
| Dropout | categorical | {0.0, 0.1, 0.2, 0.3} |
| Learning rate | log-uniform | [3 × 10⁻⁴, 3 × 10⁻³] |
| Weight decay | log-uniform | [10⁻⁶, 10⁻³] |
| Batch size | categorical | {16, 32, 64} |

---

## Citation

If you use this code, data, or methodology in your research, please cite:

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

[To be specified upon publication]

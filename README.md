# Solar GHI Forecasting with Disentangled Uncertainty Quantification

Short-term Global Horizontal Irradiance (GHI) forecasting at horizons of 1, 3, and 6 hours,
driven by GOES-16 satellite imagery and ground-station measurements.
The project evaluates two spatial-temporal deep learning architectures (ResNet+LSTM,
GraphSAGE+LSTM, including a fixed-vs-learned graph-construction ablation for GraphSAGE)
and a flat MLP baseline against SARIMA and persistence, then layers a disentangled
aleatoric/epistemic uncertainty-quantification pipeline (Gaussian NLL variance head +
SGLD posterior sampling) on top of the best backbone, so forecasts carry not just an
interval but a diagnosis of *why* they are uncertain. Split Conformal Prediction is also
implemented in code as an alternative post-hoc UQ method.
Benchmarked on two Colombian stations with contrasting tropical climates: El Paso (semi-arid,
Caribbean lowlands) and Uniandes (equatorial Andean, Bogotá).

---

## Study sites

| Site | Coordinates | Elevation | Climate | Data period |
|------|------------|-----------|---------|-------------|
| **El Paso** (César, Colombia) | 9.737° N, 74.066° W | ~50 m | Semi-arid (BSh) | Mar 2022 – Mar 2023 |
| **Uniandes** (Bogotá, Colombia) | 4.602° N, 73.695° W | ~2 600 m | Andean equatorial (Cfb) | Sep 2023 – Jan 2025 |

---

## Repository structure

```
.
├── configs/
│   └── exp_v0.yaml                      # Reference config (paths, splits, patch size)
│
├── data/                                # Processed data — NOT in git (.gitignore)
│   ├── ground_aligned/                  #   10-min UTC GHI parquets per site
│   ├── datasets/manifest_v1/            #   Supervised manifests (train/val/test) per site & horizon
│   ├── patches_v1/                      #   Pre-extracted GOES-16 patches (.npz) per site
│   └── metadata/                        #   Site pixel coordinates (site_center_pix_256.json)
│
├── data_raw/                            # Raw ground-station CSVs — NOT in git
├── data_processed/                      # Raw GOES-16 NetCDF + processed GOES_v2/ — NOT in git
│
├── docs/
│   ├── article/                         # LaTeX manuscript (main.tex, sections/, figures/)
│   └── avances.tex                      # Progress notes (advisor-facing, Spanish)
│
├── notebooks/
│   ├── 01_ground_eda.ipynb              # Ground data EDA and UTC alignment → data/ground_aligned/
│   ├── 02_ground_splits_and_baselines.ipynb  # Temporal splits and persistence baseline
│   ├── 03_satellite_georeferencing.ipynb     # GOES-16 pixel identification for each site
│   ├── 04_manifest_build.ipynb          # Interactive manifest builder (see also scripts/04_*)
│   ├── 05b_sarima_order_selection.ipynb # ACF/PACF / AIC grid-search for SARIMA order on raw GHI
│   └── Support_build_patch_store.ipynb  # Batch patch extraction from GOES-16 NetCDF files
│
├── results/
│   ├── figures/                         # Publication figures (PDF)
│   └── summary.md                       # Aggregated metrics by arch / site / horizon
│
├── runs/                                # Run summaries (summary.json); checkpoints/.csv NOT in git
│
├── scripts/                             # Numbered entrypoints — run from project root
│   ├── 04_build_manifests.py            # Build train/val/test manifests (CLI version of nb 04)
│   ├── 05_resnet_lstm_baseline.py       # Train ResNet+LSTM with fixed hyperparameters
│   ├── 05_graphsage_lstm_baseline.py    # Train GraphSAGE+LSTM with fixed hyperparameters
│   ├── 05_sarima_baseline.py            # Fit SARIMAX on raw hourly GHI and evaluate
│   ├── 06_resnet_lstm_optuna.py         # Optuna HPO for ResNet+LSTM (v1: 50 trials/4 seeds; v2: 75/2)
│   ├── 06_graphsage_lstm_optuna.py      # Optuna HPO for GraphSAGE+LSTM (v1 fixed graph / v2 weighted k-NN)
│   ├── 06_mlp_optuna.py                 # Optuna HPO for FlatMLP (50 trials, 4 seeds)
│   ├── 07_conformal_explore.py          # Split Conformal Prediction on any completed run
│   ├── 08_sgld.py                       # SGLD posterior sampling (ResNet / GraphSAGE / MLP)
│   ├── 09_results_table.py              # Aggregate runs/ → results/summary.md and LaTeX table
│   ├── 10_paper_figures.py              # Generate publication figures (skill_day, RMSE_day, ts)
│   ├── compare_fusion_vs_satellite.py   # Compare satellite-only vs. fusion (satellite+surface) models
│   └── gpu_check.py                     # Quick torch CUDA sanity check
│
├── src/solar_uq/                        # Core Python package (pip install -e .)
│   ├── data.py                          #   PatchSeqDataset, GraphSeqDataset, TargetNormalizer
│   ├── train.py                         #   train_one_model(), eval_model(), collect_predictions()
│   ├── train_sgld.py                    #   SGLD burn-in + sampling loop, ensemble inference
│   ├── conformal.py                     #   SplitCP calibration and coverage evaluation
│   ├── metrics.py                       #   rmse_day, skill_day, eval_persistence
│   ├── sgld.py                          #   SGLD optimizer (torch.optim.Optimizer subclass)
│   ├── loaders/fusion_dataset.py        #   Joint satellite + ground-surface feature loader
│   └── models/
│       ├── resnet_lstm.py               #     SmallResNet spatial encoder + LSTM temporal decoder
│       ├── graphsage_lstm.py            #     GraphSAGE encoder (pure PyTorch) + LSTM decoder
│       ├── mlp.py                       #     FlatMLP: spatial avg-pool + LayerNorm MLP
│       └── fusion/                      #     Fusion variants (satellite + ground-surface inputs)
│
├── .gitignore
├── launch_experiments.sh                # tmux-based parallel launcher for GPU pipelines
├── run_sequential.sh                    # Sequential runner with skip-if-done logic
├── pyproject.toml                       # Package build config (pip install -e .)
├── requirements.txt                     # Full dependency freeze (CUDA 13 / PyTorch 2.11)
└── README.md
```

---

## Implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| ResNet+LSTM baseline | ✅ complete | 30/30 runs |
| GraphSAGE+LSTM baseline | ✅ complete | 30/30 runs |
| FlatMLP baseline | ✅ complete | architecture + Optuna script |
| SARIMA baseline | 🚧 needs re-run | Rewritten for raw GHI (was clear-sky index). Run `05b_sarima_order_selection.ipynb` first to re-derive SARIMA order |
| Optuna HPO — ResNet v1 (fixed graph n/a) | ✅ complete | 24/24 runs, 50 trials × 4 seeds |
| Optuna HPO — GraphSAGE v1 (fixed unweighted graph) | ✅ complete | 24/24 runs, 50 trials × 4 seeds — used as the GraphSAGE backbone in the paper's main results |
| Optuna HPO — MLP | 🚧 in progress | 16/24 runs, 50 trials × 4 seeds |
| Optuna HPO — ResNet v2 (expanded search space) | ✅ complete | 12/12 runs, 75 trials × 2 seeds {42,1} |
| Optuna HPO — GraphSAGE v2 (learned weighted k-NN graph) | 🚧 in progress | 1/12 runs, 75 trials × 2 seeds {42,1} — graph-construction ablation vs. v1 |
| Split Conformal Prediction | ✅ implemented | `07_conformal_explore.py`; supports ResNet / GraphSAGE / MLP; not yet run at scale |
| SGLD posterior sampling | ✅ implemented | `08_sgld.py` + `src/solar_uq/train_sgld.py`; 1/24 ResNet runs, 0/24 GraphSAGE, 0 MLP |
| Variance head (Gaussian NLL) | 📋 planned | μ + σ² head, not yet implemented — required for the aleatoric/epistemic decomposition |
| Fusion architecture (satellite + ground-surface) | ✅ implemented | `src/solar_uq/models/fusion/`; 0 runs so far |
| Results aggregation | ✅ complete | `09_results_table.py` → `results/summary.md` |
| Publication figures | ✅ complete | `10_paper_figures.py` → `results/figures/` |

**Split:** train 2022–2023 | val 2024-H1 | test 2024-H2→2025 · **Metric:** RMSE_day (GHI ≥ 20 W/m²)
**Seeds:** v1/baselines use 4 seeds {42, 1, 7, 13}; the v2 expanded-search-space ablations
(ResNet, GraphSAGE) use 2 seeds {42, 1} due to higher per-trial cost — see `docs/article/sections/methodology.tex`.

---

## Requirements

| Dependency | Tested version |
|------------|---------------|
| Python | 3.12 |
| PyTorch | 2.11 (CUDA 13) |
| statsmodels | 0.14.6 |
| optuna | 4.7.0 |
| pandas | 3.0 |
| pyarrow | 23.0 |
| numpy | 2.4 |
| scipy | 1.17 |

GraphSAGE is implemented in **pure PyTorch** — no `torch_geometric` dependency.
GPU with ≥ 12 GB VRAM recommended (tested on NVIDIA RTX 5070).

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

## Running a complete experiment

All commands assume the virtual environment is active and the working directory is the project root.

### Step 0 — Prepare data

Run notebooks in order (skip if you have the pre-processed files from the authors):

```bash
# In JupyterLab or VS Code:
notebooks/01_ground_eda.ipynb                  # align and QC ground data → data/ground_aligned/
notebooks/03_satellite_georeferencing.ipynb    # identify site pixels in GOES-16
notebooks/Support_build_patch_store.ipynb      # extract satellite patches → data/patches_v1/
notebooks/04_manifest_build.ipynb              # build supervised manifests → data/datasets/manifest_v1/
```

Or use the CLI version of the manifest builder:
```bash
.venv/bin/python scripts/04_build_manifests.py --site elpaso
.venv/bin/python scripts/04_build_manifests.py --site uniandes
```

Expected data layout after preparation:
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

### Step 1 — DL baselines (fixed hyperparameters)

```bash
bash run_sequential.sh baseline          # ResNet + GraphSAGE, both sites (60 runs)
bash run_sequential.sh baseline elpaso   # El Paso only
```

Single run:
```bash
.venv/bin/python scripts/05_resnet_lstm_baseline.py    --site elpaso --hours_ahead 6 --seed 42
.venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 7
```

### Step 2 — SARIMA baseline

⚠️ Run `notebooks/05b_sarima_order_selection.ipynb` first to derive the SARIMA order on raw GHI,
then:
```bash
.venv/bin/python scripts/05_sarima_baseline.py --site elpaso   --hours_ahead 1 3 6
.venv/bin/python scripts/05_sarima_baseline.py --site uniandes --hours_ahead 1 3 6
```

### Step 3 — Hyperparameter optimisation (Optuna)

```bash
bash run_sequential.sh optuna              # ResNet + GraphSAGE + MLP, both sites (v1, 4 seeds)
bash run_sequential.sh resnet_optuna       # ResNet only (v1, 50 trials × 4 seeds)
bash run_sequential.sh gsage_optuna        # GraphSAGE only (v1, fixed graph, 50 trials × 4 seeds)
bash run_sequential.sh mlp_optuna          # FlatMLP (50 trials × 4 seeds)
bash run_sequential.sh resnet_optuna_v2    # ResNet v2 (expanded search space, 75 trials × 2 seeds)
bash run_sequential.sh gsage_optuna_v2     # GraphSAGE v2 (learned weighted k-NN graph, 75 trials × 2 seeds)
```

For long parallel runs (4 GPU windows via tmux):
```bash
bash launch_experiments.sh          # creates tmux session 'solar_runs'
bash launch_experiments.sh --status # check progress without attaching
tmux attach -t solar_runs           # watch live logs
```

### Step 4 — Uncertainty quantification

**Split Conformal Prediction** (post-hoc, any completed run):
```bash
.venv/bin/python scripts/07_conformal_explore.py \
    --run_dir runs/graphsage_lstm_optuna/elpaso_H36_L24_P16_seed42_... \
    --alphas 0.05 0.10 0.20
```

**SGLD posterior sampling** (after Optuna v2 runs are complete):
```bash
bash run_sequential.sh sgld          # all 3 architectures, 2 sites (72 runs × 1500 epochs)
bash run_sequential.sh sgld_resnet   # ResNet only
bash run_sequential.sh sgld_gsage    # GraphSAGE only
bash run_sequential.sh sgld_mlp      # MLP only
```

### Step 5 — Aggregate results and figures

```bash
.venv/bin/python scripts/09_results_table.py         # → results/summary.csv, results/summary.md
.venv/bin/python scripts/09_results_table.py --flat  # → results/master.csv (per-run flat table)
.venv/bin/python scripts/10_paper_figures.py         # → results/figures/{fig1,fig2,fig3}.pdf
```

---

## Key results (Optuna v1, test set, 4-seed mean)

| Arch | Site | Horizon | skill_day |
|------|------|---------|-----------|
| GraphSAGE+LSTM (fixed graph) | El Paso | 6 h | **0.636 ± 0.013** |
| ResNet+LSTM | El Paso | 6 h | 0.614 ± 0.023 |
| GraphSAGE+LSTM (fixed graph) | Uniandes | 6 h | 0.417 ± 0.008 |
| ResNet+LSTM | Uniandes | 1 h | 0.161 ± 0.014 |

GraphSAGE+LSTM rows use the **fixed unweighted graph** variant (Section `ssec:gsage` of the
paper); results for the learned, distance-weighted $k$-NN variant are in progress
(`graphsage_lstm_optuna_v2`, see status table above). Full table: `results/summary.md`.

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

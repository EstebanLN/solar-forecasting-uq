#!/usr/bin/env python
"""
09_paper_figures.py — Generate publication-quality figures for the article.

Figures produced:
  fig1_skill_day.pdf/.png  — Grouped bar chart: skill_day by model / horizon / site
  fig2_rmse_day.pdf/.png   — Grouped bar chart: RMSE_day by model / horizon / site
  fig3_timeseries.pdf/.png — Test-set time series: GraphSAGE-LSTM vs observation
                             (elpaso, 6h horizon, seed 42, 5-day window in Jan 2024)

Usage (from project root):
    .venv/bin/python scripts/09_paper_figures.py
    .venv/bin/python scripts/09_paper_figures.py --no-timeseries   # skip slow inference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (colour-blind friendly) ──────────────────────────────────
PALETTE = {
    "Persistence":              "#999999",
    "SARIMA":                   "#E69F00",
    "ResNet+LSTM":              "#56B4E9",
    "GraphSAGE+LSTM":           "#009E73",
    "ResNet+LSTM (Optuna)":     "#0072B2",
    "GraphSAGE+LSTM (Optuna)": "#D55E00",
    "MLP (Optuna)":             "#CC79A7",
}

MODEL_ORDER = [
    "Persistence",
    "SARIMA",
    "ResNet+LSTM",
    "GraphSAGE+LSTM",
    "ResNet+LSTM (Optuna)",
    "GraphSAGE+LSTM (Optuna)",
]

HORIZON_LABELS = {1.0: "1 h", 3.0: "3 h", 6.0: "6 h"}
SITE_LABELS    = {"elpaso": "El Paso (César)", "uniandes": "Uniandes (Bogotá)"}

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_summary() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "summary.csv"
    df = pd.read_csv(path)
    # Filter to Optuna models only (4-seed runs) and keep Persistence/SARIMA
    keep_models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    df = df[df["model"].isin(keep_models)].copy()
    df["model_cat"] = pd.Categorical(df["model"], categories=keep_models, ordered=True)
    df = df.sort_values(["model_cat", "site", "horizon_hours"])
    return df


def _bar_positions(n_horizons: int, n_models: int, group_gap: float = 0.35) -> np.ndarray:
    """Return (n_horizons, n_models) x-positions for grouped bars."""
    bar_w = (1.0 - group_gap) / n_models
    offsets = np.arange(n_models) * bar_w - (n_models - 1) * bar_w / 2
    centers = np.arange(n_horizons, dtype=float)
    return centers[:, None] + offsets[None, :]   # shape (n_horizons, n_models)


# ── Figure 1: Skill_day ──────────────────────────────────────────────────────

def fig_skill_day(df: pd.DataFrame) -> None:
    sites      = ["elpaso", "uniandes"]
    horizons   = [1.0, 3.0, 6.0]
    models     = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    bar_w = (1.0 - 0.30) / len(models)
    positions = _bar_positions(len(horizons), len(models), group_gap=0.30)

    for ax, site in zip(axes, sites):
        sub = df[df["site"] == site]
        for mi, model in enumerate(models):
            msub = sub[sub["model"] == model].set_index("horizon_hours")
            vals = [msub.loc[h, "skill_day_mean"] if h in msub.index else np.nan for h in horizons]
            errs = [msub.loc[h, "skill_day_std"]  if h in msub.index else np.nan for h in horizons]
            xs   = positions[:, mi]
            color = PALETTE.get(model, "#888888")
            ax.bar(xs, vals, width=bar_w * 0.9, color=color, alpha=0.88, zorder=3,
                   label=model if site == sites[0] else None)
            for x, v, e in zip(xs, vals, errs):
                if not np.isnan(v) and not np.isnan(e) and e > 0:
                    ax.errorbar(x, v, yerr=e, fmt="none", color="black",
                                capsize=2, linewidth=0.8, zorder=4)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)
        ax.set_title(SITE_LABELS[site], fontweight="bold")
        ax.set_xticks(np.arange(len(horizons)))
        ax.set_xticklabels([HORIZON_LABELS[h] for h in horizons])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(r"$\mathrm{Skill}_\mathrm{day}$")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    # Legend below both panels
    handles = [Patch(facecolor=PALETTE.get(m, "#888888"), alpha=0.88, label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), frameon=True, edgecolor="gray")

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig1_skill_day.{ext}")
    plt.close(fig)
    print(f"[fig1] saved → {FIGURES_DIR}/fig1_skill_day.{{pdf,png}}")


# ── Figure 2: RMSE_day ───────────────────────────────────────────────────────

def fig_rmse_day(df: pd.DataFrame) -> None:
    sites    = ["elpaso", "uniandes"]
    horizons = [1.0, 3.0, 6.0]
    models   = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    bar_w     = (1.0 - 0.30) / len(models)
    positions = _bar_positions(len(horizons), len(models), group_gap=0.30)

    for ax, site in zip(axes, sites):
        sub = df[df["site"] == site]
        for mi, model in enumerate(models):
            msub = sub[sub["model"] == model].set_index("horizon_hours")
            vals = [msub.loc[h, "rmse_day_mean"] if h in msub.index else np.nan for h in horizons]
            errs = [msub.loc[h, "rmse_day_std"]  if h in msub.index else np.nan for h in horizons]
            xs   = positions[:, mi]
            color = PALETTE.get(model, "#888888")
            ax.bar(xs, vals, width=bar_w * 0.9, color=color, alpha=0.88, zorder=3,
                   label=model if site == sites[0] else None)
            for x, v, e in zip(xs, vals, errs):
                if not np.isnan(v) and not np.isnan(e) and e > 0:
                    ax.errorbar(x, v, yerr=e, fmt="none", color="black",
                                capsize=2, linewidth=0.8, zorder=4)

        # Persistence reference line (RMSE_day of persistence at each horizon)
        pers_sub = sub[sub["model"] == "Persistence"].set_index("horizon_hours")
        if not pers_sub.empty:
            pers_vals = [pers_sub.loc[h, "rmse_day_mean"] if h in pers_sub.index else np.nan
                         for h in horizons]
            for xi, pv in zip(np.arange(len(horizons)), pers_vals):
                if not np.isnan(pv):
                    ax.plot([xi - 0.4, xi + 0.4], [pv, pv],
                            color=PALETTE["Persistence"], linewidth=1.2,
                            linestyle="--", zorder=5)

        ax.set_title(SITE_LABELS[site], fontweight="bold")
        ax.set_xticks(np.arange(len(horizons)))
        ax.set_xticklabels([HORIZON_LABELS[h] for h in horizons])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(r"$\mathrm{RMSE}_\mathrm{day}$ (W/m²)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=PALETTE.get(m, "#888888"), alpha=0.88, label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), frameon=True, edgecolor="gray")

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig2_rmse_day.{ext}")
    plt.close(fig)
    print(f"[fig2] saved → {FIGURES_DIR}/fig2_rmse_day.{{pdf,png}}")


# ── Figure 3: Time-series prediction ────────────────────────────────────────

def fig_timeseries() -> None:
    import torch
    from solar_uq.data import (
        GraphSeqDataset, TargetNormalizer, make_loader,
        read_history_steps_from_manifest,
    )
    from solar_uq.models.graphsage_lstm import GraphSAGE_LSTM, build_edge_index_8n
    from solar_uq.train import collect_predictions

    # Best run: GraphSAGE-LSTM Optuna, elpaso, H=36 (6h), seed=42
    RUN_DIR = PROJECT_ROOT / "runs" / "graphsage_lstm_optuna" / \
              "elpaso_H36_L24_P16_seed42_20260503_091745"
    assert RUN_DIR.exists(), f"Run dir not found: {RUN_DIR}"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fig3] Loading model from {RUN_DIR.name}  device={DEVICE}")

    ckpt    = torch.load(RUN_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    summary = json.loads((RUN_DIR / "summary.json").read_text())

    meta       = ckpt["meta"]
    patch      = int(meta["patch"])
    normalizer = TargetNormalizer(mean=meta["y_mean_train"], std=meta["y_std_train"])

    hp = meta.get("arch_hparams", summary["optuna"]["best_params"])
    edge_index = build_edge_index_8n(patch)
    model = GraphSAGE_LSTM(
        in_dim=16,
        hidden_g=hp.get("hidden_g",      96),
        n_sage_layers=hp.get("n_sage_layers", 4),
        hidden_t=hp.get("hidden_t",      128),
        n_lstm_layers=hp.get("n_lstm_layers", 1),
        dropout_head=hp.get("dropout_head",  0.0),
        input_bn=hp.get("input_bn",      True),
        concat_agg=hp.get("concat_agg",  True),
        edge_index=edge_index,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()
    print(f"[fig3] Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.3f}M params")

    SITE_DIR     = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / "elpaso" / "h6"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / "elpaso" / f"P{patch}"

    test_man = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L        = read_history_steps_from_manifest(test_man)
    test_ds  = GraphSeqDataset(test_man, PATCHES_ROOT, normalizer)
    loader   = make_loader(test_ds, batch_size=64, shuffle=False, num_workers=2,
                           device=DEVICE)

    print("[fig3] Running inference on test set …")
    y_true, y_pred = collect_predictions(model, loader, normalizer, DEVICE)
    timestamps = pd.to_datetime(test_man["t_target"].values)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")

    # 5-day window: Jan 13–17 2024 (UTC, good clear-sky days in elpaso)
    ts_local = timestamps.tz_convert("America/Bogota")
    mask = (ts_local >= "2024-01-13") & (ts_local < "2024-01-18")
    ts_w  = ts_local[mask]
    yt_w  = y_true[mask]
    yp_w  = y_pred[mask]

    # Also load persistence (y at t, compare to y at t+H in manifest)
    ground = pd.read_parquet(
        PROJECT_ROOT / "data" / "ground_aligned" / "ground_10min_utc_elpaso.parquet"
    )
    # Persistence: GHI(t) for t_target — that is GHI(t_label)
    gidx = pd.to_datetime(ground.index)
    if gidx.tz is None:
        gidx = gidx.tz_localize("UTC")
    ground.index = gidx.tz_convert("America/Bogota")
    test_labels  = pd.to_datetime(test_man["t_label"].values)
    if test_labels.tz is None:
        test_labels = test_labels.tz_localize("UTC")
    test_labels  = test_labels.tz_convert("America/Bogota")
    pers_vals    = ground["ghi"].reindex(test_labels).values
    ypers_w      = pers_vals[mask]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    ax.plot(ts_w, yt_w,   color="#333333", linewidth=1.0, label="Observed GHI", zorder=4)
    ax.plot(ts_w, ypers_w, color=PALETTE["Persistence"], linewidth=0.9,
            linestyle="--", alpha=0.75, label="Persistence (6 h)", zorder=3)
    ax.plot(ts_w, yp_w,   color=PALETTE["GraphSAGE+LSTM (Optuna)"], linewidth=1.2,
            label="GraphSAGE-LSTM (6 h)", zorder=5)

    ax.set_xlabel("Date (America/Bogota)")
    ax.set_ylabel("GHI (W/m²)")
    ax.set_title("El Paso (César) — 6-hour forecast, test set sample (Jan 2024)",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig3_timeseries.{ext}")
    plt.close(fig)
    print(f"[fig3] saved → {FIGURES_DIR}/fig3_timeseries.{{pdf,png}}")


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures")
    p.add_argument("--no-timeseries", action="store_true",
                   help="Skip Fig 3 (model inference — slow without GPU)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    df     = _load_summary()

    print("Generating Fig 1: Skill_day …")
    fig_skill_day(df)

    print("Generating Fig 2: RMSE_day …")
    fig_rmse_day(df)

    if not args.no_timeseries:
        print("Generating Fig 3: Time series (loads model + runs inference) …")
        fig_timeseries()
    else:
        print("[fig3] Skipped (--no-timeseries)")

    print("\nDone. Figures saved to:", FIGURES_DIR)

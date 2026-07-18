#!/usr/bin/env python
"""
11_discussion_figures.py — Supplementary figures backing Discussion-section
claims with simple analyses of the same dataset (no new model training).

Figures produced (results/figures/):
  fig4_diurnal_profile.pdf/.png — Mean +/- std GHI by local hour of day, both
                                  sites. Backs the Discussion claim that the
                                  Uniandes clear-sky/cloud cycle is highly
                                  regular relative to El Paso.
  fig5_acf_ghi.pdf/.png         — Autocorrelation of the raw 10-min GHI series
                                  up to lag 6h, both sites, with 1h/3h/6h
                                  horizons marked. Backs the claim that much of
                                  the predictable structure at Uniandes is
                                  already encoded in the recent history itself.
  fig6_delta_hist.pdf/.png      — Histogram of 1-hour GHI deltas during
                                  daytime, both sites. Backs the claim that
                                  El Paso's 1h persistence error is low because
                                  irradiance rarely changes much in an hour.
  fig7_patch_snapshot.pdf/.png  — Example 16x16 satellite patch (single ABI
                                  channel) at both sites, same relative scale.
                                  Illustrates the field-of-view-vs-cloud-scale
                                  argument used to explain reduced skill at
                                  longer horizons at Uniandes.

Usage (from project root):
    .venv/bin/python scripts/11_discussion_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GROUND_FILES = {
    "elpaso": PROJECT_ROOT / "data" / "ground_aligned" / "ground_10min_utc_elpaso.parquet",
    "uniandes": PROJECT_ROOT / "data" / "ground_aligned" / "ground_10min_utc_uniandes.parquet",
}
SITE_LABELS = {"elpaso": "El Paso (César)", "uniandes": "Uniandes (Bogotá)"}
SITE_COLORS = {"elpaso": "#0072B2", "uniandes": "#D55E00"}
TZ = "America/Bogota"  # both sites share UTC-5 year-round (no DST in Colombia)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_ground(site: str) -> pd.DataFrame:
    df = pd.read_parquet(GROUND_FILES[site])
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert(TZ)
    return df.dropna(subset=["ghi"])


# ── Figure 4: diurnal profile ────────────────────────────────────────────────

def fig_diurnal_profile() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for site in ("elpaso", "uniandes"):
        df = _load_ground(site)
        hourly = df.groupby(df.index.hour)["ghi"]
        mean, std = hourly.mean(), hourly.std()
        hours = mean.index.values
        ax.plot(hours, mean.values, color=SITE_COLORS[site], linewidth=1.4,
                label=SITE_LABELS[site])
        ax.fill_between(hours, mean.values - std.values, mean.values + std.values,
                         color=SITE_COLORS[site], alpha=0.18, linewidth=0)
    ax.set_xlabel("Local hour of day (America/Bogotá)")
    ax.set_ylabel("GHI (W/m²)")
    ax.set_title("Diurnal GHI profile: mean ± 1 std", fontweight="bold")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig4_diurnal_profile.{ext}")
    plt.close(fig)
    print(f"[fig4] saved -> {FIGURES_DIR}/fig4_diurnal_profile.{{pdf,png}}")


# ── Figure 5: autocorrelation ────────────────────────────────────────────────

def fig_acf() -> None:
    max_lag_steps = 36  # 6h at 10-min cadence
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for site in ("elpaso", "uniandes"):
        df = _load_ground(site)
        x = df["ghi"].to_numpy()
        x = x - x.mean()
        denom = np.dot(x, x)
        acf = np.array([np.dot(x[:len(x) - lag], x[lag:]) / denom if lag > 0 else 1.0
                         for lag in range(max_lag_steps + 1)])
        lags_hours = np.arange(max_lag_steps + 1) / 6.0
        ax.plot(lags_hours, acf, color=SITE_COLORS[site], linewidth=1.4,
                label=SITE_LABELS[site])
    for h, ls in zip((1, 3, 6), (":", "--", "-.")):
        ax.axvline(h, color="black", linewidth=0.7, linestyle=ls, alpha=0.5)
        ax.text(h, 1.02, f"{h}h", ha="center", va="bottom", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of raw GHI series", fontweight="bold")
    ax.set_xlim(0, 6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig5_acf_ghi.{ext}")
    plt.close(fig)
    print(f"[fig5] saved -> {FIGURES_DIR}/fig5_acf_ghi.{{pdf,png}}")


# ── Figure 6: 1-hour delta histogram ─────────────────────────────────────────

def fig_delta_hist() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for site in ("elpaso", "uniandes"):
        df = _load_ground(site)
        day = df[df["ghi"] >= 20.0]
        delta_1h = day["ghi"].diff(6).dropna()  # 6 steps of 10 min = 1h
        ax.hist(delta_1h, bins=80, range=(-600, 600), histtype="step",
                density=True, color=SITE_COLORS[site], linewidth=1.4,
                label=SITE_LABELS[site])
    ax.set_xlabel("1-hour daytime GHI change,  GHI(t) - GHI(t-1h)  (W/m²)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of 1-hour GHI changes (daytime only)",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig6_delta_hist.{ext}")
    plt.close(fig)
    print(f"[fig6] saved -> {FIGURES_DIR}/fig6_delta_hist.{{pdf,png}}")


# ── Figure 7: example satellite patch snapshot ───────────────────────────────

def fig_patch_snapshot(channel: int = 1) -> None:
    """One illustrative (C,P,P) frame per site, same ABI channel index,
    to visualise the 16x16-pixel field of view relative to cloud scale.
    Channel index is arbitrary (for spatial-extent illustration only, not a
    quantitative comparison of reflectance/brightness-temperature values)."""
    examples = {
        "elpaso": PROJECT_ROOT / "data/patches_v1/elpaso/P16/2024/02/20240202_03_patch.npz",
        "uniandes": PROJECT_ROOT / "data/patches_v1/uniandes/P16/2025/02/20250227_07_patch.npz",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.2))
    for ax, site in zip(axes, ("elpaso", "uniandes")):
        path = examples[site]
        if not path.exists():
            ax.set_visible(False)
            continue
        data = np.load(path)["patch"]  # (6, 16, P, P)
        frame = data[0, channel].astype(np.float32)  # first 10-min slot
        im = ax.imshow(frame, cmap="viridis")
        ax.set_title(SITE_LABELS[site], fontweight="bold")
        ax.set_xlabel("pixel (E-W)")
        ax.set_ylabel("pixel (N-S)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Example 16$\\times$16 satellite patch (ABI channel index {channel})",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig7_patch_snapshot.{ext}")
    plt.close(fig)
    print(f"[fig7] saved -> {FIGURES_DIR}/fig7_patch_snapshot.{{pdf,png}}")


if __name__ == "__main__":
    print("Generating Fig 4: diurnal GHI profile ...")
    fig_diurnal_profile()
    print("Generating Fig 5: autocorrelation of GHI ...")
    fig_acf()
    print("Generating Fig 6: 1-hour GHI delta histogram ...")
    fig_delta_hist()
    print("Generating Fig 7: example satellite patch snapshot ...")
    fig_patch_snapshot()
    print("\nDone. Figures saved to:", FIGURES_DIR)

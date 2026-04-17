#!/usr/bin/env python
"""
04_build_manifests.py — Build supervised manifests for solar forecasting.

Validates satellite history ONCE per split, then generates a manifest for each
requested forecast horizon without repeating the expensive coverage check.

Output layout:
  data/datasets/manifest_v1/{site}/h{hours}/manifest_{split}.parquet
  data/datasets/manifest_v1/{site}/h{hours}/dataset_meta.json

Usage:
    python scripts/04_build_manifests.py
    python scripts/04_build_manifests.py --sites uniandes --horizons 1 3 6
    python scripts/04_build_manifests.py --sites uniandes elpaso --horizons 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
MCMIPF_ROOT  = PROJECT_ROOT / "data_processed" / "GOES_v2" / "MCMIPF"
META_PATH    = PROJECT_ROOT / "data" / "metadata" / "site_center_pix_256.json"
OUT_ROOT     = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"

FREQ_MIN = 10
HIST_HOURS = 4
L = int((HIST_HOURS * 60) / FREQ_MIN)   # 24 steps

SPLITS = {
    "uniandes": {
        "train": ("2023-09-01", "2024-09-30"),
        "val":   ("2024-10-01", "2024-12-31"),
        "test":  ("2025-01-01", "2025-03-28"),
    },
    "elpaso": {
        "train": ("2022-03-01", "2023-06-30"),
        "val":   ("2023-07-01", "2023-10-31"),
        "test":  ("2023-11-01", "2024-03-07"),
    },
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-horizon manifests")
    p.add_argument("--sites",    nargs="+", default=["uniandes", "elpaso"],
                   choices=["uniandes", "elpaso"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6],
                   help="Forecast horizons in hours (e.g. 1 3 6)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Satellite coverage index
# ---------------------------------------------------------------------------

_RX = re.compile(r"(?P<ymd>\d{8})_(?P<h>\d{2})_MCMIPF\.npz$")

def build_covered_set(mcmipf_root: Path) -> set[pd.Timestamp]:
    """All timestamps (UTC) covered by an MCMIPF file."""
    covered: set[pd.Timestamp] = set()
    for p in mcmipf_root.rglob("*_MCMIPF.npz"):
        m = _RX.search(p.name)
        if not m:
            continue
        ymd, hh = m.group("ymd"), int(m.group("h"))
        year, month, day = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
        for slot in range(6):
            covered.add(
                pd.Timestamp(year=year, month=month, day=day,
                             hour=hh, minute=slot * 10, tz="UTC")
            )
    return covered


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC")
    return df.loc[(df.index >= s) & (df.index < e)].copy()


# ---------------------------------------------------------------------------
# Core: compute valid label timestamps (done ONCE per split)
# ---------------------------------------------------------------------------

def find_valid_labels(df_split: pd.DataFrame, covered: set[pd.Timestamp]) -> list[pd.Timestamp]:
    """
    Returns timestamps t where ALL L history steps are:
      1. present in df_split.index
      2. covered by a satellite file

    The inner loop breaks early on the first missing step, so typical runtime
    is much less than O(N * L).
    """
    idx_set = set(df_split.index)
    step = pd.Timedelta(minutes=FREQ_MIN)
    valid = []

    for t in df_split.index:
        ok = True
        for k in range(L):
            th = t - k * step
            if th not in idx_set or th not in covered:
                ok = False
                break
        if ok:
            valid.append(t)

    return valid


# ---------------------------------------------------------------------------
# Build manifest for one (split, horizon) given pre-validated labels
# ---------------------------------------------------------------------------

def build_manifest(
    valid_labels: list[pd.Timestamp],
    df_split: pd.DataFrame,
    H: int,
    site: str,
) -> pd.DataFrame:
    """
    For each valid label t, look up y at t + H steps.
    Skips if t_target is outside the split or GHI is NaN.
    """
    idx_set   = set(df_split.index)
    step      = pd.Timedelta(minutes=FREQ_MIN)
    delta_H   = pd.Timedelta(minutes=FREQ_MIN * H)

    rows = []
    for t in valid_labels:
        t_target = t + delta_H
        if t_target not in idx_set:
            continue
        y = df_split.loc[t_target, "ghi"]
        if pd.isna(y):
            continue

        t_hist_start = t - (L - 1) * step
        hist = pd.date_range(t_hist_start, t, freq=f"{FREQ_MIN}min", tz="UTC")

        rows.append({
            "site":       site,
            "t_label":    t,
            "t_target":   t_target,
            "y":          float(y),
            "history_ts": [ts.isoformat() for ts in hist],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Sites:    {args.sites}")
    print(f"Horizons: {args.horizons}h")
    print(f"L={L} steps ({HIST_HOURS}h history)\n")

    # Site metadata (grid, center pixels)
    with open(META_PATH, encoding="utf-8") as f:
        meta_json = json.load(f)
    GRID_SIZE = int(meta_json["grid_out"])
    SITE_CENTER_PIX = {
        k: (int(v["row"]), int(v["col"]))
        for k, v in meta_json["sites"].items()
    }

    # Satellite coverage set (built once, shared across all sites/horizons)
    print("Indexing MCMIPF files ...", end=" ", flush=True)
    t0 = time.time()
    covered = build_covered_set(MCMIPF_ROOT)
    print(f"{len(covered):,} timestamps indexed in {time.time()-t0:.1f}s\n")

    for site in args.sites:
        ground_path = GROUND_DIR / f"ground_10min_utc_{site}.parquet"
        ground = pd.read_parquet(ground_path)
        assert "ghi" in ground.columns and str(ground.index.tz) == "UTC"

        site_center = SITE_CENTER_PIX[site]
        spec = SPLITS[site]

        for split_name, (start, end) in spec.items():
            df_split = slice_by_date(ground, start, end)
            print(f"[{site}] {split_name}: {len(df_split):,} ground rows", end=" -> ", flush=True)

            # --- validate history once ---
            t_valid = time.time()
            valid_labels = find_valid_labels(df_split, covered)
            print(f"{len(valid_labels):,} valid labels ({time.time()-t_valid:.1f}s)", flush=True)

            # --- one manifest per horizon ---
            for h_hours in args.horizons:
                H = int((h_hours * 60) / FREQ_MIN)

                man = build_manifest(valid_labels, df_split, H, site)

                out_dir = OUT_ROOT / site / f"h{h_hours}"
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"manifest_{split_name}.parquet"
                man.to_parquet(out_path, index=False)

                # dataset_meta.json (written once per horizon dir, idempotent)
                meta_out = {
                    "site":             site,
                    "grid_size":        GRID_SIZE,
                    "site_center_pix":  {"row": site_center[0], "col": site_center[1]},
                    "freq_min":         FREQ_MIN,
                    "horizon_steps":    H,
                    "horizon_hours":    h_hours,
                    "history_steps":    L,
                    "history_hours":    HIST_HOURS,
                    "mcmipf_root":      str(MCMIPF_ROOT),
                    "notes": (
                        "Multi-horizon manifest. "
                        "Satellite patches extracted on-the-fly by the training script."
                    ),
                }
                with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, indent=2)

                print(f"  h{h_hours:>2}h  H={H:>3}  rows={len(man):>6,}  -> {out_path}")

            print()


if __name__ == "__main__":
    main()

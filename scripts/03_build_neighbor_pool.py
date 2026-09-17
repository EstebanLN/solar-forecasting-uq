"""Build a per-hour pool of neighbouring 16x16 satellite patches for the
conv+graph hybrid architecture (research-seminar addition, 2026-09).

From the 256x256 MCMIPF source frame, cut all non-overlapping 16x16 tiles on a
7x7 neighbourhood grid around the site (spacing = patch size, so tiles border
but never overlap), excluding the central tile and any tile that falls out of
bounds. The dataloader later selects K of these at random as GraphSAGE nodes.

Output: data/neighbor_pool_v1/<site>/<YYYY>/<MM>/<YYYYMMDD>_<HH>_neighpool.npz
        key "pool", shape (M, 6, 16, 16, 16)  == (tiles, slots, channels, H, W)

Usage (from repo root):
    .venv/bin/python scripts/03_build_neighbor_pool.py --site elpaso
    .venv/bin/python scripts/03_build_neighbor_pool.py --site uniandes --limit 3   # smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META         = PROJECT_ROOT / "data" / "metadata" / "site_center_pix_256.json"
MCMIPF_ROOT  = PROJECT_ROOT / "data_processed" / "GOES_v2" / "MCMIPF"
PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1"
OUT_ROOT     = PROJECT_ROOT / "data" / "neighbor_pool_v1"

GRID  = 256
PATCH = 16
HALF  = PATCH // 2
RING  = 3          # 7x7 grid of tile centres (offsets -3..3 * PATCH), minus centre


def neighbor_centers(center, grid=GRID, patch=PATCH, ring=RING):
    r0, c0 = center
    out = []
    for dr in range(-ring, ring + 1):
        for dc in range(-ring, ring + 1):
            if dr == 0 and dc == 0:
                continue                                  # exclude the central tile
            r, c = r0 + dr * patch, c0 + dc * patch
            if HALF <= r <= grid - HALF and HALF <= c <= grid - HALF:
                out.append((r, c))                        # in-bounds only
    return out


def extract_tile(frame_chw: np.ndarray, center_rc) -> np.ndarray:
    r0, c0 = center_rc
    return frame_chw[:, r0 - HALF:r0 + HALF, c0 - HALF:c0 + HALF]   # (C, PATCH, PATCH)


def hour_src_path(t: pd.Timestamp) -> Path:
    return MCMIPF_ROOT / t.strftime("%Y") / t.strftime("%m") / f"{t.strftime('%Y%m%d')}_{t.strftime('%H')}_MCMIPF.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["elpaso", "uniandes"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N hours")
    args = ap.parse_args()

    meta   = json.load(open(META))
    center = (int(meta["sites"][args.site]["row"]), int(meta["sites"][args.site]["col"]))
    centers = neighbor_centers(center)
    print(f"[{args.site}] center={center} | pool tiles M={len(centers)}")

    src_patches = sorted((PATCHES_ROOT / args.site / "P16").rglob("*_patch.npz"))
    if args.limit:
        src_patches = src_patches[:args.limit]

    n_done = n_skip = n_miss = 0
    for pth in src_patches:
        stem = pth.stem.replace("_patch", "")                 # YYYYMMDD_HH
        ymd, hh = stem.split("_")
        t = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]} {hh}:00", tz="UTC")
        out_dir = OUT_ROOT / args.site / t.strftime("%Y") / t.strftime("%m")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stem}_neighpool.npz"
        if out.exists():
            n_skip += 1
            continue
        src = hour_src_path(t)
        if not src.exists():
            n_miss += 1
            continue
        with np.load(src) as d:
            arr = d["mcmipf"].astype(np.float32)              # (6, 16, 256, 256)
        tiles = [np.stack([extract_tile(arr[s], rc) for s in range(arr.shape[0])], axis=0)
                 for rc in centers]                            # each (6, 16, 16, 16)
        pool = np.stack(tiles, axis=0).astype(np.float16)      # (M, 6, 16, 16, 16)
        np.savez(out, pool=pool)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {args.site}: {n_done} done...")
    print(f"[{args.site}] done={n_done} skip={n_skip} miss={n_miss} | M={len(centers)}")


if __name__ == "__main__":
    main()

"""Convert neighbour-pool .npz files to raw .npy so they can be memory-mapped
by DataLoader workers (num_workers>0). Resume-safe; removes each .npz after
converting to avoid doubling disk use.

Usage (from repo root):
    .venv/bin/python scripts/03b_convert_pool_to_npy.py            # both sites
    .venv/bin/python scripts/03b_convert_pool_to_npy.py elpaso     # one site
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def main():
    sites = sys.argv[1:] or ["elpaso", "uniandes"]
    for site in sites:
        root = ROOT / "data" / "neighbor_pool_v1" / site
        files = sorted(root.rglob("*_neighpool.npz"))
        conv = skip = 0
        for f in files:
            npy = f.with_suffix(".npy")            # ..._neighpool.npy
            if npy.exists():
                skip += 1
                continue
            with np.load(f) as d:
                arr = d["pool"]
            np.save(npy, arr)                       # raw .npy (memory-mappable)
            f.unlink()                              # reclaim the .npz space
            conv += 1
            if conv % 1000 == 0:
                print(f"[{site}] {conv} converted...", flush=True)
        print(f"[{site}] converted={conv} skipped={skip} total={len(files)+skip}", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
08b_deep_ensemble.py — Deep-ensemble epistemic uncertainty from the
independently-trained seed checkpoints (the recommended alternative to SGLD;
see docs/sgld_metodologia).

For a given (arch, site, hours_ahead, optuna_version) it loads EVERY seed's
Optuna-trained checkpoint, predicts on the test set, and treats the M seed
predictions as the ensemble: the mean is the point forecast and the spread is
the epistemic uncertainty. Each seed is loaded with ITS OWN best_params (the
Optuna search is per-seed), so the members are genuinely independent optima.

It reuses the exact ensemble/band/coverage/CRPS metrics of the SGLD pipeline
(solar_uq.train_sgld._ensemble_metrics), so the two UQ methods are directly
comparable, and writes a summary.json in the same shape as scripts/08_sgld.py.
No new training — inference only.

Usage (from project root):
    .venv/bin/python scripts/08b_deep_ensemble.py --arch resnet    --site uniandes --hours_ahead 6 --optuna_version v1
    .venv/bin/python scripts/08b_deep_ensemble.py --arch graphsage --site elpaso   --hours_ahead 6 --optuna_version v2 --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.data import (
    GraphSeqDataset,
    PatchSeqDataset,
    TargetNormalizer,
    make_loader,
    preload_patch_cache,
    read_history_steps_from_manifest,
)
from solar_uq.metrics import eval_persistence, skill_score
from solar_uq.train_sgld import _ensemble_metrics

# Reuse build_model + ARCH_CFG from the (digit-named) SGLD driver, so the model
# is instantiated exactly as it was for training/SGLD.
_spec = importlib.util.spec_from_file_location("_sgld_driver", PROJECT_ROOT / "scripts" / "08_sgld.py")
_sgld = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sgld)
build_model = _sgld.build_model
ARCH_CFG = _sgld.ARCH_CFG


def find_seed_runs(runs_root: Path, site: str, horizon_hours: float) -> dict[int, tuple[dict, str]]:
    """Return {seed: (best_params, ckpt_path)} for every run matching
    (site, horizon_hours) that has a usable checkpoint on disk."""
    out: dict[int, tuple[dict, str]] = {}
    if not runs_root.exists():
        return out
    for run_dir in sorted(runs_root.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        meta = json.loads(sj.read_text(encoding="utf-8"))
        if meta.get("site") != site:
            continue
        if meta.get("temporal", {}).get("horizon_hours") != horizon_hours:
            continue
        bp = meta.get("optuna", {}).get("best_params")
        ckpt = meta.get("best_model", {}).get("best_ckpt_path")
        seed = meta.get("seed")
        if bp is None or ckpt is None or not Path(ckpt).exists():
            continue
        out[int(seed)] = (bp, ckpt)   # latest run per seed wins (sorted order)
    return out


@torch.no_grad()
def predict_test(model, loader, normalizer, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, yhats = [], []
    for x_seq, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        yhats.append(model(x_seq).detach().cpu().numpy())
        ys.append(y.numpy())
    return (normalizer.denormalize(np.concatenate(yhats)),
            normalizer.denormalize(np.concatenate(ys)))


def main() -> None:
    p = argparse.ArgumentParser(description="Deep-ensemble epistemic UQ from seed checkpoints")
    p.add_argument("--arch", required=True, choices=list(ARCH_CFG))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--optuna_version", default="v1", choices=["v1", "v2"])
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--runs_root", default=None, help="Override output dir (default runs/{arch}_deep_ensemble)")
    args = p.parse_args()

    cfg = ARCH_CFG[args.arch]
    FREQ_MIN = 10
    horizon_hours = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    DEVICE = args.device

    print(f"\nDEEP ENSEMBLE | arch={args.arch} | optuna={args.optuna_version} | "
          f"site={args.site} | hours={args.hours_ahead} | device={DEVICE}")

    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    OPTUNA_ROOT  = PROJECT_ROOT / "runs" / cfg[args.optuna_version]
    OUT_ROOT     = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / f"{args.arch}_deep_ensemble"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"
    preload_patch_cache(PATCHES_ROOT)

    seed_runs = find_seed_runs(OPTUNA_ROOT, args.site, horizon_hours)
    if len(seed_runs) < 2:
        raise SystemExit(f"Need >=2 seed checkpoints for a deep ensemble; found {len(seed_runs)} "
                         f"in {OPTUNA_ROOT} for site={args.site}, horizon={horizon_hours}h.")
    seeds = sorted(seed_runs)
    print(f"  ensemble members: {len(seeds)} seeds {seeds}")

    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    H, GRID = int(meta["horizon_steps"]), int(meta["grid_size"])

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())

    ground = pd.read_parquet(GROUND_DIR / f"ground_10min_utc_{args.site}.parquet")
    baseline_test = eval_persistence(test_man, ground, args.day_threshold)
    print(f"  Persistence test RMSE_day={baseline_test['rmse_day']:.1f}")

    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    test_ds = DS(test_man, PATCHES_ROOT, normalizer)
    test_loader = make_loader(test_ds, 64, shuffle=False, num_workers=0, seed=0, device=DEVICE)

    member_preds: list[np.ndarray] = []
    y_true: np.ndarray | None = None
    for seed in seeds:
        bp, ckpt = seed_runs[seed]
        model = build_model(args.arch, bp, args.patch, L, DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        yhat, yt = predict_test(model, test_loader, normalizer, DEVICE)
        if y_true is None:
            y_true = yt
        member_preds.append(yhat)
        print(f"  seed {seed}: loaded {Path(ckpt).parent.name}")

    ensemble_preds = np.stack(member_preds, axis=0)                 # (M, N)
    et = _ensemble_metrics(y_true, ensemble_preds, args.day_threshold)
    et["skill_vs_persistence"]     = skill_score(et["rmse"],     baseline_test["rmse"])
    et["skill_day_vs_persistence"] = skill_score(et["rmse_day"], baseline_test["rmse_day"])

    print(f"\n=== Deep-ensemble test ({len(seeds)} members) ===")
    print(f"  RMSE_day={et['rmse_day']:.2f}  skill_day={et['skill_day_vs_persistence']:.3f}")
    print(f"  sigma_epi_day={et.get('ensemble_std_day_mean', float('nan')):.2f}  "
          f"cov_day90(gauss)={et.get('coverage_day_90', float('nan')):.2f}  "
          f"cov_day95(pct)={et.get('coverage_day_pct_95', float('nan')):.2f}")

    run_name = f"{args.site}_H{H}_L{L}_P{args.patch}_{args.arch}_{args.optuna_version}_M{len(seeds)}"
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name, "method": "deep_ensemble", "arch": args.arch,
        "site": args.site, "optuna_version": args.optuna_version,
        "device": DEVICE, "day_threshold": args.day_threshold,
        "temporal": {"history_steps_L": L, "horizon_steps_H": H,
                     "horizon_hours": horizon_hours, "freq_min": FREQ_MIN},
        "spatial": {"grid_size": GRID, "patch": args.patch, "channels": 16},
        "ensemble": {"n_members": len(seeds), "seeds": seeds,
                     "member_ckpts": [seed_runs[s][1] for s in seeds]},
        "baselines": {"persistence_test": baseline_test},
        "best_model": {"inference_method": "deep_ensemble_mean", "final_test": et},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {run_dir}")


if __name__ == "__main__":
    main()

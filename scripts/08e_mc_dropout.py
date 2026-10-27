#!/usr/bin/env python
"""
08e_mc_dropout.py — MC Dropout epistemic UQ from a single trained checkpoint.

Advisor's request (2026-10-19): try MC Dropout as a faster route to a usable
uncertainty band than SGLD, which does not converge on these nets.

Method: load one Optuna-trained checkpoint, put the model in eval() so
normalisation layers stay deterministic, then re-enable ONLY the Dropout
modules and run T stochastic forward passes over the test set. The spread
across passes is the epistemic estimate.

IMPORTANT CAVEAT (report it): in these architectures dropout sits only in the
MLP head (FlatMLP is the exception), so this is effectively *last-layer* MC
Dropout — it samples uncertainty over the final regressor, not over the
spatio-temporal encoder. It therefore gives a lower bound on epistemic
uncertainty.

A checkpoint whose tuned dropout is 0.0 yields zero variance; use --list to see
which (site, horizon, seed) cells were tuned with dropout > 0.

Metrics come from solar_uq.train_sgld._ensemble_metrics, the same function used
for the SGLD ensemble and the deep ensembles, so the three UQ methods are
directly comparable.

Usage (from repo root):
    .venv/bin/python scripts/08e_mc_dropout.py --arch resnet --list
    .venv/bin/python scripts/08e_mc_dropout.py --arch resnet --site uniandes \
        --hours_ahead 6 --seed 42 --optuna_version v2 --n_passes 50
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.data import (
    GraphSeqDataset, PatchSeqDataset, TargetNormalizer,
    make_loader, preload_patch_cache, read_history_steps_from_manifest,
)
from solar_uq.metrics import eval_persistence, skill_score
from solar_uq.train_sgld import _ensemble_metrics
from solar_uq.run_select import best_members

_spec = importlib.util.spec_from_file_location("_sgld_driver", PROJECT_ROOT / "scripts" / "08_sgld.py")
_sgld = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sgld)
build_model = _sgld.build_model
ARCH_CFG = _sgld.ARCH_CFG


def dropout_of(best_params: dict) -> float:
    """Largest tuned dropout rate in best_params (key name varies per arch:
    'dropout' for resnet/convgraph/mlp, 'dropout_head' for the graph models)."""
    vals = [v for k, v in best_params.items() if "dropout" in k.lower()
            and isinstance(v, (int, float))]
    return float(max(vals)) if vals else 0.0


def scan_runs(runs_root: Path):
    """Yield (site, horizon_hours, seed, dropout, ckpt_exists, best_params, ckpt)."""
    if not runs_root.exists():
        return
    for run_dir in sorted(runs_root.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        bp = meta.get("optuna", {}).get("best_params")
        if bp is None:
            continue
        ckpt = (meta.get("best_model") or {}).get("best_ckpt_path")
        yield (meta.get("site"), meta.get("temporal", {}).get("horizon_hours"),
               meta.get("seed"), dropout_of(bp),
               bool(ckpt) and Path(ckpt).exists(), bp, ckpt)


def enable_mc_dropout(model) -> int:
    """eval() the whole model (so LayerNorm/BatchNorm are deterministic), then
    re-enable only the Dropout modules. Returns how many were re-enabled."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_predict(model, loader, normalizer, device, n_passes: int, seed: int):
    """T stochastic forward passes over a deterministic (shuffle=False) loader."""
    torch.manual_seed(seed)
    preds, y_true = [], None
    for t in range(n_passes):
        yhats, ys = [], []
        for x_seq, y in loader:
            yhats.append(model(x_seq.to(device, non_blocking=True)).detach().cpu().numpy())
            if y_true is None:
                ys.append(y.numpy())
        preds.append(normalizer.denormalize(np.concatenate(yhats)))
        if y_true is None:
            y_true = normalizer.denormalize(np.concatenate(ys))
        if (t + 1) % 10 == 0 or t == 0:
            print(f"    pass {t+1}/{n_passes}", flush=True)
    return np.stack(preds, axis=0), y_true          # (T, N), (N,)


def main() -> None:
    p = argparse.ArgumentParser(description="MC Dropout epistemic UQ from one checkpoint")
    p.add_argument("--arch", required=True, choices=list(ARCH_CFG))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--optuna_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--n_passes", type=int, default=50, help="T stochastic forward passes")
    p.add_argument("--mc_seed", type=int, default=0, help="RNG seed for the dropout masks")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--runs_root", default=None)
    p.add_argument("--list", action="store_true",
                   help="list cells with their tuned dropout and exit")
    args = p.parse_args()

    cfg = ARCH_CFG[args.arch]
    OPTUNA_ROOT = PROJECT_ROOT / "runs" / cfg[args.optuna_version]

    if args.list:
        rows = sorted((r for r in scan_runs(OPTUNA_ROOT) if r[0]),
                      key=lambda r: (r[0], r[1] or 0, r[2] or 0))
        print(f"\n{args.arch} / {args.optuna_version} — tuned dropout per cell "
              f"(MC Dropout needs > 0)\n")
        print(f"  {'site':10s} {'h':>3s} {'seed':>5s} {'dropout':>8s}  ckpt")
        for site, h, seed, dr, has_ckpt, _, _ in rows:
            flag = "" if dr > 0 else "   <- zero variance"
            print(f"  {site:10s} {int(h or 0):3d} {seed:5d} {dr:8.2f}  "
                  f"{'yes' if has_ckpt else 'NO '}{flag}")
        usable = [r for r in rows if r[3] > 0 and r[4]]
        print(f"\n  usable for MC Dropout (dropout>0 and checkpoint present): {len(usable)}/{len(rows)}")
        return

    horizon_hours = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    DEVICE = args.device

    match = [r for r in scan_runs(OPTUNA_ROOT)
             if r[0] == args.site and r[1] == horizon_hours and r[2] == args.seed]
    if not match:
        raise SystemExit(f"No run for site={args.site} h={args.hours_ahead} seed={args.seed} "
                         f"in {OPTUNA_ROOT}")
    if len(match) > 1:
        # Canonical rule: lowest validation RMSE, never sort order. elpaso h6 has
        # two runs for seed 42 and "last wins" disagreed with the variance net.
        # Ask run_select rather than rebuilding the rule from scan_runs' tuples:
        # those carry no validation score, so an earlier attempt here silently
        # fell back to sort order while printing that it had compared scores.
        canon = best_members(OPTUNA_ROOT, args.site, horizon_hours).get(args.seed)
        if canon is None:
            raise SystemExit(f"No usable run for seed {args.seed} in {OPTUNA_ROOT}")
        want = Path(canon[1]).resolve()
        match = [r for r in match if Path(r[6]).resolve() == want] or match[-1:]
    _, _, _, dropout, has_ckpt, bp, ckpt = match[-1]
    if not has_ckpt:
        raise SystemExit(f"Checkpoint missing for that cell ({ckpt}).")
    if dropout <= 0:
        raise SystemExit(f"Tuned dropout is {dropout}: MC Dropout would give zero variance. "
                         f"Pick another cell (--list) or retrain with dropout > 0.")

    print(f"\nMC DROPOUT | arch={args.arch} optuna={args.optuna_version} site={args.site} "
          f"h={args.hours_ahead} seed={args.seed} device={DEVICE}")
    print(f"  tuned dropout = {dropout} | T = {args.n_passes} passes")

    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    OUT_ROOT = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / f"{args.arch}_mc_dropout"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists() and PATCHES_ROOT.exists()
    preload_patch_cache(PATCHES_ROOT)

    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        dmeta = json.load(f)
    H = int(dmeta["horizon_steps"])
    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())

    ground = pd.read_parquet(GROUND_DIR / f"ground_10min_utc_{args.site}.parquet")
    baseline_test = eval_persistence(test_man, ground, args.day_threshold)
    print(f"  persistence test RMSE_day={baseline_test['rmse_day']:.1f}")

    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    test_ds = DS(test_man, PATCHES_ROOT, normalizer)
    test_loader = make_loader(test_ds, 64, shuffle=False, num_workers=0, seed=0, device=DEVICE)

    model = build_model(args.arch, bp, args.patch, L, DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    n_do = enable_mc_dropout(model)
    print(f"  dropout modules re-enabled: {n_do}")
    if n_do == 0:
        raise SystemExit("Model has no Dropout modules; MC Dropout is not applicable.")

    preds, y_true = mc_predict(model, test_loader, normalizer, DEVICE,
                               args.n_passes, args.mc_seed)
    et = _ensemble_metrics(y_true, preds, args.day_threshold)
    et["skill_vs_persistence"]     = skill_score(et["rmse"],     baseline_test["rmse"])
    et["skill_day_vs_persistence"] = skill_score(et["rmse_day"], baseline_test["rmse_day"])

    print(f"\n=== MC Dropout test (T={args.n_passes}) ===")
    print(f"  RMSE_day={et['rmse_day']:.2f}  skill_day={et['skill_day_vs_persistence']:.3f}")
    print(f"  sigma_epi_day={et.get('ensemble_std_day_mean', float('nan')):.2f}")
    print(f"  coverage_day 80/90/95 (gauss) = "
          f"{et.get('coverage_day_80', float('nan')):.3f} / "
          f"{et.get('coverage_day_90', float('nan')):.3f} / "
          f"{et.get('coverage_day_95', float('nan')):.3f}")

    run_name = (f"{args.site}_H{H}_L{L}_P{args.patch}_{args.arch}_{args.optuna_version}"
                f"_seed{args.seed}_T{args.n_passes}")
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": f"{args.arch}_mc_dropout", "method": "mc_dropout",
        "arch": args.arch, "site": args.site, "seed": args.seed,
        "optuna_version": args.optuna_version, "patch": args.patch,
        "temporal": {"horizon_hours": horizon_hours, "history_steps": L, "horizon_steps_H": H},
        "mc_dropout": {"n_passes": args.n_passes, "tuned_dropout": dropout,
                       "dropout_modules": n_do, "mc_seed": args.mc_seed,
                       "scope": "head only (last-layer MC Dropout)" if args.arch != "mlp" else "all layers",
                       "source_ckpt": ckpt},
        "baselines": {"persistence_test": baseline_test},
        "final_test": et,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {run_dir}")


if __name__ == "__main__":
    main()

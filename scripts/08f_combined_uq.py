#!/usr/bin/env python
"""
08f_combined_uq.py — combine epistemic and aleatoric uncertainty into one band.

Every epistemic-only method tried here is badly under-covered on this data
(90% nominal: SGLD gives no band at all, deep ensembles 0.10-0.47, MC Dropout
0.08). The reason is that the dominant term is ALEATORIC: the variance network
returns sigma_a of order 300 W/m^2 against a sigma_epi of 11-96. No amount of
sampling over weights can recover the missing term.

This script forms the predictive band that uses both:

    mu(x)        = mean over the deep-ensemble members
    sigma_epi(x) = std  over the deep-ensemble members
    sigma_a(x)   = variance network (Gamma mean, sqrt)
    sigma_tot(x) = sqrt(sigma_a^2 + sigma_epi^2)

and reports coverage for each of the three bands so the decomposition is
explicit.

IMPORTANT — unit convention: variance networks trained BEFORE 2026-10-21 fitted
the squared residual in physical units; from that date they fit it in
normalised units and sigma_a must be rescaled by the normaliser's std. This
script assumes the NEW convention and therefore only accepts --variance_version
v2 runs. Mixing an old run in would silently scale sigma_a wrong.

Usage:
    .venv/bin/python scripts/08f_combined_uq.py --arch resnet --site uniandes \
        --hours_ahead 6 --optuna_version v2
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
from solar_uq.metrics import eval_persistence, metrics_from_arrays, skill_score
from solar_uq.variance_net import gamma_params, aleatoric_variance, make_variance_model

_spec = importlib.util.spec_from_file_location("_sgld_driver", PROJECT_ROOT / "scripts" / "08_sgld.py")
_sgld = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sgld)
build_model, ARCH_CFG = _sgld.build_model, _sgld.ARCH_CFG

Z = {"80": 1.281552, "90": 1.644854, "95": 1.959964}


def find_mean_members(runs_root: Path, site: str, horizon: float):
    """{seed: (best_params, ckpt)} for runs with a checkpoint on disk."""
    out = {}
    for d in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        sj = d / "summary.json"
        if not sj.exists():
            continue
        m = json.loads(sj.read_text(encoding="utf-8"))
        if m.get("site") != site or m.get("temporal", {}).get("horizon_hours") != horizon:
            continue
        bp = m.get("optuna", {}).get("best_params")
        ck = (m.get("best_model") or {}).get("best_ckpt_path")
        if bp and ck and Path(ck).exists():
            out[int(m["seed"])] = (bp, ck)
    return out


def find_variance_run(var_root: Path, site: str, horizon: float, seed: int):
    """Latest variance-net run for the cell that carries a checkpoint."""
    best = None
    for d in sorted(var_root.iterdir()) if var_root.exists() else []:
        sj, ck = d / "summary.json", d / "best_variance_model.pt"
        if not (sj.exists() and ck.exists()):
            continue
        m = json.loads(sj.read_text(encoding="utf-8"))
        if m.get("site") != site or m.get("seed") != seed:
            continue
        if m.get("horizon_hours") != horizon:
            continue
        best = (m, str(ck))          # sorted order -> last wins
    return best


@torch.no_grad()
def predict_members(models, loader, normalizer, device):
    """Per-member test predictions (M, N) plus the targets (N,)."""
    preds, y_true = [], None
    for model in models:
        yh, ys = [], []
        for x_seq, y in loader:
            yh.append(model(x_seq.to(device, non_blocking=True)).cpu().numpy())
            if y_true is None:
                ys.append(y.numpy())
        preds.append(normalizer.denormalize(np.concatenate(yh)))
        if y_true is None:
            y_true = normalizer.denormalize(np.concatenate(ys))
    return np.stack(preds), y_true


@torch.no_grad()
def predict_sigma_a(var_model, loader, normalizer, device):
    """Per-sample aleatoric sd in W/m^2 (variance net trained in normalised units)."""
    out = []
    for x_seq, _ in loader:
        alpha, lam = gamma_params(var_model(x_seq.to(device, non_blocking=True)))
        out.append((float(normalizer.std) * torch.sqrt(aleatoric_variance(alpha, lam))).cpu().numpy())
    return np.concatenate(out)


def coverage(y, mu, sigma, day_mask):
    cov = {}
    for lvl, z in Z.items():
        inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
        cov[lvl] = float(inside[day_mask].mean()) if day_mask.any() else float("nan")
    return cov


def main():
    p = argparse.ArgumentParser(description="Combined aleatoric + epistemic predictive band")
    p.add_argument("--arch", required=True, choices=list(ARCH_CFG))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--optuna_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--variance_root", default=None)
    p.add_argument("--runs_root", default=None)
    args = p.parse_args()

    cfg = ARCH_CFG[args.arch]
    horizon = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    DEV = args.device
    OPTUNA_ROOT = PROJECT_ROOT / "runs" / cfg[args.optuna_version]
    VAR_ROOT = Path(args.variance_root) if args.variance_root else PROJECT_ROOT / "runs" / f"{cfg['runs_dir']}_variance" \
        if "runs_dir" in cfg else PROJECT_ROOT / "runs" / f"{args.arch}_lstm_variance"
    OUT_ROOT = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / f"{args.arch}_combined_uq"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\nCOMBINED UQ | {args.arch} {args.site} h{args.hours_ahead} optuna={args.optuna_version} dev={DEV}")

    members = find_mean_members(OPTUNA_ROOT, args.site, horizon)
    if len(members) < 2:
        raise SystemExit(f"Need >=2 mean checkpoints, found {len(members)} in {OPTUNA_ROOT}")
    seeds = sorted(members)
    print(f"  ensemble members: {seeds}")

    SITE_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / args.site / f"h{args.hours_ahead}"
    PATCHES = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    preload_patch_cache(PATCHES)
    meta = json.loads((SITE_DIR / "dataset_meta.json").read_text())
    H = int(meta["horizon_steps"])
    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    test_man = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())
    ground = pd.read_parquet(PROJECT_ROOT / "data" / "ground_aligned" / f"ground_10min_utc_{args.site}.parquet")
    base = eval_persistence(test_man, ground, args.day_threshold)

    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    test_loader = make_loader(DS(test_man, PATCHES, normalizer), 64, shuffle=False,
                              num_workers=0, seed=0, device=DEV)

    models = []
    for s in seeds:
        bp, ck = members[s]
        m = build_model(args.arch, bp, args.patch, L, DEV)
        st = torch.load(ck, map_location=DEV, weights_only=False)
        m.load_state_dict(st.get("model_state", st)); m.eval()
        models.append(m)
    preds, y_true = predict_members(models, test_loader, normalizer, DEV)
    mu, sigma_epi = preds.mean(0), preds.std(0)

    # aleatoric: average sigma_a over the per-seed variance nets available
    sig_a_list, used = [], []
    for s in seeds:
        found = find_variance_run(VAR_ROOT, args.site, horizon, s)
        if not found:
            continue
        vmeta, vck = found
        bp, _ = members[s]
        vm = make_variance_model(build_model(args.arch, bp, args.patch, L, DEV)).to(DEV)  # swapped head starts on CPU
        st = torch.load(vck, map_location=DEV, weights_only=False)
        vm.load_state_dict(st.get("model_state", st)); vm.eval()
        sig_a_list.append(predict_sigma_a(vm, test_loader, normalizer, DEV))
        used.append(s)
    if not sig_a_list:
        raise SystemExit(f"No variance-net checkpoints found under {VAR_ROOT} for this cell. "
                         f"Train them first (scripts/12_variance_net.py).")
    sigma_a = np.mean(sig_a_list, axis=0)
    print(f"  variance nets used (seeds): {used}")

    sigma_tot = np.sqrt(sigma_a ** 2 + sigma_epi ** 2)
    day = y_true >= args.day_threshold
    pt = metrics_from_arrays(y_true, mu, day_threshold=args.day_threshold)

    cov_e = coverage(y_true, mu, sigma_epi, day)
    cov_a = coverage(y_true, mu, sigma_a, day)
    cov_t = coverage(y_true, mu, sigma_tot, day)

    print(f"\n  RMSE_day={pt['rmse_day']:.1f}  skill_day={skill_score(pt['rmse_day'], base['rmse_day']):.3f}")
    print(f"  mean sd (daytime):  sigma_epi={sigma_epi[day].mean():.1f}  "
          f"sigma_a={sigma_a[day].mean():.1f}  sigma_tot={sigma_tot[day].mean():.1f}")
    print(f"\n  {'band':12s} {'cov80':>7s} {'cov90':>7s} {'cov95':>7s}")
    for name, c in (("epistemic", cov_e), ("aleatoric", cov_a), ("COMBINED", cov_t)):
        print(f"  {name:12s} {c['80']:7.3f} {c['90']:7.3f} {c['95']:7.3f}")

    run_dir = OUT_ROOT / f"{args.site}_H{H}_L{L}_P{args.patch}_{args.arch}_{args.optuna_version}_M{len(seeds)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "method": "combined_uq", "arch": args.arch, "site": args.site,
        "optuna_version": args.optuna_version,
        "temporal": {"horizon_hours": horizon, "history_steps": L, "horizon_steps_H": H},
        "ensemble_seeds": seeds, "variance_net_seeds": used,
        "baselines": {"persistence_test": base},
        "final_test": {**pt,
                       "skill_day_vs_persistence": skill_score(pt["rmse_day"], base["rmse_day"]),
                       "sigma_epi_day_mean": float(sigma_epi[day].mean()),
                       "sigma_a_day_mean": float(sigma_a[day].mean()),
                       "sigma_tot_day_mean": float(sigma_tot[day].mean()),
                       "coverage_day_epistemic": cov_e,
                       "coverage_day_aleatoric": cov_a,
                       "coverage_day_combined": cov_t},
    }, indent=2))
    print(f"\nSaved -> {run_dir}")


if __name__ == "__main__":
    main()

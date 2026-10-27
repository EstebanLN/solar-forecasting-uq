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
from solar_uq.run_select import best_members

_spec = importlib.util.spec_from_file_location("_sgld_driver", PROJECT_ROOT / "scripts" / "08_sgld.py")
_sgld = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sgld)
build_model, ARCH_CFG = _sgld.build_model, _sgld.ARCH_CFG

Z = {"80": 1.281552, "90": 1.644854, "95": 1.959964}


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


def sharpness(y, mu, sigma, day_mask, level="90"):
    """Does a band of varying width beat a flat one at the same coverage?

    Coverage alone cannot answer that. A constant width set to the right quantile
    of the global error covers its nominal share exactly and is still useless for
    telling an easy hour from a hard one, so a calibrated band is not yet an
    informative one. The comparison here is the interval (Winkler) score at one
    level, which charges for width and for every miss:

        IS = 2*z*sigma + (2/alpha) * distance outside the interval

    against the same score for the single best constant width, chosen as the
    empirical quantile of |y - mu| that attains the nominal level. A positive
    skill means sigma(x) earns its place; around zero means the model is only
    reproducing the average error.
    """
    z = Z[level]
    alpha = 1.0 - int(level) / 100.0
    y, mu, sigma = y[day_mask], mu[day_mask], sigma[day_mask]
    if y.size == 0:
        return {}

    def iscore(w):
        lo, hi = mu - w, mu + w
        pen = np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0)
        return float(np.mean(2.0 * w + (2.0 / alpha) * pen))

    abs_err = np.abs(y - mu)
    w_const = float(np.quantile(abs_err, int(level) / 100.0))
    # corr(sigma, |error|) has a ceiling well below 1 even for a perfectly
    # specified model, because |error| is a noisy draw around sigma, so the
    # observed value must be read against what is achievable and not against 1.
    # Closed form, for y|x ~ N(mu, sigma(x)) and |e| = sigma*|z| with z ~ N(0,1):
    #   Cov(sigma, sigma|z|) = E|z| * Var(sigma)
    #   Var(sigma|z|)        = E[sigma^2] - (2/pi) * E[sigma]^2
    #   ceiling              = E|z| * Var(sigma) / (sd(sigma) * sqrt(Var(sigma|z|)))
    # with E|z| = sqrt(2/pi). Derived rather than simulated: an earlier version
    # drew the errors with a fixed seed, which can correlate with a sigma that
    # happens to come from the same stream and inflates the ceiling.
    e_abs_z = np.sqrt(2.0 / np.pi)
    m1, m2 = float(np.mean(sigma)), float(np.mean(sigma ** 2))
    var_sigma = m2 - m1 ** 2
    var_prod = m2 - (2.0 / np.pi) * m1 ** 2
    corr_ceiling = (float(e_abs_z * var_sigma / (np.sqrt(var_sigma) * np.sqrt(var_prod)))
                    if var_sigma > 0 and var_prod > 0 else float("nan"))
    is_model, is_const = iscore(z * sigma), iscore(np.full_like(y, w_const))
    width = z * sigma
    corr = float(np.corrcoef(sigma, abs_err)[0, 1]) if sigma.std() > 0 else float("nan")
    return {
        "level": level,
        "interval_score_model": is_model,
        "interval_score_constant_width": is_const,
        "interval_score_skill_vs_constant": float(1.0 - is_model / is_const) if is_const else float("nan"),
        "constant_width": w_const,
        "mean_width": float(width.mean()),
        "width_cv": float(width.std() / width.mean()) if width.mean() else float("nan"),
        "corr_sigma_vs_abs_error": corr,
        "r2_sigma_vs_abs_error": corr ** 2 if corr == corr else float("nan"),
        "corr_ceiling_if_perfectly_specified": corr_ceiling,
        "corr_fraction_of_ceiling": float(corr / corr_ceiling) if corr_ceiling else float("nan"),
    }


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

    # Canonical member per seed: lowest validation RMSE, never sort order.
    members = best_members(OPTUNA_ROOT, args.site, horizon)
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
        bp, mck = members[s]
        # The variance net is a swapped head on ONE specific mean model. If it was
        # fitted on a different run of this seed its shapes will not match, so skip
        # it loudly instead of dying on a shape mismatch halfway through the grid.
        src = vmeta.get("source_mean_ckpt")
        if src and Path(src).resolve() != Path(mck).resolve():
            print(f"  SKIP variance net for seed {s}: fitted on {Path(src).parent.name}, "
                  f"but the canonical member is {Path(mck).parent.name}. Retrain with "
                  f"scripts/12_variance_net.py --site {args.site} --hours_ahead "
                  f"{args.hours_ahead} --seed {s}")
            continue
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

    sharp = sharpness(y_true, mu, sigma_tot, day, "90")
    print(f"\n  sharpness at 90% (combined band)")
    print(f"    interval score   model={sharp['interval_score_model']:.1f}  "
          f"constant width={sharp['interval_score_constant_width']:.1f}  "
          f"skill={sharp['interval_score_skill_vs_constant']:+.3f}")
    print(f"    width            mean={sharp['mean_width']:.1f}  cv={sharp['width_cv']:.3f}  "
          f"(flat band would be {sharp['constant_width']:.1f})")
    print(f"    corr(sigma,|err|)={sharp['corr_sigma_vs_abs_error']:+.3f}  "
          f"ceiling={sharp['corr_ceiling_if_perfectly_specified']:.3f}  "
          f"({sharp['corr_fraction_of_ceiling']:.0%} of achievable)")

    # Per-sample arrays for the figures and for any later recalibration. Gitignored
    # by the *.npz rule, so this stays out of the repo.
    np.savez_compressed(OUT_ROOT / "_tmp_arrays.npz", y=y_true, mu=mu,
                        sigma_a=sigma_a, sigma_epi=sigma_epi, day=day)

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
                       "coverage_day_combined": cov_t,
                       "sharpness_day_combined": sharp},
    }, indent=2))
    (OUT_ROOT / "_tmp_arrays.npz").replace(run_dir / "per_sample.npz")
    print(f"\nSaved -> {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
13_swag.py — SWAG and MultiSWAG epistemic uncertainty for a tuned cell.

SWAG (Maddox et al. 2019, arXiv:1902.02476): starting from the tuned model,
run SGD at a constant learning rate, keep running moments of the weights plus
a low-rank deviation matrix, and sample weights from the resulting Gaussian.
MultiSWAG: the equal-weight mixture of the per-seed SWAG posteriors.

Asked for by the advisor on 2026-11-10 after SGLD failed to mix on ResNet and
GraphSAGE. Works on every architecture in solar_uq.zoo, in particular the
three the project reports: mlp, fusion_* and convgraph.

    # one SWAG posterior per seed
    .venv/bin/python scripts/13_swag.py --arch mlp --site uniandes --hours_ahead 6 --seed 42
    # MultiSWAG over whatever seeds have a SWAG run for the cell
    .venv/bin/python scripts/13_swag.py --arch mlp --site uniandes --hours_ahead 6 --multi

Output: runs/{arch}_swag/{site}_H{H}_L{L}_P{patch}_seed{seed}_{ts}/summary.json
        + test_pred.npz (y, mu, sigma per test sample; what --multi mixes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq import zoo
from solar_uq.data import make_loader
from solar_uq.metrics import metrics_from_arrays, skill_score
from solar_uq.swag import SWAG, bn_update
from solar_uq.train import collect_predictions, eval_model, seed_everything, train_one_model

Z = {"80": 1.2816, "90": 1.6449, "95": 1.9600}


def parse_args():
    p = argparse.ArgumentParser(description="SWAG / MultiSWAG",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(zoo.ARCHS))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--k_neighbors", type=int, default=8, help="convgraph only")
    p.add_argument("--swa_epochs", type=int, default=20, help="constant-LR epochs, one collection each")
    p.add_argument("--swa_lr", type=float, default=1e-2, help="SGD learning rate of the collection phase")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--rank", type=int, default=20, help="columns of the low-rank deviation matrix")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--scale", type=float, default=0.5, help="covariance scale (paper: 0.5)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--multi", action="store_true", help="mix the existing per-seed SWAG runs of the cell")
    return p.parse_args()


def band_metrics(y, mu, sigma, base_rmse_day, thr):
    m = metrics_from_arrays(y, mu, thr)
    day = y >= thr
    out = dict(m, skill_day_vs_persistence=skill_score(m["rmse_day"], base_rmse_day),
               sigma_day_mean=float(sigma[day].mean()),
               corr_sigma_abs_err_day=float(np.corrcoef(sigma[day], np.abs(y[day] - mu[day]))[0, 1]))
    out["coverage_day"] = {k: float((np.abs(y[day] - mu[day]) <= z * sigma[day]).mean()) for k, z in Z.items()}
    return out


def run_multi(args):
    cfg = zoo.ARCHS[args.arch]
    root = PROJECT_ROOT / "runs" / f"{args.arch}_swag"
    best = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        sj = d / "summary.json"
        if not sj.is_file() or not (d / "test_pred.npz").is_file() or d.name.startswith("multi_"):
            continue
        s = json.load(open(sj))
        if s["site"] != args.site or s["hours_ahead"] != args.hours_ahead:
            continue
        best[s["seed"]] = d                       # newest run of each seed
    if len(best) < 2:
        raise SystemExit(f"MultiSWAG necesita >= 2 semillas con SWAG; hay {sorted(best)}")
    P = [np.load(d / "test_pred.npz") for _, d in sorted(best.items())]
    y = P[0]["y"]
    mus = np.stack([p["mu"] for p in P]); sig = np.stack([p["sigma"] for p in P])
    mu = mus.mean(0)
    # mixture variance = mean of the component variances + variance of the component means
    sigma = np.sqrt((sig ** 2).mean(0) + mus.var(0))
    base = json.load(open(next(iter(best.values())) / "summary.json"))["persistence_test_rmse_day"]
    res = band_metrics(y, mu, sigma, base, args.day_threshold)
    res["sigma_within_day_mean"] = float(np.sqrt((sig ** 2).mean(0))[y >= args.day_threshold].mean())
    res["sigma_between_day_mean"] = float(mus.std(0)[y >= args.day_threshold].mean())
    name = f"multi_{args.site}_h{args.hours_ahead}_M{len(best)}"
    out = root / name; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(method="multiswag", arch=args.arch, site=args.site, hours_ahead=args.hours_ahead,
                   seeds=sorted(best), members=[d.name for _, d in sorted(best.items())], final_test=res),
              open(out / "summary.json", "w"), indent=2)
    print(f"MultiSWAG {args.arch} {args.site} h{args.hours_ahead} M={len(best)} seeds={sorted(best)}")
    print(f"  skill_day={res['skill_day_vs_persistence']:.3f} sigma={res['sigma_day_mean']:.1f} "
          f"(dentro {res['sigma_within_day_mean']:.1f} / entre {res['sigma_between_day_mean']:.1f}) "
          f"cov90={res['coverage_day']['90']:.3f} corr={res['corr_sigma_abs_err_day']:.3f}")
    print(f"Saved -> {out}")


def main():
    args = parse_args()
    if args.multi:
        return run_multi(args)
    cfg = zoo.ARCHS[args.arch]; two = cfg["two_input"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    cell = zoo.canonical_run(args.arch, args.site, args.hours_ahead, args.seed)
    bp = cell["best_params"]
    D = zoo.build_data(args.arch, args.site, args.hours_ahead, args.seed, args.patch,
                       args.k_neighbors, args.day_threshold)
    norm, L = D["normalizer"], D["L"]
    bs = bp["batch_size"]
    tl = make_loader(D["train"], bs, shuffle=True, num_workers=args.num_workers, seed=args.seed, device=dev)
    vl = make_loader(D["val"], bs, shuffle=False, num_workers=0, seed=args.seed, device=dev)
    te = make_loader(D["test"], bs, shuffle=False, num_workers=0, seed=args.seed, device=dev)
    model = zoo.build_model(args.arch, bp, args.patch, L, D["d_tab"], args.k_neighbors).to(dev)
    print(f"\nSWAG | {args.arch} {args.site} h{args.hours_ahead} seed{args.seed} dev={dev} "
          f"| origen {cell['run_dir'].name}")

    if zoo.load_state(model, cell["ckpt"]):
        pre = "checkpoint"
    else:
        # Several cells lost their checkpoint; retrain the tuned configuration
        # with the same budget as the tuning scripts' own final retraining.
        print("  sin checkpoint: reentreno la configuracion ganadora (30 epocas, paciencia 8)")
        out = train_one_model(model=model, train_loader=tl, val_loader=vl, normalizer=norm,
                              lr=bp["lr"], weight_decay=bp["weight_decay"], l1_reg=bp.get("l1_reg", 0.0),
                              use_amp=(dev == "cuda"), epochs=30, patience=8,
                              day_threshold=args.day_threshold, device=dev, fusion=two)
        model = out.get("model", model); pre = "retrained"
    v0 = eval_model(model, vl, norm, args.day_threshold, dev, fusion=two)
    t0m = eval_model(model, te, norm, args.day_threshold, dev, fusion=two)
    print(f"  punto de partida ({pre}): val RMSE_day={v0['rmse_day']:.2f} test RMSE_day={t0m['rmse_day']:.2f}")

    # --- collection phase: SGD, constant learning rate, one collection per epoch ---
    opt = torch.optim.SGD(model.parameters(), lr=args.swa_lr, momentum=args.momentum,
                          weight_decay=bp.get("weight_decay", 0.0))
    lossf = nn.MSELoss(); swag = SWAG(model, rank=args.rank); trace = []
    t_start = time.time()
    for ep in range(1, args.swa_epochs + 1):
        model.train()
        for batch in tl:
            opt.zero_grad(set_to_none=True)
            if two:
                yhat = model(batch[0].to(dev), batch[1].to(dev)); y = batch[2]
            else:
                yhat = model(batch[0].to(dev)); y = batch[1]
            lossf(yhat, y.to(dev)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        swag.collect()
        v = eval_model(model, vl, norm, args.day_threshold, dev, fusion=two)["rmse_day"]
        trace.append(v)
        print(f"  swa ep {ep:02d}/{args.swa_epochs} val_rmse_day={v:.2f} ({time.time()-t_start:.0f}s)")

    # --- SWA mean ---
    swag.set_mean(); bn_update(model, tl, dev, two)
    v_swa = eval_model(model, vl, norm, args.day_threshold, dev, fusion=two)
    y_true, mu_swa = collect_predictions(model, te, norm, dev, fusion=two)

    # --- SWAG samples ---
    g = torch.Generator().manual_seed(args.seed)
    preds = []
    for s_i in range(args.n_samples):
        swag.sample(scale=args.scale, generator=g); bn_update(model, tl, dev, two)
        _, yp = collect_predictions(model, te, norm, dev, fusion=two)
        preds.append(yp)
    preds = np.stack(preds); mu = preds.mean(0); sigma = preds.std(0)
    base = D["baseline_test"]["rmse_day"]
    res = band_metrics(y_true, mu, sigma, base, args.day_threshold)
    swa = metrics_from_arrays(y_true, mu_swa, args.day_threshold)

    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    H = int(round(args.hours_ahead * 6))
    out = PROJECT_ROOT / "runs" / f"{args.arch}_swag" / f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_pred.npz", y=y_true, mu=mu, sigma=sigma, mu_swa=mu_swa)
    summary = dict(
        method="swag", arch=args.arch, site=args.site, hours_ahead=args.hours_ahead, seed=args.seed,
        source_run=cell["run_dir"].name, start=pre, n_params=int(sum(p.numel() for p in swag.params)),
        protocol=dict(swa_epochs=args.swa_epochs, swa_lr=args.swa_lr, momentum=args.momentum,
                      rank=args.rank, n_samples=args.n_samples, scale=args.scale),
        persistence_test_rmse_day=base,
        start_point=dict(val_rmse_day=v0["rmse_day"], test_rmse_day=t0m["rmse_day"],
                         test_skill_day=skill_score(t0m["rmse_day"], base)),
        swa_val_trace=trace,
        swa_mean=dict(val_rmse_day=v_swa["rmse_day"], test_rmse_day=swa["rmse_day"],
                      test_skill_day=skill_score(swa["rmse_day"], base)),
        final_test=res,
    )
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"\n=== SWAG {args.arch} {args.site} h{args.hours_ahead} seed{args.seed} ===")
    print(f"  partida  skill_day={summary['start_point']['test_skill_day']:.3f}")
    print(f"  SWA      skill_day={summary['swa_mean']['test_skill_day']:.3f}")
    print(f"  SWAG     skill_day={res['skill_day_vs_persistence']:.3f} sigma_epi={res['sigma_day_mean']:.1f} W/m2 "
          f"cov80/90/95={res['coverage_day']['80']:.3f}/{res['coverage_day']['90']:.3f}/{res['coverage_day']['95']:.3f} "
          f"corr(sigma,|err|)={res['corr_sigma_abs_err_day']:.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

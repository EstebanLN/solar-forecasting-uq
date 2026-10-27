#!/usr/bin/env python
"""
08g_sgld_screen.py — cheap screen for an SGLD regime that actually converges.

The 4 x 2500-epoch run on GraphSAGE diverged (predictions reaching +/-25,000
W/m^2 against a physical 0-1400). Rather than spend another 12 days per
configuration, this screens many (prior_precision, step size) pairs with SHORT
chains and asks one question: does the chain STAY in the warm-start basin?

Screening signal: val_rmse_day after `--epochs` epochs, against the warm-start
value and against persistence. A configuration is 'contained' if the chain ends
no worse than `--tol` times the warm-start error; only contained ones are worth
a full multi-chain run with R-hat/ESS.

Note on --sgld_prior_precision in the main driver: 100.0 is a SENTINEL meaning
"reuse Adam's weight_decay" (order 1e-3..1e-6, i.e. an extremely weak prior that
leaves the chain unconfined). This screen passes explicit precisions instead, so
avoid exactly 100.0 as a grid value.

Usage:
    .venv/bin/python scripts/08g_sgld_screen.py --arch mlp --site uniandes \
        --hours_ahead 6 --epochs 150
"""
from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from solar_uq.data import (GraphSeqDataset, PatchSeqDataset, TargetNormalizer,
                           make_loader, preload_patch_cache, read_history_steps_from_manifest)
from solar_uq.metrics import eval_persistence
from solar_uq.sgld import SGLD
from solar_uq.train import seed_everything, eval_model

_spec = importlib.util.spec_from_file_location("_sgld", PROJECT_ROOT / "scripts" / "08_sgld.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
build_model, ARCH_CFG, find_optuna_best_params = _m.build_model, _m.ARCH_CFG, _m.find_optuna_best_params


def run_one(arch, bp, ckpt, L, patch, dev, train_loader, val_loader, normalizer,
            lr, lr_final, prior, epochs, day_threshold, seed, lik_scale, clip):
    seed_everything(seed)
    model = build_model(arch, bp, patch, L, dev)
    st = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(st.get("model_state", st))
    opt = SGLD(model.parameters(), lr=lr, weight_decay=prior,
               lr_final=lr_final, total_steps=epochs * max(1, len(train_loader)),
               schedule="poly", gamma=0.55)
    lf = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            # negative log-posterior: likelihood carries the full dataset
            (lik_scale * lf(model(x.to(dev, non_blocking=True)),
                            y.to(dev, non_blocking=True))).backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return float(eval_model(model, val_loader, normalizer, day_threshold, dev)["rmse_day"])


def main():
    p = argparse.ArgumentParser(description="Screen SGLD (prior, step) regimes")
    p.add_argument("--arch", required=True, choices=list(ARCH_CFG))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--optuna_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--priors", type=float, nargs="+", default=[1.0, 10.0, 1000.0])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-7, 1e-8, 1e-9])
    p.add_argument("--sgld_sigma2", type=float, default=1.0,
                   help="observation noise variance in normalised units (loss x n/(2 sigma2))")
    p.add_argument("--clip_grad_norm", type=float, default=0.0, help="0 = off (clipping biases the sampler)")
    p.add_argument("--tol", type=float, default=1.2, help="contained if final <= tol * warm-start")
    p.add_argument("--subsample_train", type=int, default=0,
                   help="0 = full train set. Data loading scales with EPOCHS (full passes), not "
                        "with step count, so shrinking the batch does not make a screen cheap -- "
                        "subsampling does. The likelihood scale stays at the FULL n_train so the "
                        "target posterior is unchanged.")
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 = use the tuned batch. A screen with the tuned batch (often 16) costs "
                        "n/16 steps per epoch and is NOT cheap; override it to screen quickly. "
                        "Noise accumulates per STEP, so confirm the winner at the tuned batch.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--warm_start_ckpt", default=None)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    cfg = ARCH_CFG[args.arch]; dev = args.device
    horizon = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    SITE_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / args.site / f"h{args.hours_ahead}"
    PATCHES = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    preload_patch_cache(PATCHES)

    bp, found = find_optuna_best_params(PROJECT_ROOT / "runs" / cfg[args.optuna_version],
                                        args.site, horizon, args.seed)
    ckpt = args.warm_start_ckpt or found
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"No warm-start checkpoint (found={found}).")

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    L = read_history_steps_from_manifest(train_man)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())
    ground = pd.read_parquet(PROJECT_ROOT / "data" / "ground_aligned" / f"ground_10min_utc_{args.site}.parquet")
    pers = eval_persistence(val_man, ground, args.day_threshold)["rmse_day"]

    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    bs = args.batch_size if args.batch_size > 0 else int(bp.get("batch_size", 32))
    train_fit = (train_man.sample(n=min(args.subsample_train, len(train_man)),
                                  random_state=args.seed).reset_index(drop=True)
                 if args.subsample_train > 0 else train_man)
    train_loader = make_loader(DS(train_fit, PATCHES, normalizer), bs, shuffle=True,
                               num_workers=args.num_workers, seed=args.seed, device=dev)
    val_loader = make_loader(DS(val_man, PATCHES, normalizer), bs, shuffle=False,
                             num_workers=0, seed=args.seed, device=dev)

    # warm-start reference
    m0 = build_model(args.arch, bp, args.patch, L, dev)
    st = torch.load(ckpt, map_location=dev, weights_only=True)
    m0.load_state_dict(st.get("model_state", st))
    warm = float(eval_model(m0, val_loader, normalizer, args.day_threshold, dev)["rmse_day"])
    lik_scale = len(train_man) / (2.0 * args.sgld_sigma2)
    print(f"\nSGLD SCREEN | {args.arch} {args.site} h{args.hours_ahead} | epochs={args.epochs}")
    print(f"  warm-start val_rmse_day={warm:.1f} | persistence={pers:.1f} | "
          f"contained if <= {args.tol*warm:.1f}")
    print(f"  batch={bs} -> {len(train_loader)} steps/epoch, {len(train_loader)*args.epochs:,} steps per config")
    print(f"  likelihood scale n/(2*sigma2) = {lik_scale:,.0f}  (mean-MSE alone would under-weight "
          f"the likelihood by this factor)\n")
    print(f"  {'prior':>9s} {'lr':>9s} {'final':>11s} {'ratio':>7s}  verdict")

    rows = []
    for prior in args.priors:
        for lr in args.lrs:
            t0 = time.time()
            v = run_one(args.arch, bp, ckpt, L, args.patch, dev, train_loader, val_loader,
                        normalizer, lr, lr * 0.1, prior, args.epochs, args.day_threshold,
                        args.seed, lik_scale, args.clip_grad_norm)
            ratio = v / warm
            ok = ratio <= args.tol
            rows.append({"prior_precision": prior, "sgld_lr": lr, "val_rmse_day": v,
                         "ratio_vs_warm": ratio, "contained": bool(ok),
                         "seconds": time.time() - t0})
            print(f"  {prior:9.4g} {lr:9.1e} {v:11.1f} {ratio:7.2f}  "
                  f"{'CONTAINED' if ok else 'diverged'}")

    out = PROJECT_ROOT / "runs" / f"{args.arch}_sgld_screen"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.site}_h{args.hours_ahead}_seed{args.seed}_ep{args.epochs}.json"
    f.write_text(json.dumps({"arch": args.arch, "site": args.site,
                             "horizon_hours": horizon, "seed": args.seed,
                             "epochs": args.epochs, "warm_start_val_rmse_day": warm,
                             "persistence_val_rmse_day": pers, "grid": rows}, indent=2))
    best = sorted([r for r in rows if r["contained"]], key=lambda r: r["ratio_vs_warm"])
    print(f"\n  contained configs: {len(best)}/{len(rows)}")
    if best:
        b = best[0]
        print(f"  best: prior={b['prior_precision']:g} lr={b['sgld_lr']:.1e} "
              f"ratio={b['ratio_vs_warm']:.2f}")
    print(f"Saved -> {f}")


if __name__ == "__main__":
    main()

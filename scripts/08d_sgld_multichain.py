#!/usr/bin/env python
"""
08d_sgld_multichain.py — multi-chain SGLD with R-hat + ESS (Daniel's protocol).

Reunión 2026-10-06: run M independent SGLD chains (default 4) for E epochs
(default 2500) warm-started from the Optuna MAP, and diagnose convergence with
split-R-hat and bulk-ESS — the standard multi-chain MCMC diagnostics. Prereq
(the naive validation, scripts/08c_sgld_naive_validation.py) already PASSED, so
the sampler is known correct; this script applies it to the predictive nets.

Parallel chains (Daniel: 4 in parallel)
---------------------------------------
Each chain is an independent process so the 4 run concurrently on one GPU
(share ~16 GB; ~2.5 GB each). Orchestration is in scripts/run_sgld_multichain.sh:
it launches ``--chain_only {0..3} --run_dir DIR`` in parallel, waits, then runs
``--combine --run_dir DIR`` for the cross-chain R-hat/ESS. Running without
``--chain_only`` keeps the old single-process sequential behaviour.

What is monitored
-----------------
Parameter-space R-hat is ill-defined for neural nets (weight-space symmetries),
so we monitor the PREDICTIVE FUNCTION: the model's prediction at a fixed set of
``n_monitor`` validation inputs, recorded every epoch for each chain. R-hat/ESS
are computed per monitored point and summarised (max/median R-hat, min/median
ESS). We also record the per-sample validation RMSE_day trace as a scalar
chain to visualise (non)stationarity / divergence.

Output: runs/{arch}_sgld_multichain/{site}_H{H}_L{L}_P{patch}_seed{seed}_{ts}/
        summary.json + diagnostics.png  (+ per-chain chain_{i}.npz, meta.json)
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

from solar_uq.data import (
    GraphSeqDataset, PatchSeqDataset, TargetNormalizer,
    make_loader, preload_patch_cache, read_history_steps_from_manifest,
)
from solar_uq.metrics import eval_persistence, metrics_from_arrays, skill_score
from solar_uq.sgld import SGLD
from solar_uq.train import seed_everything, eval_model
from solar_uq.mcmc_diag import split_rhat, ess_bulk

# reuse the architecture registry, Optuna finder and model factory
from importlib import import_module
_m = import_module("08_sgld")
ARCH_CFG = _m.ARCH_CFG
find_optuna_best_params = _m.find_optuna_best_params
build_model = _m.build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-chain SGLD + R-hat/ESS",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(ARCH_CFG))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42, help="Optuna source seed (warm-start checkpoint)")
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--optuna_version", default="v2", choices=["v1", "v2"])
    p.add_argument("--warm_start_ckpt", default=None,
                   help="explicit warm-start .pt (overrides the path stored in the Optuna "
                        "summary, whose absolute path is machine-specific)")
    # multi-chain protocol
    p.add_argument("--n_chains", type=int, default=4, help="independent chains (Daniel: 4)")
    p.add_argument("--epochs", type=int, default=2500, help="epochs per chain (Daniel: 2500)")
    p.add_argument("--burn_in", type=int, default=500, help="epochs discarded before diagnostics")
    p.add_argument("--n_monitor", type=int, default=64, help="fixed val points whose predictions are tracked")
    p.add_argument("--val_every", type=int, default=25, help="epochs between full-val RMSE_day records")
    # parallel orchestration
    p.add_argument("--chain_only", type=int, default=-1,
                   help=">=0 runs ONLY that chain index and saves chain_{i}.npz into --run_dir (parallel mode)")
    p.add_argument("--combine", action="store_true",
                   help="load chain_*.npz from --run_dir and write diagnostics (no training)")
    p.add_argument("--run_dir", default=None,
                   help="explicit shared run dir for parallel chains / combine")
    # SGLD hyperparameters (same meaning as 08_sgld.py)
    p.add_argument("--sgld_lr", type=float, default=1e-7)
    p.add_argument("--sgld_lr_final", type=float, default=1e-8)
    p.add_argument("--sgld_schedule", default="poly", choices=["poly", "geom"])
    p.add_argument("--sgld_gamma", type=float, default=0.55)
    p.add_argument("--sgld_prior_precision", type=float, default=100.0,
                   help="100.0 sentinel = reuse Adam weak weight_decay (warm-start); else this precision")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--runs_root", default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


@torch.no_grad()
def _predict_subset(model, x_mon, normalizer, device):
    model.eval()
    yhat = model(x_mon.to(device, non_blocking=True))
    return normalizer.denormalize(yhat.detach().cpu().numpy())


def run_chain(chain_idx, args, best_params, mean_ckpt_path, L, DEVICE,
              train_loader, val_loader, x_mon, normalizer):
    """Run one SGLD chain; return per-epoch monitored preds + scalar traces."""
    seed_everything(args.seed * 1000 + 17 * (chain_idx + 1))   # chain-specific RNG
    model = build_model(args.arch, best_params, args.patch, L, DEVICE)
    state = torch.load(mean_ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)

    l1_reg = float(best_params.get("l1_reg", 0.0))
    adam_wd = float(best_params.get("weight_decay", 1e-4))
    sgld_wd = (adam_wd if args.sgld_prior_precision == 100.0 else args.sgld_prior_precision)

    total_steps = args.epochs * max(1, len(train_loader))
    opt = SGLD(model.parameters(), lr=args.sgld_lr, weight_decay=sgld_wd,
               lr_final=args.sgld_lr_final, total_steps=total_steps,
               schedule=args.sgld_schedule, gamma=args.sgld_gamma)
    loss_fn = nn.MSELoss()

    mon_preds = np.empty((args.epochs, x_mon.shape[0]), dtype=np.float64)
    val_rmse_day = []   # (epoch_recorded, value)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl = []
        for x_seq, y in train_loader:
            x_seq = x_seq.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_seq), y)
            if l1_reg > 0.0:
                loss = loss + l1_reg * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
        mon_preds[ep] = _predict_subset(model, x_mon, normalizer, DEVICE)
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            vm = eval_model(model, val_loader, normalizer, args.day_threshold, DEVICE)
            val_rmse_day.append((ep, float(vm["rmse_day"])))
            print(f"    [chain {chain_idx}] ep {ep:04d}/{args.epochs} "
                  f"train={np.mean(tl):.4f} val_rmse_day={vm['rmse_day']:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return {"mon_preds": mon_preds, "val_rmse_day": val_rmse_day,
            "seconds": time.time() - t0}


def _save_chain_npz(run_dir: Path, idx: int, res: dict) -> None:
    eps = np.array([e for e, _ in res["val_rmse_day"]], dtype=np.int64)
    vals = np.array([v for _, v in res["val_rmse_day"]], dtype=np.float64)
    np.savez_compressed(run_dir / f"chain_{idx}.npz", mon_preds=res["mon_preds"],
                        val_eps=eps, val_vals=vals, seconds=np.array([res["seconds"]]))


def _load_chains(run_dir: Path, n_chains: int) -> list:
    chains = []
    for c in range(n_chains):
        f = run_dir / f"chain_{c}.npz"
        if not f.exists():
            print(f"  [combine] missing {f.name}, skipping")
            continue
        d = np.load(f)
        chains.append({"mon_preds": d["mon_preds"],
                       "val_rmse_day": list(zip(d["val_eps"].tolist(), d["val_vals"].tolist())),
                       "seconds": float(d["seconds"][0])})
    return chains


def diagnose(args, chains, baseline_rmse_day, run_dir, n_mon, L, H, horizon_hours, mean_ckpt_path):
    """Cross-chain R-hat/ESS on the monitored predictive function + figure + summary."""
    # clamp burn-in to the actual recorded length (robust when chains ran fewer
    # epochs than args.epochs, e.g. an interrupted or debug chain)
    actual_epochs = min(ch["mon_preds"].shape[0] for ch in chains)
    bi = min(args.burn_in, max(0, actual_epochs - 4))
    mon = np.stack([ch["mon_preds"][bi:actual_epochs] for ch in chains], axis=0)   # (M, N, n_mon)
    rhat = np.asarray(split_rhat(mon))
    ess = np.asarray(ess_bulk(mon))
    rhat = rhat[np.isfinite(rhat)]
    ess = ess[np.isfinite(ess)]

    # scalar chain: full-val RMSE_day trace (common record grid across chains)
    rec_len = min(len(ch["val_rmse_day"]) for ch in chains)
    vtrace = np.stack([[v for _, v in ch["val_rmse_day"][:rec_len]] for ch in chains], axis=0)  # (M, rec_len)
    v_bi = max(1, int(bi / max(1, args.val_every)))
    vt_post = vtrace[:, v_bi:] if vtrace.shape[1] - v_bi >= 4 else vtrace
    rhat_val = float(split_rhat(vt_post))
    ess_val = float(ess_bulk(vt_post))
    drift = float(np.mean(vtrace[:, -1] - vtrace[:, 0]))

    ok_rhat = bool(np.all(rhat < 1.1))
    print("\n" + "=" * 70)
    print(f"MULTI-CHAIN SGLD DIAGNOSTICS — {args.arch} {args.site} h{args.hours_ahead} "
          f"({len(chains)} chains)")
    print(f"  monitored predictive function ({len(rhat)} points):")
    print(f"    split-R-hat: max={rhat.max():.3f} median={np.median(rhat):.3f}  "
          f"(<1.1 for all: {ok_rhat})")
    print(f"    bulk-ESS:    min={ess.min():.0f} median={np.median(ess):.0f}  "
          f"(of {mon.shape[0]*mon.shape[1]} draws)")
    print(f"  val RMSE_day scalar chain: R-hat={rhat_val:.3f} ESS={ess_val:.0f} "
          f"| mean drift first->last = {drift:+.1f} W/m^2")
    verdict = ("chains mix to a common stationary predictive dist (R-hat<1.1)"
               if ok_rhat and abs(drift) < 0.1 * baseline_rmse_day
               else "chains NOT stationary / diverge (R-hat>=1.1 or large drift) "
                    "-> SGLD does not converge on this net")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        M = len(chains)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        xs = [e for e, _ in chains[0]["val_rmse_day"][:rec_len]]
        for c in range(M):
            ax[0].plot(xs, vtrace[c], lw=1.0, alpha=0.85, label=f"chain {c}")
        ax[0].axhline(baseline_rmse_day, color="k", ls="--", lw=1, label="persistence")
        ax[0].axvline(bi, color="r", ls=":", lw=1, label="burn-in end")
        ax[0].set_title(f"(a) Val RMSE_day per chain (R-hat={rhat_val:.2f})")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val RMSE_day"); ax[0].legend(fontsize=7)
        for c in range(M):
            ax[1].plot(mon[c, :, 0], lw=0.6, alpha=0.8)
        ax[1].set_title("(b) Monitored prediction #0 (post burn-in)")
        ax[1].set_xlabel("post-burn-in epoch"); ax[1].set_ylabel("pred (W/m²)")
        ax[2].hist(rhat, bins=30, color="tab:blue", alpha=0.75)
        ax[2].axvline(1.1, color="r", ls=":", lw=1.2, label="1.1")
        ax[2].axvline(1.01, color="g", ls=":", lw=1.2, label="1.01")
        ax[2].set_title(f"(c) R-hat over {len(rhat)} monitored preds")
        ax[2].set_xlabel("split-R-hat"); ax[2].legend(fontsize=8)
        fig.suptitle(f"Multi-chain SGLD ({M}×{args.epochs}ep) — {args.arch} "
                     f"{args.site} h{args.hours_ahead}: max R-hat {rhat.max():.2f}, min ESS {ess.min():.0f}",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(run_dir / "diagnostics.png", dpi=130, bbox_inches="tight")
    except Exception as e:
        print(f"[fig] skipped ({e})")

    summary = {
        "arch": args.arch, "site": args.site, "seed": args.seed,
        "optuna_version": args.optuna_version, "n_chains_used": len(chains),
        "temporal": {"horizon_hours": horizon_hours, "history_steps": L, "horizon_steps_H": H},
        "protocol": {"n_chains": args.n_chains, "epochs": args.epochs, "burn_in": args.burn_in,
                     "n_monitor": n_mon, "val_every": args.val_every,
                     "sgld_lr": args.sgld_lr, "sgld_lr_final": args.sgld_lr_final,
                     "sgld_schedule": args.sgld_schedule, "sgld_gamma": args.sgld_gamma,
                     "warm_start_ckpt": mean_ckpt_path, "parallel": True},
        "diagnostics": {
            "pred_rhat_max": float(rhat.max()), "pred_rhat_median": float(np.median(rhat)),
            "pred_ess_min": float(ess.min()), "pred_ess_median": float(np.median(ess)),
            "val_rmse_day_rhat": rhat_val, "val_rmse_day_ess": ess_val,
            "val_rmse_day_drift": drift, "persistence_val_rmse_day": baseline_rmse_day,
            "all_rhat_below_1p1": ok_rhat,
        },
        "chain_seconds": [ch["seconds"] for ch in chains],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {run_dir}")


def main():
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    horizon_hours = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    cfg = ARCH_CFG[args.arch]

    # ---- combine mode: no data/training, just load per-chain npz + diagnose ----
    if args.combine:
        if not args.run_dir:
            raise SystemExit("--combine requires --run_dir")
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        chains = _load_chains(run_dir, args.n_chains)
        if len(chains) < 2:
            raise SystemExit(f"need >=2 chains to combine, found {len(chains)}")
        diagnose(args, chains, meta["persistence_val_rmse_day"], run_dir,
                 meta["n_monitor"], meta["L"], meta["H"], meta["horizon_hours"],
                 meta["warm_start_ckpt"])
        return

    print(f"\nMULTI-CHAIN SGLD | arch={args.arch} optuna={args.optuna_version} "
          f"site={args.site} h={args.hours_ahead} seed={args.seed} device={DEVICE} "
          f"{'chain_only='+str(args.chain_only) if args.chain_only >= 0 else 'all-sequential'}")
    print(f"  chains={args.n_chains} epochs={args.epochs} burn_in={args.burn_in} "
          f"| lr {args.sgld_lr:.1e}->{args.sgld_lr_final:.1e} poly(gamma={args.sgld_gamma})", flush=True)

    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR = PROJECT_ROOT / "data" / "ground_aligned"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    OPTUNA_ROOT = PROJECT_ROOT / "runs" / cfg[args.optuna_version]
    OUT_ROOT = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / f"{args.arch}_sgld_multichain"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists() and PATCHES_ROOT.exists()
    preload_patch_cache(PATCHES_ROOT)

    best_params, found_ckpt = find_optuna_best_params(OPTUNA_ROOT, args.site, horizon_hours, args.seed)
    mean_ckpt_path = args.warm_start_ckpt or found_ckpt
    if mean_ckpt_path is None or not Path(mean_ckpt_path).exists():
        raise FileNotFoundError(
            f"No usable warm-start checkpoint. found={found_ckpt} override={args.warm_start_ckpt}. "
            "Pass --warm_start_ckpt pointing to the .pt on this machine.")

    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    H = int(meta["horizon_steps"])
    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    L = read_history_steps_from_manifest(train_man)

    if args.debug:
        train_man = train_man.sample(n=400, random_state=args.seed).reset_index(drop=True)
        val_man = val_man.sample(n=160, random_state=args.seed).reset_index(drop=True)
        args.epochs, args.burn_in, args.val_every = 12, 4, 2
        args.n_monitor = 32

    seed_everything(args.seed)
    ground = pd.read_parquet(GROUND_DIR / f"ground_10min_utc_{args.site}.parquet")
    baseline_val = eval_persistence(val_man, ground, args.day_threshold)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())

    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    train_ds = DS(train_man, PATCHES_ROOT, normalizer)
    val_ds = DS(val_man, PATCHES_ROOT, normalizer)
    batch_size = int(best_params.get("batch_size", 32))
    train_loader = make_loader(train_ds, batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, num_workers=0, seed=args.seed, device=DEVICE)

    n_mon = min(args.n_monitor, len(val_ds))
    x_mon = torch.stack([val_ds[i][0] for i in range(n_mon)], dim=0)
    print(f"  H={H} L={L} | warm-start={Path(mean_ckpt_path).name} | n_monitor={n_mon} "
          f"| persistence val_rmse_day={baseline_val['rmse_day']:.1f}", flush=True)

    # resolve the shared run dir (parallel chains + combine all point here)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
        run_dir = OUT_ROOT / f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "persistence_val_rmse_day": baseline_val["rmse_day"], "n_monitor": n_mon,
        "L": L, "H": H, "horizon_hours": horizon_hours, "warm_start_ckpt": mean_ckpt_path}))

    # ---- parallel: run ONLY one chain, save its npz, exit ----
    if args.chain_only >= 0:
        res = run_chain(args.chain_only, args, best_params, mean_ckpt_path, L, DEVICE,
                        train_loader, val_loader, x_mon, normalizer)
        _save_chain_npz(run_dir, args.chain_only, res)
        print(f"[chain {args.chain_only}] done in {res['seconds']:.0f}s → {run_dir}/chain_{args.chain_only}.npz",
              flush=True)
        return

    # ---- default: all chains sequentially in one process ----
    chains = []
    for c in range(args.n_chains):
        print(f"\n  === chain {c+1}/{args.n_chains} ===", flush=True)
        chains.append(run_chain(c, args, best_params, mean_ckpt_path, L, DEVICE,
                                train_loader, val_loader, x_mon, normalizer))
    diagnose(args, chains, baseline_val["rmse_day"], run_dir, n_mon, L, H, horizon_hours, mean_ckpt_path)


if __name__ == "__main__":
    main()

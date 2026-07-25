#!/usr/bin/env python
"""
07_sgld.py — SGLD posterior sampling for all neural solar-forecasting architectures.

Unified entry point: one script handles ResNet+LSTM, GraphSAGE+LSTM, and FlatMLP.
Hyperparameters are loaded automatically from the corresponding Optuna run
(v1 or v2), so no manual tuning is needed to start sampling.

Protocol
--------
1. Find the Optuna run for (arch, optuna_version, site, hours_ahead, seed).
2. Instantiate the model and WARM-START it from the Optuna-trained checkpoint
   (default), so SGLD samples the local posterior around the good solution
   rather than trying to train from scratch under the tiny SGLD step size
   (fresh init underfits at sgld_lr=1e-5 -- pass --fresh_init to force it).
3. Run SGLD burn-in (default 20 epochs, samples discarded) to let the Langevin
   noise equilibrate into the local posterior.
4. Run SGLD sampling (default 10 checkpoints × 5 epochs apart).
5. Evaluate the ensemble mean on the test set → write summary.json compatible
   with 08_results_table.py.

Usage (from project root):
    .venv/bin/python scripts/07_sgld.py --arch resnet --site elpaso --hours_ahead 3 --seed 42
    .venv/bin/python scripts/07_sgld.py --arch graphsage --site uniandes --hours_ahead 1 --seed 7
    .venv/bin/python scripts/07_sgld.py --arch mlp --site elpaso --hours_ahead 6 --seed 13
    .venv/bin/python scripts/07_sgld.py --arch resnet --optuna_version v1 --site elpaso --hours_ahead 3 --seed 42

Output directory: runs/{arch}_sgld/{site}_H{H}_L{L}_P{patch}_seed{seed}_{timestamp}/
"""
from __future__ import annotations

import argparse
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
from solar_uq.models.graphsage_lstm import (
    GraphSAGE_LSTM,
    build_edge_index_8n,
    build_weighted_knn_edge_index,
)
from solar_uq.models.mlp import FlatMLP
from solar_uq.models.resnet_lstm import ResNetLSTM
from solar_uq.train import seed_everything, eval_model
from solar_uq.train_sgld import train_sgld


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

# Maps --arch to: optuna run dirs (v1, v2) and sgld output dir.
# MLP has only one optuna version; v2 falls back to the same directory.
ARCH_CFG = {
    "resnet": {
        "v1":      "resnet_lstm_optuna",
        "v2":      "resnet_lstm_optuna_v2",
        "sgld":    "resnet_lstm_sgld",
        "dataset": "patch",
    },
    "graphsage": {
        "v1":      "graphsage_lstm_optuna",
        "v2":      "graphsage_lstm_optuna_v2",
        "sgld":    "graphsage_lstm_sgld",
        "dataset": "graph",
    },
    "mlp": {
        "v1":      "mlp_optuna",
        "v2":      "mlp_optuna",        # MLP has no v2 — same directory
        "sgld":    "mlp_sgld",
        "dataset": "patch",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SGLD posterior sampling — all architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arch",            required=True, choices=list(ARCH_CFG))
    p.add_argument("--site",            default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",     type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--patch",           type=int, default=16)
    p.add_argument("--optuna_version",  default="v2", choices=["v1", "v2"],
                   help="Which Optuna run to read best_params from")
    # SGLD hyperparameters
    p.add_argument("--sgld_lr",         type=float, default=1e-7,
                   help="SGLD step size ε. Default 1e-7, calibrated on "
                        "full-resolution data: each epoch is ~3.4k SGLD steps "
                        "(54k train samples / batch 16), so the per-step Langevin "
                        "noise accumulates over many more steps than the original "
                        "1e-5 (set for a much smaller/subsampled setup) assumed. "
                        "Verified sweep from a warm-started backbone (Uniandes "
                        "ResNet 1h): 1e-5 diverges (train loss 0.9->78 in 4 "
                        "epochs), 1e-6 degrades the fit toward the predict-mean "
                        "floor (ensemble skill ~0.03), 1e-7 keeps train loss "
                        "stable at the backbone level (~0.36) with ensemble "
                        "skill 0.184 (>= backbone 0.175) and a non-degenerate "
                        "sigma_epi (~17 W/m^2).")
    p.add_argument("--sgld_prior_precision", type=float, default=100.0,
                   help="Gaussian prior precision for the SGLD confining term "
                        "(NOT the Optuna-tuned Adam weight_decay). The chain's "
                        "relaxation timescale to its confined equilibrium is "
                        "~1/(sgld_lr * precision) steps, so precision must "
                        "satisfy sgld_lr * precision * total_epochs >> 1 to "
                        "actually confine the chain within the sampling "
                        "budget -- reusing Optuna's Adam weight_decay "
                        "(~1e-6) or an under-scaled precision like 1e-2 "
                        "leaves an effectively unconfined random walk that "
                        "diverges over 1000+ epochs (empirically verified: "
                        "1e-2 -> val_rmse_day explodes into the "
                        "thousands/tens-of-thousands by epoch ~150; "
                        "100-1000 -> stays within ~20% of the persistence "
                        "baseline over 150 epochs)")
    p.add_argument("--burn_in",         type=int, default=20,
                   help="Epochs to discard before collecting samples. Default 20 "
                        "(warm-start): the chain starts AT the Optuna-trained mode, "
                        "so burn-in only needs to let the Langevin noise equilibrate "
                        "into the local posterior, not train from scratch. The "
                        "confined-chain relaxation timescale is ~1/(sgld_lr*prior) "
                        "= ~1000 steps << one epoch (~3.4k steps), so 20 epochs is "
                        "well past equilibration. (A prior fresh-init attempt with a "
                        "much longer schedule underfit badly: at sgld_lr=1e-5, plain "
                        "SGD-like updates cannot train a net from random init in any "
                        "practical epoch budget -- train loss stayed at the "
                        "predict-the-mean floor. Warm-start fixes this at the root.)")
    p.add_argument("--sample_every",    type=int, default=5,
                   help="Epochs between consecutive checkpoint saves. Default 5: "
                        "the sample autocorrelation time near the mode is "
                        "~1/(sgld_lr*prior) ~ 1000 steps (~0.3 epoch), so 5-epoch "
                        "spacing (~17k steps) yields well-decorrelated samples.")
    p.add_argument("--n_samples",       type=int, default=10,
                   help="Number of posterior checkpoints to collect")
    p.add_argument("--fresh_init",      action="store_true",
                   help="Start SGLD from a fresh random init instead of warm-starting "
                        "from the Optuna checkpoint. NOT recommended: at sgld_lr=1e-5 "
                        "the chain cannot train from scratch and underfits. Kept only "
                        "to reproduce the original (broken) behaviour if ever needed.")
    # Training misc
    p.add_argument("--num_workers",     type=int, default=0)
    p.add_argument("--day_threshold",   type=float, default=20.0)
    p.add_argument("--runs_root",       default=None,
                   help="Override output directory (default: runs/{arch}_sgld)")
    p.add_argument("--debug",           action="store_true",
                   help="Subsample data for fast smoke-test")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Optuna run finder
# ---------------------------------------------------------------------------

def find_optuna_best_params(
    runs_root: Path,
    site: str,
    horizon_hours: float,
    seed: int,
) -> tuple[dict, str | None]:
    """Scan runs_root for a summary.json matching (site, horizon_hours, seed).

    Returns ``(optuna.best_params, best_ckpt_path)`` from the first matching
    run. ``best_ckpt_path`` is the trained mean-network checkpoint used for
    warm-starting SGLD (None if the run has no usable checkpoint).
    Raises FileNotFoundError if no matching run is found.
    """
    if not runs_root.exists():
        raise FileNotFoundError(
            f"Optuna runs directory not found: {runs_root}\n"
            f"Run the corresponding Optuna script first."
        )

    for run_dir in sorted(runs_root.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        meta = json.loads(sj.read_text(encoding="utf-8"))
        if (
            meta.get("site") == site
            and meta.get("temporal", {}).get("horizon_hours") == horizon_hours
            and meta.get("seed") == seed
        ):
            bp = meta.get("optuna", {}).get("best_params")
            if bp is None:
                raise ValueError(f"summary.json in {run_dir} has no optuna.best_params")
            ckpt = meta.get("best_model", {}).get("best_ckpt_path")
            if ckpt is not None and not Path(ckpt).exists():
                ckpt = None
            print(f"  Optuna source: {run_dir.name}")
            print(f"  best_params:   {bp}")
            print(f"  checkpoint:    {ckpt if ckpt else '(none — warm-start unavailable)'}")
            return bp, ckpt

    raise FileNotFoundError(
        f"No completed Optuna run found in {runs_root} "
        f"for site={site}, horizon_hours={horizon_hours}, seed={seed}.\n"
        f"Check that the run finished and has a summary.json."
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    arch: str,
    best_params: dict,
    patch: int,
    L: int,
    device: str,
) -> torch.nn.Module:
    """Instantiate the correct model using Optuna best_params.

    GraphSAGE: also builds the weighted k-NN graph and registers it as a buffer.
    """
    if arch == "resnet":
        model = ResNetLSTM(
            in_ch=16,
            base=best_params["base"],
            emb_dim=best_params["emb_dim"],
            hidden_t=best_params["hidden_t"],
            dropout=best_params.get("dropout", 0.0),
            n_lstm_layers=best_params.get("n_lstm_layers", 1),
        )

    elif arch == "graphsage":
        # best_params only contains "k_neighbors" for studies that actually
        # tuned the weighted k-NN graph (the ablation, v2, introduced in the
        # 2026-05-20 rewrite of 06_graphsage_lstm_optuna.py). Runs predating
        # that ablation (v1, the fixed-graph baseline used in the paper's
        # main results Table 2) were built with the fixed unweighted
        # 8-neighbour grid (build_edge_index_8n) -- reconstructing them with
        # a k-NN graph instead silently changes the topology (and, since the
        # two graphs have different edge counts, makes the saved checkpoint
        # fail to load with a size-mismatch error rather than a helpful one).
        if "k_neighbors" in best_params:
            edge_index, edge_weight = build_weighted_knn_edge_index(patch, best_params["k_neighbors"])
        else:
            edge_index, edge_weight = build_edge_index_8n(patch), None
        model = GraphSAGE_LSTM(
            in_dim=16,
            hidden_g=best_params["hidden_g"],
            n_sage_layers=best_params["n_sage_layers"],
            hidden_t=best_params["hidden_t"],
            n_lstm_layers=best_params.get("n_lstm_layers", 1),
            dropout_head=best_params.get("dropout_head", 0.0),
            input_bn=best_params.get("input_bn", False),
            concat_agg=best_params.get("concat_agg", True),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    elif arch == "mlp":
        model = FlatMLP(
            L=L,
            C=16,
            n_layers=best_params["n_layers"],
            hidden_dim=best_params["hidden_dim"],
            dropout=best_params.get("dropout", 0.0),
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg        = ARCH_CFG[args.arch]
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    FREQ_MIN   = 10  # fixed across all datasets

    # Compute horizon_hours for matching Optuna runs
    horizon_hours = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]

    print(
        f"\nSGLD | arch={args.arch} | optuna={args.optuna_version} | "
        f"site={args.site} | hours={args.hours_ahead} | seed={args.seed} | "
        f"device={DEVICE}"
    )
    print(
        f"  burn_in={args.burn_in} | sample_every={args.sample_every} | "
        f"n_samples={args.n_samples} | sgld_lr={args.sgld_lr:.1e}"
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    OPTUNA_ROOT  = PROJECT_ROOT / "runs" / cfg[args.optuna_version]
    SGLD_ROOT    = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / cfg["sgld"]
    SGLD_ROOT.mkdir(parents=True, exist_ok=True)

    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"
    preload_patch_cache(PATCHES_ROOT)

    # ------------------------------------------------------------------
    # Optuna best_params
    # ------------------------------------------------------------------
    print(f"\nLooking for Optuna run in: {OPTUNA_ROOT}")
    best_params, mean_ckpt_path = find_optuna_best_params(
        OPTUNA_ROOT, args.site, horizon_hours, args.seed
    )
    if not args.fresh_init and mean_ckpt_path is None:
        raise FileNotFoundError(
            "Warm-start requested (default) but the Optuna run has no usable "
            "checkpoint on this machine. Either run where the .pt exists, or "
            "pass --fresh_init (not recommended — underfits at sgld_lr=1e-5)."
        )
    l1_reg           = float(best_params.get("l1_reg", 0.0))
    adam_weight_decay = float(best_params.get("weight_decay", 1e-4))
    batch_size       = int(best_params.get("batch_size", 32))
    # Prior precision (Gaussian prior N(0, 1/lambda) confining term):
    #   fresh-init: a strong prior (--sgld_prior_precision, default 100) is
    #     needed to keep the from-scratch chain from diverging over a long run.
    #   warm-start: the chain starts at the trained mode, which sits far from
    #     zero; a strong zero-centred prior would drag the weights back toward
    #     zero (gradient lambda*theta) and destroy the fit (verified: lambda=100
    #     makes train loss climb and val explode). We therefore sample under the
    #     SAME weak Gaussian prior the backbone was trained with (Adam's tuned
    #     weight_decay), so SGLD samples the local posterior consistent with the
    #     point estimate rather than a different, over-regularised one. Override
    #     with --sgld_prior_precision if given explicitly.
    if args.fresh_init:
        sgld_weight_decay = args.sgld_prior_precision
    else:
        sgld_weight_decay = (
            args.sgld_prior_precision if args.sgld_prior_precision != 100.0
            else adam_weight_decay
        )

    # ------------------------------------------------------------------
    # Dataset & loaders
    # ------------------------------------------------------------------
    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    H         = int(meta["horizon_steps"])
    GRID_SIZE = int(meta["grid_size"])

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man   = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")

    L = read_history_steps_from_manifest(train_man)
    print(f"  H={H} ({H * FREQ_MIN / 60:.1f}h) | L={L} ({L * FREQ_MIN / 60:.1f}h)")

    if args.debug:
        train_man = train_man.sample(n=400, random_state=args.seed).reset_index(drop=True)
        val_man   = val_man.sample(n=120,  random_state=args.seed).reset_index(drop=True)
        test_man  = test_man.sample(n=120, random_state=args.seed).reset_index(drop=True)
        # Override SGLD schedule for a fast end-to-end smoke-test
        args.burn_in      = 3
        args.n_samples    = 2
        args.sample_every = 2

    seed_everything(args.seed)

    ground_path = GROUND_DIR / f"ground_10min_utc_{args.site}.parquet"
    ground      = pd.read_parquet(ground_path)

    baseline_train = eval_persistence(train_man, ground, args.day_threshold)
    baseline_val   = eval_persistence(val_man,   ground, args.day_threshold)
    baseline_test  = eval_persistence(test_man,  ground, args.day_threshold)
    print(f"  Persistence test RMSE_day={baseline_test['rmse_day']:.1f}")

    y_train_arr = train_man["y"].astype(float).to_numpy()
    normalizer  = TargetNormalizer.from_train(y_train_arr)

    # Dataset class is architecture-specific
    DS = GraphSeqDataset if cfg["dataset"] == "graph" else PatchSeqDataset
    train_ds = DS(train_man, PATCHES_ROOT, normalizer)
    val_ds   = DS(val_man,   PATCHES_ROOT, normalizer)
    test_ds  = DS(test_man,  PATCHES_ROOT, normalizer)

    # num_workers=0 by default: forking a DataLoader worker after this process
    # has touched CUDA can crash with "CUDA error: initialization error"
    # (c10::cuda::ExchangeDevice) -- a real risk here given SGLD chains run
    # for up to 1500 epochs in one process. Patches are already preloaded to
    # RAM via preload_patch_cache(), so this costs little.
    train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)

    # ------------------------------------------------------------------
    # Model instantiation (+ warm-start from the Optuna-trained checkpoint)
    # ------------------------------------------------------------------
    model = build_model(args.arch, best_params, args.patch, L, DEVICE)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"  Model params: {n_params:,}")

    if args.fresh_init:
        print("  Init: FRESH random (--fresh_init) -- NOT recommended, underfits at "
              "sgld_lr=1e-5")
    else:
        # Warm-start: load the Optuna-trained weights so SGLD samples the
        # LOCAL posterior around the good solution instead of trying (and
        # failing) to train from scratch under the tiny SGLD step size.
        state = torch.load(mean_ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        print(f"  Init: WARM-START from {Path(mean_ckpt_path).name}")

    # ------------------------------------------------------------------
    # Run directory (created inside train_sgld)
    # ------------------------------------------------------------------
    run_ts   = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{run_ts}"
    RUN_DIR  = SGLD_ROOT / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # SGLD training
    # ------------------------------------------------------------------
    print(f"\nStarting SGLD — total epochs: {args.burn_in + args.n_samples * args.sample_every}")
    sgld_out = train_sgld(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        run_dir=RUN_DIR,
        sgld_lr=args.sgld_lr,
        weight_decay=sgld_weight_decay,
        l1_reg=l1_reg,
        burn_in=args.burn_in,
        sample_every=args.sample_every,
        n_samples=args.n_samples,
        day_threshold=args.day_threshold,
        device=DEVICE,
    )

    # ------------------------------------------------------------------
    # Skill scores (need baseline reference)
    # ------------------------------------------------------------------
    et = sgld_out["ensemble_test"]
    et["skill_vs_persistence"]     = skill_score(et["rmse"],     baseline_test["rmse"])
    et["skill_day_vs_persistence"] = skill_score(et["rmse_day"], baseline_test["rmse_day"])

    print(
        f"\n=== Final ensemble test ===\n"
        f"  RMSE={et['rmse']:.2f}  RMSE_day={et['rmse_day']:.2f}\n"
        f"  skill={et['skill_vs_persistence']:.3f}  skill_day={et['skill_day_vs_persistence']:.3f}\n"
        f"  uncertainty std_mean={et['ensemble_std_mean']:.2f}  "
        f"std_day_mean={et.get('ensemble_std_day_mean', float('nan')):.2f}"
    )

    # ------------------------------------------------------------------
    # summary.json — compatible with 08_results_table.py
    # ------------------------------------------------------------------
    ARCH_LABELS = {
        "resnet":    "SmallResNetEncoder + LayerNorm + LSTM + MLP head",
        "graphsage": "GraphSAGE + LSTM + MLP head (weighted k-NN graph)",
        "mlp":       "FlatMLP (spatial avg-pool)",
    }

    summary = {
        "run_name":       run_name,
        "arch":           args.arch,
        "site":           args.site,
        "device":         DEVICE,
        "seed":           args.seed,
        "debug":          args.debug,
        "day_threshold":  args.day_threshold,
        "optuna_source": {
            "version":    args.optuna_version,
            "runs_root":  str(OPTUNA_ROOT),
            "best_params": best_params,
        },
        "data_paths": {
            "site_dir":    str(SITE_DIR),
            "patches_root": str(PATCHES_ROOT),
            "ground_path": str(ground_path),
            "run_dir":     str(RUN_DIR),
        },
        "temporal": {
            "freq_min":        FREQ_MIN,
            "history_steps_L": L,
            "horizon_steps_H": H,
            "history_hours":   L * FREQ_MIN / 60.0,
            "horizon_hours":   H * FREQ_MIN / 60.0,
        },
        "spatial": {
            "grid_size": GRID_SIZE,
            "patch":     args.patch,
            "channels":  16,
        },
        "target_norm": {
            "y_mean_train":        normalizer.mean,
            "y_std_train":         normalizer.std,
            "y_percentiles_train": np.percentile(y_train_arr, [0, 50, 90, 95, 99]).tolist(),
        },
        "baselines": {
            "persistence_train": baseline_train,
            "persistence_val":   baseline_val,
            "persistence_test":  baseline_test,
        },
        "sgld": {
            "init":               "fresh" if args.fresh_init else "warm_start",
            "warm_start_ckpt":    None if args.fresh_init else mean_ckpt_path,
            "sgld_lr":            args.sgld_lr,
            "sgld_weight_decay":  sgld_weight_decay,
            "adam_weight_decay":  adam_weight_decay,
            "burn_in":         args.burn_in,
            "sample_every":    args.sample_every,
            "n_samples":       args.n_samples,
            "total_epochs":    args.burn_in + args.n_samples * args.sample_every,
            "n_collected":     sgld_out["n_samples_collected"],
            "checkpoint_paths": sgld_out["checkpoint_paths"],
            "train_seconds_total": sgld_out["train_seconds_total"],
        },
        # "best_model" key mirrors Optuna summary structure → 08_results_table.py works as-is
        "best_model": {
            "arch":              ARCH_LABELS[args.arch],
            "n_params":          n_params,
            "train_seconds_total": sgld_out["train_seconds_total"],
            "inference_method":  "ensemble_mean",
            "final_val":         None,   # not computed; SGLD has no val-based selection
            "final_test":        et,
        },
    }

    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {RUN_DIR}")


if __name__ == "__main__":
    main()

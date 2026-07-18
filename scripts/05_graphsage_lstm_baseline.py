#!/usr/bin/env python
"""
05_graphsage_lstm_baseline.py — Train an improved GraphSAGE+LSTM point-forecast baseline.

Improvements over the original notebook baseline:
  - BatchNorm1d on node features (channel normalisation before first SAGE layer).
  - Standard SAGEConv aggregation (concat + linear) controlled by --concat_agg.
  - Configurable number of SAGE layers (--n_sage_layers).
  - LayerNorm on the graph-level embedding before LSTM.
  - AdamW optimizer (consistent with ResNet).
  - edge_index registered as model buffer → same forward(x_seq) API as ResNetLSTM.

Usage (from project root):
    python scripts/05_graphsage_lstm_baseline.py
    python scripts/05_graphsage_lstm_baseline.py --site elpaso --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------
# Resolve project root and add src/ to path
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.data import (
    TargetNormalizer,
    GraphSeqDataset,
    make_loader,
    read_history_steps_from_manifest,
)
from solar_uq.metrics import eval_persistence, skill_score
from solar_uq.models.graphsage_lstm import GraphSAGE_LSTM, build_edge_index_8n
from solar_uq.train import seed_everything, train_one_model, eval_model


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphSAGE+LSTM baseline")
    p.add_argument("--site",           default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",    type=int,   default=6, choices=[1, 3, 6])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--patch",          type=int,   default=16)
    p.add_argument("--debug",          action="store_true")
    # Model hyper-parameters
    p.add_argument("--hidden_g",       type=int,   default=64)
    p.add_argument("--n_sage_layers",  type=int,   default=2)
    p.add_argument("--hidden_t",       type=int,   default=96)
    p.add_argument("--n_lstm_layers",  type=int,   default=1)
    p.add_argument("--dropout_head",   type=float, default=0.1)
    p.add_argument("--input_bn",       type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--concat_agg",     type=lambda x: x.lower() != "false", default=True)
    # Optimisation
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--patience",       type=int,   default=8)
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--day_threshold",  type=float, default=20.0)
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Directories
    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    RUNS_ROOT    = PROJECT_ROOT / "runs" / "graphsage_lstm"
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"

    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE={DEVICE} | site={args.site} | hours_ahead={args.hours_ahead} | seed={args.seed}")

    # Dataset meta
    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    FREQ_MIN  = int(meta["freq_min"])
    H         = int(meta["horizon_steps"])
    GRID_SIZE = int(meta["grid_size"])

    # Manifests
    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man   = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")

    L = read_history_steps_from_manifest(train_man)
    print(f"H={H} ({H*FREQ_MIN/60:.1f}h) | L={L} ({L*FREQ_MIN/60:.1f}h)")
    print(f"Train={len(train_man)} | Val={len(val_man)} | Test={len(test_man)}")

    if args.debug:
        train_man = train_man.sample(n=4000, random_state=args.seed).reset_index(drop=True)
        val_man   = val_man.sample(n=1200,  random_state=args.seed).reset_index(drop=True)
        test_man  = test_man.sample(n=1200, random_state=args.seed).reset_index(drop=True)
        print(f"[DEBUG] Train={len(train_man)} Val={len(val_man)} Test={len(test_man)}")

    # Seeding
    seed_everything(args.seed)

    # Ground truth & persistence baseline
    ground_path = GROUND_DIR / f"ground_10min_utc_{args.site}.parquet"
    assert ground_path.exists(), f"Missing ground parquet: {ground_path}"
    ground = pd.read_parquet(ground_path)
    assert "ghi" in ground.columns and str(ground.index.tz) == "UTC"

    baseline_train = eval_persistence(train_man, ground, args.day_threshold)
    baseline_val   = eval_persistence(val_man,   ground, args.day_threshold)
    baseline_test  = eval_persistence(test_man,  ground, args.day_threshold)
    print(f"Persistence test: RMSE={baseline_test['rmse']:.1f}  RMSE_day={baseline_test['rmse_day']:.1f}")

    # Target normalizer
    y_train_arr = train_man["y"].astype(float).to_numpy()
    normalizer  = TargetNormalizer.from_train(y_train_arr)
    print(f"Target: mean={normalizer.mean:.2f}  std={normalizer.std:.2f}")

    # Graph structure (P×P 8-neighbourhood)
    N_NODES    = args.patch * args.patch
    edge_index = build_edge_index_8n(args.patch)
    print(f"Graph: {N_NODES} nodes, {edge_index.shape[1]} edges")

    # Datasets & loaders
    USE_AMP  = (DEVICE == "cuda")
    train_ds = GraphSeqDataset(train_man, PATCHES_ROOT, normalizer)
    val_ds   = GraphSeqDataset(val_man,   PATCHES_ROOT, normalizer)
    test_ds  = GraphSeqDataset(test_man,  PATCHES_ROOT, normalizer)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)
    test_loader  = make_loader(test_ds,  args.batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)

    # Model — edge_index registered as buffer so forward(x_seq) works uniformly
    model = GraphSAGE_LSTM(
        in_dim=16,
        hidden_g=args.hidden_g,
        n_sage_layers=args.n_sage_layers,
        hidden_t=args.hidden_t,
        n_lstm_layers=args.n_lstm_layers,
        dropout_head=args.dropout_head,
        input_bn=args.input_bn,
        concat_agg=args.concat_agg,
        edge_index=edge_index,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GraphSAGE_LSTM params: {n_params/1e6:.3f}M")

    # Train
    out = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=USE_AMP,
        epochs=args.epochs,
        patience=args.patience,
        day_threshold=args.day_threshold,
        device=DEVICE,
    )

    # Final evaluation on test set
    final_val  = out["final_val"]
    final_test = eval_model(model, test_loader, normalizer, args.day_threshold, DEVICE)

    final_val["skill_vs_persistence"]      = skill_score(final_val["rmse"],      baseline_val["rmse"])
    final_val["skill_day_vs_persistence"]  = skill_score(final_val["rmse_day"],  baseline_val["rmse_day"])
    final_test["skill_vs_persistence"]     = skill_score(final_test["rmse"],     baseline_test["rmse"])
    final_test["skill_day_vs_persistence"] = skill_score(final_test["rmse_day"], baseline_test["rmse_day"])

    print("=== Final evaluation ===")
    print(f"Val  RMSE={final_val['rmse']:.2f}  skill={final_val['skill_vs_persistence']:.3f}")
    print(f"Test RMSE={final_test['rmse']:.2f}  skill={final_test['skill_vs_persistence']:.3f}  "
          f"RMSE_day={final_test['rmse_day']:.2f}  skill_day={final_test['skill_day_vs_persistence']:.3f}")

    # Save run artifacts
    run_ts   = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{run_ts}"
    RUN_DIR  = RUNS_ROOT / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    BEST_PATH = RUN_DIR / "best_model.pt"
    torch.save(
        {
            "epoch":             out["best_epoch"],
            "model_state":       model.state_dict(),
            "best_val_rmse_day": out["best_val_rmse_day"],
            "meta": {
                "arch": "GraphSAGE_LSTM",
                "arch_hparams": {
                    "hidden_g": args.hidden_g, "n_sage_layers": args.n_sage_layers,
                    "hidden_t": args.hidden_t, "n_lstm_layers": args.n_lstm_layers,
                    "dropout_head": args.dropout_head,
                    "input_bn": args.input_bn, "concat_agg": args.concat_agg,
                },
                "site": args.site, "patch": args.patch, "L": L, "H": H, "seed": args.seed,
                "y_mean_train": normalizer.mean, "y_std_train": normalizer.std,
            },
        },
        BEST_PATH,
    )

    summary = {
        "run_name": run_name,
        "site": args.site,
        "device": DEVICE,
        "seed": args.seed,
        "debug": args.debug,
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
            "grid_size":    GRID_SIZE,
            "patch":        args.patch,
            "site_center_rc": meta.get("site_center_pix"),
            "n_nodes":      N_NODES,
            "n_edges":      int(edge_index.shape[1]),
            "channels":     16,
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
        "model": {
            "arch":              "GraphSAGE_LSTM (improved)",
            "hidden_g":          args.hidden_g,
            "n_sage_layers":     args.n_sage_layers,
            "hidden_t":          args.hidden_t,
            "n_lstm_layers":     args.n_lstm_layers,
            "dropout_head":      args.dropout_head,
            "input_bn":          args.input_bn,
            "concat_agg":        args.concat_agg,
            "optimizer":         "AdamW",
            "lr_init":           args.lr,
            "weight_decay":      args.weight_decay,
            "batch_size":        args.batch_size,
            "num_workers":       args.num_workers,
            "use_amp":           USE_AMP,
            "grad_clip_norm":    1.0,
            "epochs_max":        args.epochs,
            "patience":          args.patience,
            "day_threshold":     args.day_threshold,
            "best_epoch":        out["best_epoch"],
            "best_val_rmse_day": out["best_val_rmse_day"],
            "train_seconds_total": out["train_seconds_total"],
            "n_params":          n_params,
            "best_ckpt_path":    str(BEST_PATH),
            "final_val":         final_val,
            "final_test":        final_test,
        },
        "training_log": out["train_log"],
    }

    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to: {RUN_DIR}")


if __name__ == "__main__":
    main()

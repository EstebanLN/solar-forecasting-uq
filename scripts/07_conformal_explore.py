#!/usr/bin/env python
"""
07_conformal_explore.py — Split Conformal Prediction on a trained GHI model.

Loads a completed run (baseline or Optuna), runs Split CP calibration on the
validation set, and evaluates coverage and interval width on the test set.

Usage (from project root):
    python scripts/07_conformal_explore.py \\
        --run_dir runs/resnet_lstm/uniandes_H6_L24_P16_seed42_20260421_191000

    python scripts/07_conformal_explore.py \\
        --run_dir runs/graphsage_lstm/elpaso_H6_L24_P16_seed1_20260418_... \\
        --alphas 0.05 0.10 0.20

Results are printed as a table and saved to <run_dir>/conformal_splitcp.json.

Notes
-----
- Arch hyper-parameters are read from the checkpoint meta when available
  (runs produced after the arch_hparams update).  For older checkpoints, the
  script falls back to the baseline defaults (--base, --hidden_t, etc.).
- Data paths are computed from PROJECT_ROOT the same way as the training
  scripts, so the script must be run on the same machine as training.
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
# Project root
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.conformal import evaluate_coverage_by_alpha
from solar_uq.data import (
    GraphSeqDataset,
    PatchSeqDataset,
    TargetNormalizer,
    make_loader,
    read_history_steps_from_manifest,
)
from solar_uq.models.graphsage_lstm import GraphSAGE_LSTM, build_edge_index_8n
from solar_uq.models.resnet_lstm import ResNetLSTM
from solar_uq.train import collect_predictions


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split CP exploration on a trained run")
    p.add_argument("--run_dir",     required=True,
                   help="Path to a completed run directory (contains best_model.pt)")
    p.add_argument("--alphas",      nargs="+", type=float, default=[0.05, 0.10, 0.20],
                   help="Miscoverage levels to evaluate (default: 0.05 0.10 0.20)")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--day_threshold", type=float, default=20.0)
    # Fallback arch hparams for ResNetLSTM (used only if not stored in checkpoint)
    p.add_argument("--base",        type=int,   default=32)
    p.add_argument("--emb_dim",     type=int,   default=128)
    p.add_argument("--hidden_t",    type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.10)
    # Fallback arch hparams for GraphSAGE_LSTM
    p.add_argument("--hidden_g",      type=int,   default=64)
    p.add_argument("--n_sage_layers", type=int,   default=2)
    p.add_argument("--n_lstm_layers", type=int,   default=1)
    p.add_argument("--dropout_head",  type=float, default=0.10)
    return p.parse_args()


# ------------------------------------------------------------------
# Model loading helpers
# ------------------------------------------------------------------

def _load_resnet(ckpt: dict, fallback: argparse.Namespace) -> ResNetLSTM:
    hp = ckpt["meta"].get("arch_hparams", {})
    return ResNetLSTM(
        in_ch=16,
        base=hp.get("base",     fallback.base),
        emb_dim=hp.get("emb_dim",  fallback.emb_dim),
        hidden_t=hp.get("hidden_t", fallback.hidden_t),
        dropout=hp.get("dropout",  fallback.dropout),
    )


def _load_graphsage(ckpt: dict, patch: int, fallback: argparse.Namespace) -> GraphSAGE_LSTM:
    hp = ckpt["meta"].get("arch_hparams", {})
    edge_index = build_edge_index_8n(patch)
    return GraphSAGE_LSTM(
        in_dim=16,
        hidden_g=hp.get("hidden_g",      fallback.hidden_g),
        n_sage_layers=hp.get("n_sage_layers", fallback.n_sage_layers),
        hidden_t=hp.get("hidden_t",      fallback.hidden_t),
        n_lstm_layers=hp.get("n_lstm_layers", fallback.n_lstm_layers),
        dropout_head=hp.get("dropout_head",  fallback.dropout_head),
        input_bn=hp.get("input_bn",      True),
        concat_agg=hp.get("concat_agg",  True),
        edge_index=edge_index,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    ckpt_path = run_dir / "best_model.pt"
    assert ckpt_path.exists(), f"best_model.pt not found in {run_dir}"

    summary_path = run_dir / "summary.json"
    assert summary_path.exists(), f"summary.json not found in {run_dir}"

    # ------------------------------------------------------------------
    # Load checkpoint and derive run config
    # ------------------------------------------------------------------
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    meta       = ckpt["meta"]
    arch       = meta["arch"]               # "ResNetLSTM" or "GraphSAGE_LSTM"
    site       = meta["site"]
    patch      = int(meta["patch"])
    H          = int(meta["H"])
    normalizer = TargetNormalizer(mean=meta["y_mean_train"], std=meta["y_std_train"])

    # H (horizon steps) → hours_ahead: H=6→1h, H=18→3h, H=36→6h
    hours_ahead = H // 6
    assert hours_ahead in (1, 3, 6), f"Unexpected horizon_steps H={H}"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"arch={arch}  site={site}  hours_ahead={hours_ahead}  patch={patch}  device={DEVICE}")

    # ------------------------------------------------------------------
    # Data paths (same convention as training scripts)
    # ------------------------------------------------------------------
    SITE_DIR     = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / site / f"h{hours_ahead}"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / site / f"P{patch}"

    assert SITE_DIR.exists(),     f"Missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"

    val_man  = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(val_man)
    print(f"H={H}  L={L}  n_val={len(val_man)}  n_test={len(test_man)}")

    # ------------------------------------------------------------------
    # Build model and loaders
    # ------------------------------------------------------------------
    if "ResNet" in arch:
        model = _load_resnet(ckpt, args)
        val_ds  = PatchSeqDataset(val_man,  PATCHES_ROOT, normalizer)
        test_ds = PatchSeqDataset(test_man, PATCHES_ROOT, normalizer)
    else:
        model = _load_graphsage(ckpt, patch, args)
        val_ds  = GraphSeqDataset(val_man,  PATCHES_ROOT, normalizer)
        test_ds = GraphSeqDataset(test_man, PATCHES_ROOT, normalizer)

    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {n_params/1e6:.3f}M params")

    val_loader  = make_loader(val_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, device=DEVICE)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, device=DEVICE)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    print("Running inference on val  set …")
    y_val,  yhat_val  = collect_predictions(model, val_loader,  normalizer, DEVICE)
    print("Running inference on test set …")
    y_test, yhat_test = collect_predictions(model, test_loader, normalizer, DEVICE)

    val_mae_day = float(np.abs(y_val  - yhat_val )[y_val  >= args.day_threshold].mean())
    tst_mae_day = float(np.abs(y_test - yhat_test)[y_test >= args.day_threshold].mean())
    print(f"Point forecast — val MAE_day={val_mae_day:.2f}  test MAE_day={tst_mae_day:.2f}")

    # ------------------------------------------------------------------
    # Split CP at multiple alpha levels
    # ------------------------------------------------------------------
    print(f"\nCalibrating Split CP on val set (n={len(y_val)}) …")
    rows = evaluate_coverage_by_alpha(
        y_val, yhat_val, y_test, yhat_test,
        alphas=sorted(args.alphas),
        day_threshold=args.day_threshold,
    )

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    header = f"{'alpha':>6}  {'target':>7}  {'q_hat':>7}  {'cov':>6}  {'cov_day':>8}  {'width':>7}  {'width_day':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        cov_day   = f"{r['coverage_day']:.4f}"   if r['coverage_day']   is not None else "  n/a  "
        width_day = f"{r['mean_width_day_wm2']:.1f}" if r['mean_width_day_wm2'] is not None else "  n/a"
        print(
            f"{r['alpha']:>6.2f}  "
            f"{r['target_coverage']:>7.4f}  "
            f"{r['q_hat_wm2']:>7.1f}  "
            f"{r['coverage']:>6.4f}  "
            f"{cov_day:>8}  "
            f"{r['mean_width_wm2']:>7.1f}  "
            f"{width_day:>10}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "run_name":    run_dir.name,
        "arch":        arch,
        "site":        site,
        "hours_ahead": hours_ahead,
        "n_val":       int(len(y_val)),
        "n_test":      int(len(y_test)),
        "day_threshold": args.day_threshold,
        "split_cp":    rows,
    }
    out_path = run_dir / "conformal_splitcp.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

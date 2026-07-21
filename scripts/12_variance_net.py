#!/usr/bin/env python
"""
12_variance_net.py — Aleatoric variance network (Step 2 of cooperative BNN-VE).

Following Yi & Bessa (2025, arXiv:2505.02743): with the already-trained,
Optuna-winning mean network FROZEN, train a separate network to predict the
squared residual r = (mu(x) - y)^2 via a Gamma likelihood, giving an
aleatoric variance sigma_a^2(x) = alpha(x)/lambda(x) disentangled from the
point forecast. See src/solar_uq/variance_net.py for the loss and a note on
why the standard (self-derived) Gamma NLL is used instead of transcribing
the paper's image-rendered Eq. 6 verbatim.

This is Step 2 only. Step 3 (warm-starting SGLD from the mean network and
using this fixed sigma_a^2(x) in its likelihood) is not yet implemented —
see docs/article/sections/conclusion.tex and the project memory note
project_uq_cooperative_bnn.md.

Usage (from project root):
    .venv/bin/python scripts/12_variance_net.py --arch resnet --site uniandes --hours_ahead 3 --seed 42
    .venv/bin/python scripts/12_variance_net.py --arch resnet --site uniandes --hours_ahead 3 --seed 42 --debug

Output directory: runs/{arch}_variance/{site}_H{H}_L{L}_P{patch}_seed{seed}_{timestamp}/
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
from solar_uq.models.graphsage_lstm import (
    GraphSAGE_LSTM,
    build_edge_index_8n,
    build_weighted_knn_edge_index,
)
from solar_uq.models.mlp import FlatMLP
from solar_uq.models.resnet_lstm import ResNetLSTM
from solar_uq.train import seed_everything
from solar_uq.variance_net import gamma_nll_loss, gamma_params, aleatoric_variance, make_variance_model


# ---------------------------------------------------------------------------
# Architecture registry (mirrors scripts/08_sgld.py)
# ---------------------------------------------------------------------------

ARCH_CFG = {
    "resnet": {
        "v1":      "resnet_lstm_optuna",
        "v2":      "resnet_lstm_optuna_v2",
        "out":     "resnet_lstm_variance",
        "dataset": "patch",
    },
    "graphsage": {
        "v1":      "graphsage_lstm_optuna",
        "v2":      "graphsage_lstm_optuna_v2",
        "out":     "graphsage_lstm_variance",
        "dataset": "graph",
    },
    "mlp": {
        "v1":      "mlp_optuna",
        "v2":      "mlp_optuna",
        "out":     "mlp_variance",
        "dataset": "patch",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aleatoric variance network training (Step 2, cooperative BNN-VE)")
    p.add_argument("--arch",           required=True, choices=list(ARCH_CFG))
    p.add_argument("--site",           default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",    type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--patch",          type=int, default=16)
    p.add_argument("--optuna_version", default="v1", choices=["v1", "v2"])
    p.add_argument("--epochs",         type=int, default=40)
    p.add_argument("--patience",       type=int, default=8)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--num_workers",    type=int, default=0)
    p.add_argument("--day_threshold",  type=float, default=20.0)
    p.add_argument("--runs_root",      default=None)
    p.add_argument("--debug", action="store_true", help="Subsample data + 2 epochs for a fast smoke-test")
    return p.parse_args()


def find_optuna_run(runs_root: Path, site: str, horizon_hours: float, seed: int) -> dict:
    """Scan runs_root for the summary.json matching (site, horizon_hours, seed).

    Returns the full summary dict (needs both optuna.best_params and
    best_model.best_ckpt_path).
    """
    if not runs_root.exists():
        raise FileNotFoundError(f"Optuna runs directory not found: {runs_root}")
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
            if meta.get("optuna", {}).get("best_params") is None:
                raise ValueError(f"summary.json in {run_dir} has no optuna.best_params")
            ckpt_path = meta.get("best_model", {}).get("best_ckpt_path")
            if not ckpt_path or not Path(ckpt_path).exists():
                raise FileNotFoundError(f"summary.json in {run_dir} has no usable best_ckpt_path")
            print(f"  Optuna source: {run_dir.name}")
            return meta
    raise FileNotFoundError(
        f"No completed Optuna run with a saved checkpoint found in {runs_root} "
        f"for site={site}, horizon_hours={horizon_hours}, seed={seed}."
    )


def build_model(arch: str, best_params: dict, patch: int, L: int, device: str) -> torch.nn.Module:
    """Instantiate the correct model using Optuna best_params (mirrors 08_sgld.py)."""
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
        # tuned the weighted k-NN graph (the ablation, v2). Runs predating
        # that ablation (v1, the fixed-graph baseline used in the paper's
        # main results Table 2) were built with the fixed unweighted
        # 8-neighbour grid (build_edge_index_8n) — reconstructing them with
        # a k-NN graph instead silently changes the topology and, if the
        # edge counts happen to differ (as they do here), makes the saved
        # checkpoint fail to load at all. Branch on whether k_neighbors is
        # actually present rather than defaulting to k-NN.
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
            L=L, C=16,
            n_layers=best_params["n_layers"],
            hidden_dim=best_params["hidden_dim"],
            dropout=best_params.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)


@torch.no_grad()
def _epoch_val_nll(mean_model, variance_model, loader, normalizer, device) -> float:
    variance_model.eval()
    losses = []
    for x_seq, y_norm in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        mu_norm = mean_model(x_seq)
        y_phys  = normalizer.denormalize(y_norm)
        mu_phys = normalizer.denormalize(mu_norm)
        r = (mu_phys - y_phys) ** 2
        raw = variance_model(x_seq)
        alpha, lam = gamma_params(raw)
        losses.append(gamma_nll_loss(alpha, lam, r).item())
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    cfg = ARCH_CFG[args.arch]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FREQ_MIN = 10
    horizon_hours = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]

    print(f"\nVariance-net (Step 2, cooperative BNN-VE) | arch={args.arch} | "
          f"site={args.site} | hours={args.hours_ahead} | seed={args.seed} | device={DEVICE}")

    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    OPTUNA_ROOT  = PROJECT_ROOT / "runs" / cfg[args.optuna_version]
    OUT_ROOT     = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / cfg["out"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"
    preload_patch_cache(PATCHES_ROOT)

    print(f"\nLooking for Optuna run (with checkpoint) in: {OPTUNA_ROOT}")
    optuna_meta = find_optuna_run(OPTUNA_ROOT, args.site, horizon_hours, args.seed)
    best_params = optuna_meta["optuna"]["best_params"]
    ckpt_path   = optuna_meta["best_model"]["best_ckpt_path"]
    print(f"  best_params: {best_params}")
    print(f"  mean-network checkpoint: {ckpt_path}")

    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    H = int(meta["horizon_steps"])

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man   = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)
    print(f"  H={H} ({H * FREQ_MIN / 60:.1f}h) | L={L} ({L * FREQ_MIN / 60:.1f}h)")

    # The normalizer MUST be fit on the full, unfiltered training manifest —
    # exactly as scripts/06_resnet_lstm_optuna.py / 06_graphsage_lstm_optuna.py
    # do before any day-time filtering — because it must match the
    # normalization the frozen mean-network checkpoint was actually trained
    # under. Fitting it on a different (e.g. day-filtered) subset would
    # silently corrupt every physical-unit denormalization below.
    y_train_full = train_man["y"].astype(float).to_numpy()
    normalizer = TargetNormalizer.from_train(y_train_full)

    # Only *after* the normalizer is fixed do we restrict to daytime samples
    # (GHI >= day_threshold) for the variance network itself, matching the
    # project-wide RMSE_day/MAE_day convention: at night all models predict
    # near-zero trivially, so night-time residuals are not informative
    # aleatoric signal and would dilute the daytime fit if included.
    n_before = {"train": len(train_man), "val": len(val_man), "test": len(test_man)}
    train_man = train_man[train_man["y"] >= args.day_threshold].reset_index(drop=True)
    val_man   = val_man[val_man["y"]   >= args.day_threshold].reset_index(drop=True)
    test_man  = test_man[test_man["y"] >= args.day_threshold].reset_index(drop=True)
    print(f"  Daytime filter (>= {args.day_threshold} W/m^2), applied AFTER "
          f"fitting the normalizer on the full training set: "
          f"train {n_before['train']}->{len(train_man)} | "
          f"val {n_before['val']}->{len(val_man)} | "
          f"test {n_before['test']}->{len(test_man)}")

    if args.debug:
        train_man = train_man.sample(n=min(2000, len(train_man)), random_state=args.seed).reset_index(drop=True)
        val_man   = val_man.sample(n=min(600, len(val_man)),  random_state=args.seed).reset_index(drop=True)
        test_man  = test_man.sample(n=min(600, len(test_man)), random_state=args.seed).reset_index(drop=True)
        # NOTE: --epochs/--patience are NOT overridden here (unlike
        # scripts/08_sgld.py's --debug) -- pass them explicitly to control
        # smoke-test length; --debug only subsamples data.

    seed_everything(args.seed)

    DatasetCls = PatchSeqDataset if cfg["dataset"] == "patch" else GraphSeqDataset
    train_ds = DatasetCls(train_man, PATCHES_ROOT, normalizer)
    val_ds   = DatasetCls(val_man,   PATCHES_ROOT, normalizer)
    test_ds  = DatasetCls(test_man,  PATCHES_ROOT, normalizer)

    batch_size = int(best_params.get("batch_size", 32))
    train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)

    # Frozen mean network (Step 1, already trained) ------------------------
    mean_model = build_model(args.arch, best_params, args.patch, L, DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    mean_model.load_state_dict(state["model_state"] if "model_state" in state else state)
    mean_model.eval()
    for p in mean_model.parameters():
        p.requires_grad_(False)

    # Fresh variance network (Step 2, this script) --------------------------
    variance_model = make_variance_model(build_model(args.arch, best_params, args.patch, L, DEVICE))
    variance_model = variance_model.to(DEVICE)  # the swapped-in head starts on CPU

    optimizer = torch.optim.Adam(variance_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{run_ts}"
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_nll = float("inf")
    best_epoch = 0
    patience_left = args.patience
    train_log = []
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        variance_model.train()
        tr_losses = []
        for x_seq, y_norm in train_loader:
            x_seq  = x_seq.to(DEVICE, non_blocking=True)
            y_norm = y_norm.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                mu_norm = mean_model(x_seq)
            y_phys  = normalizer.denormalize(y_norm)
            mu_phys = normalizer.denormalize(mu_norm)
            r = (mu_phys - y_phys) ** 2

            optimizer.zero_grad(set_to_none=True)
            raw = variance_model(x_seq)
            alpha, lam = gamma_params(raw)
            loss = gamma_nll_loss(alpha, lam, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(variance_model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        val_nll = _epoch_val_nll(mean_model, variance_model, val_loader, normalizer, DEVICE)
        entry = {
            "epoch": epoch,
            "train_gamma_nll": float(np.mean(tr_losses)),
            "val_gamma_nll": val_nll,
            "epoch_seconds": time.time() - t0,
        }
        train_log.append(entry)
        print(f"[{epoch:03d}/{args.epochs}] train_nll={entry['train_gamma_nll']:.4f} "
              f"val_nll={val_nll:.4f} ({entry['epoch_seconds']:.1f}s)")

        if val_nll < best_val_nll - 1e-4:
            best_val_nll = val_nll
            best_epoch = epoch
            patience_left = args.patience
            torch.save({"model_state": variance_model.state_dict(), "epoch": epoch},
                       run_dir / "best_variance_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # Reload best checkpoint for test-set evaluation
    best_state = torch.load(run_dir / "best_variance_model.pt", map_location=DEVICE, weights_only=True)
    variance_model.load_state_dict(best_state["model_state"])
    variance_model.eval()

    with torch.no_grad():
        abs_resid, sigma_a = [], []
        for x_seq, y_norm in test_loader:
            x_seq  = x_seq.to(DEVICE, non_blocking=True)
            y_norm = y_norm.to(DEVICE, non_blocking=True)
            mu_norm = mean_model(x_seq)
            y_phys  = normalizer.denormalize(y_norm)
            mu_phys = normalizer.denormalize(mu_norm)
            raw = variance_model(x_seq)
            alpha, lam = gamma_params(raw)
            sigma2 = aleatoric_variance(alpha, lam)
            abs_resid.append((mu_phys - y_phys).abs().cpu().numpy())
            sigma_a.append(torch.sqrt(sigma2).cpu().numpy())
        abs_resid = np.concatenate(abs_resid)
        sigma_a   = np.concatenate(sigma_a)

    test_nll = _epoch_val_nll(mean_model, variance_model, test_loader, normalizer, DEVICE)
    corr = float(np.corrcoef(abs_resid, sigma_a)[0, 1]) if len(abs_resid) > 1 else float("nan")

    summary = {
        "run_name": run_name,
        "arch": args.arch,
        "site": args.site,
        "seed": args.seed,
        "horizon_hours": horizon_hours,
        "source_mean_ckpt": ckpt_path,
        "best_epoch": best_epoch,
        "train_seconds_total": time.time() - t_total,
        "test": {
            "gamma_nll": test_nll,
            "sigma_a_mean": float(sigma_a.mean()),
            "sigma_a_std": float(sigma_a.std()),
            "abs_residual_mean": float(abs_resid.mean()),
            "corr_sigma_a_vs_abs_residual": corr,
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nDone. sigma_a mean={sigma_a.mean():.2f} W/m^2 | "
          f"corr(sigma_a, |residual|)={corr:.3f} | test_gamma_nll={test_nll:.4f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()

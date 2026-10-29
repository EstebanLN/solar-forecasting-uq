#!/usr/bin/env python
"""
06e_surface_only_optuna.py — surface-only baseline (no satellite input).

Why: the fusion models beat the satellite-only ones at every site and horizon,
but their tabular vector includes GHI at history step t -- which is exactly the
persistence predictor. Without this baseline, "satellite and surface are
complementary" cannot be separated from "the surface channel alone already
carries most of the signal". The gap

    skill(fusion)  -  skill(surface-only)

is the honest measure of what the satellite imagery adds on top of surface data.

The architecture mirrors the fusion model downstream of the fusion point (same
projector, LayerNorm, LSTM and head), so the comparison isolates the satellite
branch. Sample selection also mirrors it (filter_missing_patches), so both are
scored on the same rows. No patch is ever read, which makes this very cheap.

Usage:
    .venv/bin/python scripts/06e_surface_only_optuna.py --site uniandes \
        --hours_ahead 6 --seed 42 --n_trials 40
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np, optuna, pandas as pd, torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.data import (TargetNormalizer, make_loader, preload_patch_cache,
                           read_history_steps_from_manifest)
from solar_uq.loaders.fusion_dataset import SurfaceSeqDataset, FusionPatchSeqDataset
from solar_uq.metrics import eval_persistence, metrics_from_arrays, skill_score
from solar_uq.models.surface_only_lstm import SurfaceOnlyLSTM
from solar_uq.train import seed_everything, eval_model


def main() -> None:
    p = argparse.ArgumentParser(description="Surface-only baseline for the fusion claim")
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--n_trials", type=int, default=75,
                   help="Matches the 75 trials the satellite models get. A baseline tuned "
                        "with a smaller budget inflates skill(fusion) - skill(surface-only), "
                        "which is exactly the quantity this baseline exists to measure.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--runs_root", default=None)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    DEV = args.device
    horizon = {1: 1.0, 3: 3.0, 6: 6.0}[args.hours_ahead]
    SITE_DIR = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / args.site / f"h{args.hours_ahead}"
    PATCHES = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    SURF = PROJECT_ROOT / "data" / "ground_aligned" / f"ground_10min_utc_{args.site}.parquet"
    OUT = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / "surface_only"
    OUT.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    # No patch is ever READ here, but the datasets still run filter_missing_patches
    # to keep sample parity with the fusion models. Without the cache that check
    # stats ~1.3M paths; with it, it is an in-memory set lookup.
    preload_patch_cache(PATCHES)
    meta = json.loads((SITE_DIR / "dataset_meta.json").read_text())
    H = int(meta["horizon_steps"])
    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)
    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())

    # Tabular standardisation fitted on TRAIN only (same helper the fusion models use)
    tab_stats = FusionPatchSeqDataset.compute_tab_stats(
        train_man, PATCHES, normalizer, SURF)

    ds = lambda man: SurfaceSeqDataset(man, PATCHES, normalizer, SURF, tab_stats=tab_stats)
    train_ds, val_ds, test_ds = ds(train_man), ds(val_man), ds(test_man)
    d_tab = train_ds.d_tab
    ground = pd.read_parquet(SURF)
    base_val = eval_persistence(val_man, ground, args.day_threshold)
    base_test = eval_persistence(test_man, ground, args.day_threshold)
    print(f"\nSURFACE-ONLY | {args.site} h{args.hours_ahead} seed={args.seed} d_tab={d_tab} "
          f"| train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"  persistence val RMSE_day={base_val['rmse_day']:.1f} test={base_test['rmse_day']:.1f}")

    def build(tr):
        return SurfaceOnlyLSTM(
            d_tab=d_tab,
            emb_dim=tr.suggest_categorical("emb_dim", [64, 128, 192]),
            hidden_t=tr.suggest_categorical("hidden_t", [64, 128, 256]),
            n_lstm_layers=tr.suggest_categorical("n_lstm_layers", [1, 2]),
            dropout=tr.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
        ).to(DEV)

    def fit(model, lr, wd, bs, epochs, patience):
        tl = make_loader(train_ds, bs, shuffle=True, num_workers=args.num_workers,
                         seed=args.seed, device=DEV)
        vl = make_loader(val_ds, bs, shuffle=False, num_workers=0, seed=args.seed, device=DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        lf, best, bad, best_state = nn.MSELoss(), float("inf"), 0, None
        for _ in range(epochs):
            model.train()
            for x, y in tl:
                opt.zero_grad(set_to_none=True)
                lf(model(x.to(DEV)), y.to(DEV)).backward()
                opt.step()
            v = eval_model(model, vl, normalizer, args.day_threshold, DEV)["rmse_day"]
            if v < best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return best

    def objective(tr):
        seed_everything(args.seed)
        return fit(build(tr),
                   tr.suggest_float("lr", 5e-5, 5e-3, log=True),
                   tr.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3]),
                   tr.suggest_categorical("batch_size", [32, 64, 128]),
                   args.epochs, args.patience)

    t0 = time.time()
    # Study on disk, named per (site, horizon, seed). The in-memory version lost
    # three ~19-hour Uniandes studies when the machine rebooted for a kernel
    # upgrade on 2026-10-29; nothing of a killed in-memory study is recoverable.
    STUDY_NAME = f"surface_only_{args.site}_h{args.hours_ahead}_s{args.seed}"
    storage_dir = PROJECT_ROOT / "runs" / "optuna_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_dir / (STUDY_NAME + '.db')}"
    study = optuna.create_study(study_name=STUDY_NAME,
                                storage=storage_url,
                                load_if_exists=True,
                                direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    # A progress bar is for a terminal; a log file wants one line per trial. Optuna
    # silences its per-trial INFO records whenever the bar is on, which is how a
    # 26-hour run ended up with zero visible progress in its log.
    interactive = sys.stderr.isatty()
    optuna.logging.set_verbosity(optuna.logging.WARNING if interactive else optuna.logging.INFO)
    # Only finished trials count toward the budget, so a resumed study runs what
    # is left instead of another full n_trials.
    _done_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    n_done = sum(1 for t in study.trials if t.state in _done_states)
    n_remaining = max(0, args.n_trials - n_done)
    print(f"  optuna: {n_done} finished in storage, {n_remaining} remaining of {args.n_trials}", flush=True)
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=interactive)
    bp = study.best_params
    print(f"\n  best val RMSE_day={study.best_value:.2f} | {bp}")

    seed_everything(args.seed)
    model = SurfaceOnlyLSTM(d_tab=d_tab, emb_dim=bp["emb_dim"], hidden_t=bp["hidden_t"],
                            n_lstm_layers=bp["n_lstm_layers"], dropout=bp["dropout"]).to(DEV)
    fit(model, bp["lr"], bp["weight_decay"], bp["batch_size"], args.epochs, args.patience)
    tl = make_loader(test_ds, bp["batch_size"], shuffle=False, num_workers=0,
                     seed=args.seed, device=DEV)
    tm = eval_model(model, tl, normalizer, args.day_threshold, DEV)
    tm["skill_vs_persistence"] = skill_score(tm["rmse"], base_test["rmse"])
    tm["skill_day_vs_persistence"] = skill_score(tm["rmse_day"], base_test["rmse_day"])
    print(f"  TEST RMSE_day={tm['rmse_day']:.2f}  skill_day={tm['skill_day_vs_persistence']:.3f}")

    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    rd = OUT / f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{ts}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "summary.json").write_text(json.dumps({
        "model": "surface_only_lstm", "site": args.site, "seed": args.seed,
        "patch": args.patch, "d_tab": d_tab,
        "temporal": {"horizon_hours": horizon, "history_steps": L, "horizon_steps_H": H},
        "optuna": {"n_trials": args.n_trials, "best_params": bp,
                   "best_val_rmse_day": study.best_value},
        "baselines": {"persistence_val": base_val, "persistence_test": base_test},
        "final_test": tm, "train_seconds_total": time.time() - t0,
    }, indent=2))
    print(f"Saved -> {rd}")


if __name__ == "__main__":
    main()

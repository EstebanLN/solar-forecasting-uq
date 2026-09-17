"""Optuna tuning for the conv+graph hybrid (satellite-only): convolutional encoder
on the exact site crop + GraphSAGE over K random neighbouring patches (one node
each, drawn per step from the per-hour pool) → concat → LSTM.

Reuses the `fusion=True` path of solar_uq.train (loader yields 3-tuples
(center, neigh, y); model is called as model(center, neigh)).

Usage (from repo root):
    .venv/bin/python scripts/06_convgraph_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 42
    .venv/bin/python scripts/06_convgraph_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 1 --debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq.data import (
    NeighborPoolDataset, TargetNormalizer, make_loader,
    preload_patch_cache, preload_pool_cache, read_history_steps_from_manifest,
)
from solar_uq.metrics import eval_persistence, skill_score
from solar_uq.models.convgraph_lstm import ConvGraphLSTM
from solar_uq.train import seed_everything, train_one_model, eval_model

FREQ_MIN = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConvGraph-LSTM Optuna tuning")
    p.add_argument("--site",          default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",   type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--patch",         type=int, default=16)
    p.add_argument("--k_neighbors",   type=int, default=8)
    p.add_argument("--n_trials",      type=int, default=75)
    p.add_argument("--num_workers",   type=int, default=0)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--pool_root",     default=None, help="default: data/neighbor_pool_v1/<site>")
    p.add_argument("--runs_root",     default=None, help="default: runs/convgraph_lstm")
    p.add_argument("--debug",         action="store_true")
    return p.parse_args()


def make_objective(train_ds, val_ds, normalizer, k_neighbors, device, seed, day_threshold, use_amp):
    def objective(trial: optuna.Trial) -> float:
        base_c        = trial.suggest_categorical("base_c",        [16, 32, 48])
        emb_c         = trial.suggest_categorical("emb_c",         [64, 128, 192])
        base_n        = trial.suggest_categorical("base_n",        [8, 16, 32])
        emb_n         = trial.suggest_categorical("emb_n",         [32, 64, 96])
        hidden_g      = trial.suggest_categorical("hidden_g",      [48, 64, 128])
        n_sage_layers = trial.suggest_categorical("n_sage_layers", [1, 2, 3])
        concat_agg    = trial.suggest_categorical("concat_agg",    [True, False])
        hidden_t      = trial.suggest_categorical("hidden_t",      [64, 128, 192])
        n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
        dropout       = trial.suggest_categorical("dropout",       [0.0, 0.1, 0.2])
        lr            = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        l1_reg        = trial.suggest_categorical("l1_reg", [0.0, 1e-5, 1e-4])
        batch_size    = trial.suggest_categorical("batch_size", [8, 16, 32])

        # num_workers=4 (spawn): the neighbour pool is memory-mapped, so workers
        # share it via the OS page cache with no parse/copy → parallel dataloading
        # that actually feeds the GPU for this heavy (K neighbour encoders) model.
        train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=4, seed=seed, device=device)
        val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=0, seed=seed, device=device)

        model = ConvGraphLSTM(
            in_ch=16, k_neighbors=k_neighbors,
            base_c=base_c, emb_c=emb_c, base_n=base_n, emb_n=emb_n,
            hidden_g=hidden_g, n_sage_layers=n_sage_layers, concat_agg=concat_agg,
            hidden_t=hidden_t, n_lstm_layers=n_lstm_layers, dropout=dropout,
        ).to(device)

        try:
            out = train_one_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                normalizer=normalizer, lr=lr, weight_decay=weight_decay, l1_reg=l1_reg,
                use_amp=use_amp, epochs=20, patience=6, day_threshold=day_threshold,
                device=device, fusion=True, optuna_trial=trial,
            )
        except torch.cuda.OutOfMemoryError as e:
            del model; torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"CUDA OOM: {e}")

        vm = out["final_val"]
        trial.set_user_attr("best_epoch", out["best_epoch"])
        trial.set_user_attr("val_rmse_day", float(vm["rmse_day"]))
        trial.set_user_attr("n_params", int(sum(p.numel() for p in model.parameters())))
        return float(vm["rmse_day"])

    return objective


def main() -> None:
    args = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    H = args.hours_ahead * 60 // FREQ_MIN
    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    SITE_DIR     = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    POOL_ROOT    = Path(args.pool_root) if args.pool_root else PROJECT_ROOT / "data" / "neighbor_pool_v1" / args.site
    RUNS_ROOT    = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs" / "convgraph_lstm"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"
    assert POOL_ROOT.exists(), f"Missing neighbour pool: {POOL_ROOT} (run 03_build_neighbor_pool.py)"

    preload_patch_cache(PATCHES_ROOT)
    # Neighbour pools are memory-mapped per access (see NeighborPoolDataset);
    # spawned workers share them via the OS page cache — no RAM preload needed.

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man   = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man  = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    if args.debug:
        train_man = train_man.sample(n=400, random_state=args.seed).reset_index(drop=True)
        val_man   = val_man.sample(n=120,  random_state=args.seed).reset_index(drop=True)
        test_man  = test_man.sample(n=120,  random_state=args.seed).reset_index(drop=True)
    L = read_history_steps_from_manifest(train_man)
    print(f"DEVICE={DEVICE} | site={args.site} h={args.hours_ahead} seed={args.seed} "
          f"K={args.k_neighbors} | H={H} L={L} | n_trials={args.n_trials}")

    normalizer = TargetNormalizer.from_train(train_man["y"].astype(float).to_numpy())
    ground = pd.read_parquet(GROUND_DIR / f"ground_10min_utc_{args.site}.parquet")
    baseline_val  = eval_persistence(val_man,  ground, args.day_threshold)
    baseline_test = eval_persistence(test_man, ground, args.day_threshold)
    print(f"  Persistence test RMSE_day={baseline_test['rmse_day']:.1f}")

    def mk(man):
        return NeighborPoolDataset(man, PATCHES_ROOT, POOL_ROOT, normalizer, k_neighbors=args.k_neighbors)
    train_ds, val_ds, test_ds = mk(train_man), mk(val_man), mk(test_man)

    # Persistent Optuna storage so an interrupted run resumes instead of restarting.
    STUDY = f"convgraph_lstm_{args.site}_h{args.hours_ahead}_s{args.seed}_P{args.patch}"
    storage_dir = PROJECT_ROOT / "runs" / "optuna_storage"; storage_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_dir / (STUDY + '.db')}"
    study = optuna.create_study(
        study_name=STUDY, storage=storage_url, load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_remaining = max(0, args.n_trials - n_done)
    print(f"\nStarting Optuna ({n_done} finished in storage, {n_remaining} remaining of {args.n_trials}) ...")
    if n_remaining:
        study.optimize(
            make_objective(train_ds, val_ds, normalizer, args.k_neighbors, DEVICE,
                           args.seed, args.day_threshold, use_amp=(DEVICE == "cuda")),
            n_trials=n_remaining,
        )

    best = study.best_trial
    bp = best.params
    print(f"\nBest trial val_rmse_day={best.value:.3f} | params={bp}")

    # Retrain best config (longer) and evaluate on test.
    run_ts   = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    RUN_DIR  = RUNS_ROOT / f"{args.site}_H{H}_L{L}_P{args.patch}_K{args.k_neighbors}_seed{args.seed}_{run_ts}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_ds, bp["batch_size"], shuffle=True,  num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader   = make_loader(val_ds,   bp["batch_size"], shuffle=False, num_workers=0, seed=args.seed, device=DEVICE)
    test_loader  = make_loader(test_ds,  bp["batch_size"], shuffle=False, num_workers=0, seed=args.seed, device=DEVICE)

    model = ConvGraphLSTM(
        in_ch=16, k_neighbors=args.k_neighbors,
        base_c=bp["base_c"], emb_c=bp["emb_c"], base_n=bp["base_n"], emb_n=bp["emb_n"],
        hidden_g=bp["hidden_g"], n_sage_layers=bp["n_sage_layers"], concat_agg=bp["concat_agg"],
        hidden_t=bp["hidden_t"], n_lstm_layers=bp["n_lstm_layers"], dropout=bp["dropout"],
    ).to(DEVICE)
    out = train_one_model(
        model=model, train_loader=train_loader, val_loader=val_loader, normalizer=normalizer,
        lr=bp["lr"], weight_decay=bp["weight_decay"], l1_reg=bp["l1_reg"],
        use_amp=(DEVICE == "cuda"), epochs=(3 if args.debug else 40), patience=8,
        day_threshold=args.day_threshold, device=DEVICE, fusion=True,
    )
    best_model = out.get("model", model)
    final_test = eval_model(best_model, test_loader, normalizer, args.day_threshold, DEVICE, fusion=True)
    final_test["skill_vs_persistence"]     = skill_score(final_test["rmse"],     baseline_test["rmse"])
    final_test["skill_day_vs_persistence"] = skill_score(final_test["rmse_day"], baseline_test["rmse_day"])
    print(f"\n=== TEST === RMSE_day={final_test['rmse_day']:.2f} "
          f"skill_day={final_test['skill_day_vs_persistence']:.3f}")

    ckpt = RUN_DIR / "best_model.pt"
    torch.save({"model_state": best_model.state_dict()}, ckpt)

    summary = {
        "model": "convgraph_lstm",
        "site": args.site,
        "seed": args.seed,
        "patch": args.patch,
        "k_neighbors": args.k_neighbors,
        "temporal": {"horizon_hours": args.hours_ahead, "history_steps": L},
        "baselines": {"persistence_val": baseline_val, "persistence_test": baseline_test},
        "optuna": {"n_trials": args.n_trials, "best_value": best.value, "best_params": bp},
        "best_model": {
            "arch": "SmallResNet(center) + per-neighbour ResNet -> GraphSAGE(K nodes) -> concat -> LSTM",
            "best_ckpt_path": str(ckpt.relative_to(PROJECT_ROOT)),
            "final_test": final_test,
            "n_params": int(sum(p.numel() for p in best_model.parameters())),
        },
    }
    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {RUN_DIR}")


if __name__ == "__main__":
    main()

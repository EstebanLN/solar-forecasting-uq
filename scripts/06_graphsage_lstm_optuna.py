#!/usr/bin/env python
"""
06_graphsage_lstm_optuna.py — Optuna HPO for the improved GraphSAGE+LSTM.

Extended search space vs the original notebook:
  - n_sage_layers [2, 3, 4]
  - hidden_g up to 192 (was 128)
  - input_bn [True, False]
  - concat_agg [True, False]  — standard SAGEConv vs additive
  - n_lstm_layers [1, 2]
  - LR range extended downward to 5e-5

Objective: minimise val RMSE_day.

Usage (from project root):
    python scripts/06_graphsage_lstm_optuna.py
    python scripts/06_graphsage_lstm_optuna.py --site elpaso --seed 7 --n_trials 50
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    p = argparse.ArgumentParser(description="GraphSAGE+LSTM Optuna tuning")
    p.add_argument("--site",          default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead",   type=int,   default=6, choices=[1, 3, 6])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patch",         type=int,   default=16)
    p.add_argument("--n_trials",      type=int,   default=30)
    p.add_argument("--debug",         action="store_true")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--day_threshold", type=float, default=20.0)
    return p.parse_args()


# ------------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------------
def make_objective(
    train_ds, val_ds,
    normalizer: TargetNormalizer,
    edge_index: torch.Tensor,
    device: str,
    seed: int,
    day_threshold: float,
    use_amp: bool,
):
    def objective(trial: optuna.Trial) -> float:
        hidden_g      = trial.suggest_categorical("hidden_g",      [64, 96, 128, 192])
        n_sage_layers = trial.suggest_categorical("n_sage_layers", [2, 3, 4])
        hidden_t      = trial.suggest_categorical("hidden_t",      [64, 96, 128])
        n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
        input_bn      = trial.suggest_categorical("input_bn",      [True, False])
        concat_agg    = trial.suggest_categorical("concat_agg",    [True, False])
        dropout_head  = trial.suggest_categorical("dropout_head",  [0.0, 0.1, 0.2])
        lr            = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size    = trial.suggest_categorical("batch_size", [8, 16, 32])

        train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=4, seed=seed, device=device)
        val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=0, seed=seed, device=device)

        model = GraphSAGE_LSTM(
            in_dim=16,
            hidden_g=hidden_g,
            n_sage_layers=n_sage_layers,
            hidden_t=hidden_t,
            n_lstm_layers=n_lstm_layers,
            dropout_head=dropout_head,
            input_bn=input_bn,
            concat_agg=concat_agg,
            edge_index=edge_index,
        ).to(device)

        out = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalizer=normalizer,
            lr=lr,
            weight_decay=weight_decay,
            use_amp=use_amp,
            epochs=20,
            patience=6,
            day_threshold=day_threshold,
            device=device,
        )

        vm = out["final_val"]
        trial.set_user_attr("best_epoch",          out["best_epoch"])
        trial.set_user_attr("val_rmse",            float(vm["rmse"]))
        trial.set_user_attr("val_rmse_day",        float(vm["rmse_day"]))
        trial.set_user_attr("val_mae",             float(vm["mae"]))
        trial.set_user_attr("val_mae_day",         float(vm["mae_day"]))
        trial.set_user_attr("train_seconds_total", float(out["train_seconds_total"]))
        trial.set_user_attr("n_params",            int(sum(p.numel() for p in model.parameters())))

        return float(vm["rmse_day"])

    return objective


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Directories
    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR   = PROJECT_ROOT / "data" / "ground_aligned"
    RUNS_ROOT    = PROJECT_ROOT / "runs" / "graphsage_lstm_optuna"
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    SITE_DIR = DATASET_ROOT / args.site / f"h{args.hours_ahead}"
    assert SITE_DIR.exists(), f"Missing dataset dir: {SITE_DIR}"

    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / args.site / f"P{args.patch}"
    assert PATCHES_ROOT.exists(), f"Missing patch store: {PATCHES_ROOT}"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = (DEVICE == "cuda")
    print(f"DEVICE={DEVICE} | site={args.site} | hours_ahead={args.hours_ahead} | seed={args.seed} | n_trials={args.n_trials}")

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

    if args.debug:
        train_man = train_man.sample(n=4000, random_state=args.seed).reset_index(drop=True)
        val_man   = val_man.sample(n=1200,  random_state=args.seed).reset_index(drop=True)
        test_man  = test_man.sample(n=1200, random_state=args.seed).reset_index(drop=True)

    seed_everything(args.seed)

    # Ground truth & persistence baseline
    ground_path = GROUND_DIR / f"ground_10min_utc_{args.site}.parquet"
    ground = pd.read_parquet(ground_path)
    baseline_train = eval_persistence(train_man, ground, args.day_threshold)
    baseline_val   = eval_persistence(val_man,   ground, args.day_threshold)
    baseline_test  = eval_persistence(test_man,  ground, args.day_threshold)
    print(f"Persistence test: RMSE={baseline_test['rmse']:.1f}  RMSE_day={baseline_test['rmse_day']:.1f}")

    # Normalizer, graph structure, datasets
    y_train_arr = train_man["y"].astype(float).to_numpy()
    normalizer  = TargetNormalizer.from_train(y_train_arr)

    N_NODES    = args.patch * args.patch
    edge_index = build_edge_index_8n(args.patch)
    print(f"Graph: {N_NODES} nodes, {edge_index.shape[1]} edges")

    train_ds = GraphSeqDataset(train_man, PATCHES_ROOT, normalizer)
    val_ds   = GraphSeqDataset(val_man,   PATCHES_ROOT, normalizer)
    test_ds  = GraphSeqDataset(test_man,  PATCHES_ROOT, normalizer)

    # Optuna study
    STUDY_NAME = f"graphsage_lstm_{args.site}_P{args.patch}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(f"\nStarting Optuna ({args.n_trials} trials) ...")
    study.optimize(
        make_objective(train_ds, val_ds, normalizer, edge_index, DEVICE, args.seed, args.day_threshold, USE_AMP),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: val_rmse_day={best.value:.2f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Trials dataframe
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": str(t.state), "objective": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("objective", ascending=True)
    print("\nTop-5 trials:")
    print(trials_df.head(5).to_string(index=False))

    # Final retraining with best params
    print("\nRetraining with best params (epochs=30, patience=8) ...")
    bp = best.params

    best_batch = bp["batch_size"]
    train_loader = make_loader(train_ds, best_batch, shuffle=True,  num_workers=args.num_workers, seed=args.seed, device=DEVICE)
    val_loader   = make_loader(val_ds,   best_batch, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)
    test_loader  = make_loader(test_ds,  best_batch, shuffle=False, num_workers=0,               seed=args.seed, device=DEVICE)

    best_model = GraphSAGE_LSTM(
        in_dim=16,
        hidden_g=bp["hidden_g"],
        n_sage_layers=bp["n_sage_layers"],
        hidden_t=bp["hidden_t"],
        n_lstm_layers=bp["n_lstm_layers"],
        dropout_head=bp["dropout_head"],
        input_bn=bp["input_bn"],
        concat_agg=bp["concat_agg"],
        edge_index=edge_index,
    ).to(DEVICE)

    out = train_one_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        lr=bp["lr"],
        weight_decay=bp["weight_decay"],
        use_amp=USE_AMP,
        epochs=30,
        patience=8,
        day_threshold=args.day_threshold,
        device=DEVICE,
    )

    final_val  = out["final_val"]
    final_test = eval_model(best_model, test_loader, normalizer, args.day_threshold, DEVICE)

    final_val["skill_vs_persistence"]      = skill_score(final_val["rmse"],      baseline_val["rmse"])
    final_val["skill_day_vs_persistence"]  = skill_score(final_val["rmse_day"],  baseline_val["rmse_day"])
    final_test["skill_vs_persistence"]     = skill_score(final_test["rmse"],     baseline_test["rmse"])
    final_test["skill_day_vs_persistence"] = skill_score(final_test["rmse_day"], baseline_test["rmse_day"])

    print("=== Final evaluation (best Optuna params) ===")
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
            "model_state":       best_model.state_dict(),
            "best_val_rmse_day": out["best_val_rmse_day"],
            "meta": {
                "arch": "GraphSAGE_LSTM", "site": args.site, "patch": args.patch,
                "L": L, "H": H, "seed": args.seed,
                "y_mean_train": normalizer.mean, "y_std_train": normalizer.std,
            },
        },
        BEST_PATH,
    )

    n_params = int(sum(p.numel() for p in best_model.parameters()))

    summary = {
        "run_name":    run_name,
        "study_name":  STUDY_NAME,
        "site":        args.site,
        "device":      DEVICE,
        "seed":        args.seed,
        "debug":       args.debug,
        "day_threshold": args.day_threshold,
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
        "optuna": {
            "n_trials":                args.n_trials,
            "best_trial_number":       int(best.number),
            "best_value_val_rmse_day": float(best.value),
            "best_params":             best.params,
        },
        "best_model": {
            "arch":                "GraphSAGE_LSTM (improved)",
            "best_epoch":          out["best_epoch"],
            "train_seconds_total": out["train_seconds_total"],
            "n_params":            n_params,
            "best_ckpt_path":      str(BEST_PATH),
            "final_val":           final_val,
            "final_test":          final_test,
        },
    }

    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    trials_df.to_csv(RUN_DIR / "trials.csv", index=False)
    print(f"Saved to: {RUN_DIR}")


if __name__ == "__main__":
    main()

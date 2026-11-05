#!/usr/bin/env python
"""
06r_refit_draws.py — extra final-retraining draws for an already-tuned cell.

Why this exists
---------------
The Optuna scripts end with ONE final retraining of the best configuration.
That single draw is compared against `study.best_value`, which is the MINIMUM
over ~75 trials. A single draw versus a 75-draw minimum has a positive expected
gap, so the refit almost always looks worse than the search: over the 110 runs
that store both numbers the median gap is +1.42%. That part is the winner's
curse and nothing is wrong with it.

The tail is the problem. 23 of those 110 runs exceed +5% or stop at epoch <= 2,
and 13 of them are in `resnet_lstm_optuna_v2`, the family the paper uses as the
baseline at both sites. The worst, elpaso h3 seed 1, missed the search value by
+38.3% and reports the lowest test skill of its cell (0.456 against 0.576,
0.595 and 0.487). A refit that stops at epoch 1 is not evidence about the
configuration: the search already trained that same configuration successfully.
It is one unlucky trajectory.

So: draw the refit more than once and keep the best ON VALIDATION. That is the
criterion the search itself optimises and the one `solar_uq.run_select` already
applies when a cell holds several runs, so no extra selection logic is needed
downstream and the test split is never consulted. Every cell gets the same
treatment, which is what separates this from re-running the runs that happen to
look bad.

What it does NOT do
-------------------
It never re-runs the search and never touches the source run. Each draw is a
new run directory in the same family, so `run_select` picks whichever has the
lowest val RMSE_day — possibly still the original. Nothing is overwritten and
nothing is deleted.

Usage (from project root):
    .venv/bin/python scripts/06r_refit_draws.py \
        --family resnet_lstm_optuna_v2 --site uniandes --hours_ahead 1 --seed 42 --draws 2

    # every cell of a family, 2 extra draws each
    .venv/bin/python scripts/06r_refit_draws.py --family graphsage_lstm_optuna_v2 --all --draws 2
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
    TargetNormalizer,
    PatchSeqDataset,
    GraphSeqDataset,
    make_loader,
    preload_patch_cache,
    read_history_steps_from_manifest,
)
from solar_uq.loaders import FusionPatchSeqDataset, FusionGraphSeqDataset
from solar_uq.metrics import eval_persistence, skill_score
from solar_uq.models.resnet_lstm import ResNetLSTM
from solar_uq.models.fusion.fusion_resnet_lstm import FusionResNetLSTM
from solar_uq.models.graphsage_lstm import GraphSAGE_LSTM, build_weighted_knn_edge_index
from solar_uq.models.fusion.fusion_graphsage_lstm import FusionGraphSAGE_LSTM
from solar_uq.train import seed_everything, train_one_model, eval_model
from solar_uq.run_select import scan_cell_runs, pick_best

# Mirrors the registry in 12_variance_net.py / 08_sgld.py. "arch" selects the
# model and dataset; "fusion" selects the surface-fused variant.
FAMILIES = {
    "resnet_lstm_optuna":       dict(arch="resnet",    fusion=False),
    "resnet_lstm_optuna_v2":    dict(arch="resnet",    fusion=False),
    "graphsage_lstm_optuna":    dict(arch="graphsage", fusion=False),
    "graphsage_lstm_optuna_v2": dict(arch="graphsage", fusion=False),
    "fusion_resnet_lstm":       dict(arch="resnet",    fusion=True),
    "fusion_graphsage_lstm":    dict(arch="graphsage", fusion=True),
}

# Same budget as the refit inside the Optuna scripts, so a draw here is
# comparable to the draw stored in the source run.
REFIT_EPOCHS = 30
REFIT_PATIENCE = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extra final-retraining draws for a tuned cell")
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--site", default=None, choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=None, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all", action="store_true",
                   help="every (site, horizon, seed) cell the family already has")
    p.add_argument("--draws", type=int, default=2, help="extra draws per cell")
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--day_threshold", type=float, default=20.0)
    p.add_argument("--min_gap_pct", type=float, default=None,
                   help="with --all, skip cells whose stored refit is already "
                        "within this %% of the search value")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def source_runs(family: str) -> list[dict]:
    """One entry per (site, horizon, seed) cell, pointing at its canonical run.

    Reads every summary.json rather than using run_select directly, because
    run_select requires the checkpoint to be on disk and several cells have lost
    theirs — those are exactly the cells that need a draw most.
    """
    root = PROJECT_ROOT / "runs" / family
    by_cell: dict[tuple, dict] = {}
    if not root.exists():
        return []
    for d in sorted(root.iterdir()):
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        try:
            s = json.load(open(sj, encoding="utf-8"))
        except Exception:
            continue
        bm = s.get("best_model") or {}
        if not isinstance(bm, dict):
            continue
        opt = s.get("optuna") or {}
        bp = opt.get("best_params")
        fv = bm.get("final_val") or {}
        if not bp or fv.get("rmse_day") is None:
            continue
        h = (s.get("temporal") or {}).get("horizon_hours")
        if h is None:
            continue
        cell = (s.get("site"), int(round(float(h))), int(s.get("seed")))
        rec = dict(run_dir=d, site=cell[0], hours=cell[1], seed=cell[2],
                   best_params=bp, val_rmse_day=float(fv["rmse_day"]),
                   search_best=opt.get("best_value_val_rmse_day"),
                   best_epoch=bm.get("best_epoch"), summary=s)
        prev = by_cell.get(cell)
        if prev is None or rec["val_rmse_day"] < prev["val_rmse_day"]:
            by_cell[cell] = rec
    return [by_cell[k] for k in sorted(by_cell, key=lambda c: (c[0], c[1], c[2]))]


def build_model(arch: str, fusion: bool, bp: dict, d_tab: int, patch: int, device: str):
    """Exactly the construction the Optuna scripts use for their own refit."""
    if arch == "resnet":
        if fusion:
            m = FusionResNetLSTM(
                in_ch=16, base=bp["base"], emb_dim=bp["emb_dim"], d_tab=d_tab,
                tab_hidden=bp.get("tab_hidden", 64), hidden_t=bp["hidden_t"],
                dropout=bp["dropout"], n_lstm_layers=bp.get("n_lstm_layers", 1),
            )
        else:
            m = ResNetLSTM(
                in_ch=16, base=bp["base"], emb_dim=bp["emb_dim"],
                hidden_t=bp["hidden_t"], dropout=bp["dropout"],
                n_lstm_layers=bp.get("n_lstm_layers", 1),
            )
        return m.to(device), None, None

    # graphsage: the v2 studies tune k_neighbors on a weighted k-NN graph. The
    # v1 studies used a fixed 8-neighbour graph and store no k_neighbors, so
    # refitting them would need build_edge_index_8n and a different model
    # signature; refuse rather than silently refit a different architecture.
    if "k_neighbors" not in bp:
        raise SystemExit(
            "this run stores no k_neighbors, so it comes from the fixed-graph v1 "
            "study; refitting it here would change the graph construction. Refit "
            "only v2 cells with this script."
        )
    edge_index, edge_weight = build_weighted_knn_edge_index(patch, bp["k_neighbors"])
    common = dict(
        in_dim=16, hidden_g=bp["hidden_g"], n_sage_layers=bp["n_sage_layers"],
        hidden_t=bp["hidden_t"], n_lstm_layers=bp["n_lstm_layers"],
        dropout_head=bp["dropout_head"], input_bn=bp["input_bn"],
        concat_agg=bp["concat_agg"], edge_index=edge_index, edge_weight=edge_weight,
    )
    if fusion:
        m = FusionGraphSAGE_LSTM(d_tab=d_tab, tab_hidden=bp.get("tab_hidden", 64), **common)
    else:
        m = GraphSAGE_LSTM(**common)
    return m.to(device), edge_index, edge_weight


def refit_cell(args, cfg: dict, rec: dict) -> None:
    site, hours, seed = rec["site"], rec["hours"], rec["seed"]
    arch, fusion = cfg["arch"], cfg["fusion"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "manifest_v1"
    GROUND_DIR = PROJECT_ROOT / "data" / "ground_aligned"
    RUNS_ROOT = PROJECT_ROOT / "runs" / args.family
    SITE_DIR = DATASET_ROOT / site / f"h{hours}"
    PATCHES_ROOT = PROJECT_ROOT / "data" / "patches_v1" / site / f"P{args.patch}"
    assert SITE_DIR.exists(), f"missing dataset dir: {SITE_DIR}"
    assert PATCHES_ROOT.exists(), f"missing patch store: {PATCHES_ROOT}"

    bp = rec["best_params"]
    gap = (100.0 * (rec["val_rmse_day"] - rec["search_best"]) / rec["search_best"]
           if rec["search_best"] else float("nan"))
    print(f"\n{'='*78}\n{args.family} | {site} h{hours} seed{seed} | device={device}")
    print(f"  origen: {rec['run_dir'].name}")
    print(f"  search={rec['search_best']:.2f}  refit guardado={rec['val_rmse_day']:.2f} "
          f"({gap:+.2f}%, best_epoch={rec['best_epoch']})")

    preload_patch_cache(PATCHES_ROOT)
    with open(SITE_DIR / "dataset_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    FREQ_MIN = int(meta["freq_min"])
    H = int(meta["horizon_steps"])
    GRID_SIZE = int(meta["grid_size"])

    train_man = pd.read_parquet(SITE_DIR / "manifest_train.parquet")
    val_man = pd.read_parquet(SITE_DIR / "manifest_val.parquet")
    test_man = pd.read_parquet(SITE_DIR / "manifest_test.parquet")
    L = read_history_steps_from_manifest(train_man)

    seed_everything(seed)
    ground_path = GROUND_DIR / f"ground_10min_utc_{site}.parquet"
    ground = pd.read_parquet(ground_path)
    baseline_train = eval_persistence(train_man, ground, args.day_threshold)
    baseline_val = eval_persistence(val_man, ground, args.day_threshold)
    baseline_test = eval_persistence(test_man, ground, args.day_threshold)

    y_train_arr = train_man["y"].astype(float).to_numpy()
    normalizer = TargetNormalizer.from_train(y_train_arr)

    d_tab = 0
    surface_path = None
    if fusion:
        surface_path = GROUND_DIR / f"ground_10min_utc_{site}.parquet"
        tab_stats = FusionPatchSeqDataset.compute_tab_stats(
            train_man, PATCHES_ROOT, normalizer, surface_path, n_samples=5_000, seed=seed)
        d_tab = tab_stats["d_tab"]
        DS = FusionGraphSeqDataset if arch == "graphsage" else FusionPatchSeqDataset
        mk = lambda man: DS(man, PATCHES_ROOT, normalizer, surface_path, tab_stats=tab_stats)
    else:
        DS = GraphSeqDataset if arch == "graphsage" else PatchSeqDataset
        mk = lambda man: DS(man, PATCHES_ROOT, normalizer)
    train_ds, val_ds, test_ds = mk(train_man), mk(val_man), mk(test_man)

    bs = bp["batch_size"]
    train_loader = make_loader(train_ds, bs, shuffle=True, num_workers=args.num_workers,
                               seed=seed, device=device)
    val_loader = make_loader(val_ds, bs, shuffle=False, num_workers=0, seed=seed, device=device)
    test_loader = make_loader(test_ds, bs, shuffle=False, num_workers=0, seed=seed, device=device)

    for draw in range(1, args.draws + 1):
        # The refit inside the Optuna script inherits whatever RNG state the
        # 75-trial search left behind, which is why it cannot be reproduced and
        # why re-drawing is a fresh sample. Offset the seed per draw so each
        # draw IS reproducible, and record the offset.
        rng_seed = seed + 1000 * draw
        seed_everything(rng_seed)
        model, edge_index, _ = build_model(arch, fusion, bp, d_tab, args.patch, device)

        out = train_one_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            normalizer=normalizer, lr=bp["lr"], weight_decay=bp["weight_decay"],
            l1_reg=bp.get("l1_reg", 0.0), use_amp=use_amp,
            epochs=REFIT_EPOCHS, patience=REFIT_PATIENCE,
            day_threshold=args.day_threshold, device=device, fusion=fusion,
        )
        final_val = out["final_val"]
        final_test = eval_model(model, test_loader, normalizer, args.day_threshold,
                                device, fusion=fusion)
        for d_, base in ((final_val, baseline_val), (final_test, baseline_test)):
            d_["skill_vs_persistence"] = skill_score(d_["rmse"], base["rmse"])
            d_["skill_day_vs_persistence"] = skill_score(d_["rmse_day"], base["rmse_day"])

        new_gap = (100.0 * (final_val["rmse_day"] - rec["search_best"]) / rec["search_best"]
                   if rec["search_best"] else float("nan"))
        verdict = "MEJORA" if final_val["rmse_day"] < rec["val_rmse_day"] else "no mejora"
        print(f"  draw {draw}: val_rmse_day={final_val['rmse_day']:.2f} ({new_gap:+.2f}% "
              f"vs search) ep={out['best_epoch']} -> {verdict} | "
              f"skill_day_test={final_test['skill_day_vs_persistence']:.3f}")

        run_ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
        run_name = f"{site}_H{H}_L{L}_P{args.patch}_seed{seed}_{run_ts}"
        RUN_DIR = RUNS_ROOT / run_name
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        BEST_PATH = RUN_DIR / "best_model.pt"

        src_summary = rec["summary"]
        arch_hparams = dict(bp)
        arch_hparams.pop("batch_size", None)
        if fusion:
            arch_hparams["d_tab"] = d_tab
        torch.save(
            {
                "epoch": out["best_epoch"],
                "model_state": model.state_dict(),
                "best_val_rmse_day": out["best_val_rmse_day"],
                "meta": {
                    "arch": type(model).__name__,
                    "fusion": fusion,
                    "arch_hparams": arch_hparams,
                    "site": site, "patch": args.patch,
                    "L": L, "H": H, "seed": seed,
                    "y_mean_train": normalizer.mean, "y_std_train": normalizer.std,
                },
            },
            BEST_PATH,
        )

        summary = {
            "run_name": run_name,
            "study_name": src_summary.get("study_name"),
            "site": site,
            "device": device,
            "seed": seed,
            "debug": False,
            "day_threshold": args.day_threshold,
            "data_paths": {
                "site_dir": str(SITE_DIR),
                "patches_root": str(PATCHES_ROOT),
                "ground_path": str(ground_path),
                "run_dir": str(RUN_DIR),
            },
            "temporal": {
                "freq_min": FREQ_MIN,
                "history_steps_L": L,
                "horizon_steps_H": H,
                "history_hours": L * FREQ_MIN / 60.0,
                "horizon_hours": H * FREQ_MIN / 60.0,
            },
            "spatial": dict(src_summary.get("spatial") or {}),
            "target_norm": {
                "y_mean_train": normalizer.mean,
                "y_std_train": normalizer.std,
                "y_percentiles_train": np.percentile(y_train_arr, [0, 50, 90, 95, 99]).tolist(),
            },
            "baselines": {
                "persistence_train": baseline_train,
                "persistence_val": baseline_val,
                "persistence_test": baseline_test,
            },
            # Carried over verbatim: this run did NOT search, so the search
            # provenance stays attributable to the source study.
            "optuna": dict(src_summary.get("optuna") or {}),
            "refit": {
                "source_run": rec["run_dir"].name,
                "draw": draw,
                "rng_seed": rng_seed,
                "epochs": REFIT_EPOCHS,
                "patience": REFIT_PATIENCE,
                "source_val_rmse_day": rec["val_rmse_day"],
                "gap_pct_vs_search": new_gap,
                "script": "06r_refit_draws.py",
            },
            "best_model": {
                "arch": (src_summary.get("best_model") or {}).get("arch"),
                "best_epoch": out["best_epoch"],
                "train_seconds_total": out["train_seconds_total"],
                "n_params": int(sum(p.numel() for p in model.parameters())),
                "best_ckpt_path": str(BEST_PATH.relative_to(PROJECT_ROOT)),
                "final_val": final_val,
                "final_test": final_test,
            },
        }
        with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        src_trials = rec["run_dir"] / "trials.csv"
        if src_trials.is_file():
            # Downstream tooling reads trials.csv; keep the search's own trials
            # so the HPO record stays with every run of the cell.
            (RUN_DIR / "trials.csv").write_bytes(src_trials.read_bytes())
        print(f"    guardado en {RUN_DIR.name}")


def main() -> None:
    args = parse_args()
    cfg = FAMILIES[args.family]
    cells = source_runs(args.family)
    if not cells:
        raise SystemExit(f"no hay corridas con best_params en runs/{args.family}")

    if not args.all:
        if args.site is None or args.hours_ahead is None or args.seed is None:
            raise SystemExit("da --site, --hours_ahead y --seed, o usa --all")
        cells = [c for c in cells
                 if c["site"] == args.site and c["hours"] == args.hours_ahead
                 and c["seed"] == args.seed]
        if not cells:
            raise SystemExit("esa celda no existe en la familia")
    elif args.min_gap_pct is not None:
        keep = []
        for c in cells:
            if not c["search_best"]:
                continue
            g = 100.0 * (c["val_rmse_day"] - c["search_best"]) / c["search_best"]
            if g >= args.min_gap_pct:
                keep.append(c)
        cells = keep

    print(f"familia={args.family} | celdas={len(cells)} | draws={args.draws} "
          f"=> {len(cells)*args.draws} entrenamientos")
    for c in cells:
        g = (100.0 * (c["val_rmse_day"] - c["search_best"]) / c["search_best"]
             if c["search_best"] else float("nan"))
        print(f"  {c['site']:9s} h{c['hours']} s{c['seed']:<3d} gap={g:+6.2f}% "
              f"ep={c['best_epoch']}")
    if args.dry_run:
        print("\n(dry run) nada ejecutado")
        return
    for c in cells:
        refit_cell(args, cfg, c)


if __name__ == "__main__":
    main()

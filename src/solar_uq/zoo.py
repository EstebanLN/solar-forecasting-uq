"""One registry for every architecture the uncertainty tools can run on.

Why this exists: the UQ scripts (08_sgld, 08b, 08d, 08f, 12) each carried their
own registry, and it knew only resnet, graphsage and mlp -- the satellite-only
families. The models the project now reports are the FlatMLP, the fusion
models and ConvGraph-LSTM, and none of the last two could be built by any UQ
tool. New tools (13_swag.py) go through this module so that adding an
architecture is one entry, not five edits.

Two-input models (fusion: satellite + tabular; convgraph: centre + neighbours)
yield 3-tuples from their datasets; train.eval_model / collect_predictions
handle them with ``fusion=True``, exposed here as ``two_input``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from solar_uq.data import (
    GraphSeqDataset, NeighborPoolDataset, PatchSeqDataset, TargetNormalizer,
    preload_patch_cache, preload_pool_cache, read_history_steps_from_manifest,
)
from solar_uq.loaders import FusionGraphSeqDataset, FusionPatchSeqDataset
from solar_uq.metrics import eval_persistence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHS = {
    "mlp":              dict(runs="mlp_optuna",               two_input=False),
    "resnet":           dict(runs="resnet_lstm_optuna_v2",    two_input=False),
    "graphsage":        dict(runs="graphsage_lstm_optuna_v2", two_input=False),
    "convgraph":        dict(runs="convgraph_lstm",           two_input=True),
    "fusion_resnet":    dict(runs="fusion_resnet_lstm",       two_input=True),
    "fusion_graphsage": dict(runs="fusion_graphsage_lstm",    two_input=True),
}


def _val_key(s: dict) -> float:
    """Validation RMSE_day used for canonical selection (lowest wins)."""
    bm = s.get("best_model") if isinstance(s.get("best_model"), dict) else {}
    fv = bm.get("final_val") or s.get("final_val") or {}
    if isinstance(fv.get("rmse_day"), (int, float)):
        return float(fv["rmse_day"])
    o = s.get("optuna") or {}
    for k in ("best_value_val_rmse_day", "best_val_rmse_day", "best_value"):
        if isinstance(o.get(k), (int, float)):
            return float(o[k])
    return float("inf")


def canonical_run(arch: str, site: str, hours: int, seed: int) -> dict:
    """Canonical tuned run of a cell: best_params, checkpoint (or None), summary.

    Same rule as solar_uq.run_select (lowest validation RMSE_day), but it does
    not require a checkpoint: several cells lost theirs, and a tool that can
    retrain from best_params should still be able to use them.
    """
    root = PROJECT_ROOT / "runs" / ARCHS[arch]["runs"]
    best = None
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.name.startswith("_") or not (d / "summary.json").is_file():
            continue
        try:
            s = json.load(open(d / "summary.json", encoding="utf-8"))
        except Exception:
            continue
        h = (s.get("temporal") or {}).get("horizon_hours")
        bp = (s.get("optuna") or {}).get("best_params")
        if s.get("site") != site or s.get("seed") != seed or h is None or not bp:
            continue
        if int(round(float(h))) != hours:
            continue
        v = _val_key(s)
        if best is None or v < best["val"]:
            ck = d / "best_model.pt"
            best = dict(run_dir=d, best_params=bp, val=v, summary=s,
                        ckpt=ck if ck.is_file() else None)
    if best is None:
        raise SystemExit(f"no hay corrida con best_params para {arch} {site} h{hours} seed{seed}")
    return best


def build_data(arch: str, site: str, hours: int, seed: int, patch: int = 16,
               k_neighbors: int = 8, day_threshold: float = 20.0) -> dict:
    """Datasets, normaliser and persistence baselines, built as the tuning scripts do."""
    site_dir = PROJECT_ROOT / "data" / "datasets" / "manifest_v1" / site / f"h{hours}"
    patches = PROJECT_ROOT / "data" / "patches_v1" / site / f"P{patch}"
    ground_path = PROJECT_ROOT / "data" / "ground_aligned" / f"ground_10min_utc_{site}.parquet"
    preload_patch_cache(patches)
    mans = {k: pd.read_parquet(site_dir / f"manifest_{k}.parquet") for k in ("train", "val", "test")}
    L = read_history_steps_from_manifest(mans["train"])
    normalizer = TargetNormalizer.from_train(mans["train"]["y"].astype(float).to_numpy())
    ground = pd.read_parquet(ground_path)
    d_tab = 0
    if arch in ("fusion_resnet", "fusion_graphsage"):
        tab_stats = FusionPatchSeqDataset.compute_tab_stats(
            mans["train"], patches, normalizer, ground_path, n_samples=5_000, seed=seed)
        d_tab = tab_stats["d_tab"]
        DS = FusionGraphSeqDataset if arch == "fusion_graphsage" else FusionPatchSeqDataset
        mk = lambda m: DS(m, patches, normalizer, ground_path, tab_stats=tab_stats)
    elif arch == "convgraph":
        pool = PROJECT_ROOT / "data" / "neighbor_pool_v1" / site
        mk = lambda m: NeighborPoolDataset(m, patches, pool, normalizer, k_neighbors=k_neighbors)
    elif arch == "graphsage":
        mk = lambda m: GraphSeqDataset(m, patches, normalizer)
    else:
        mk = lambda m: PatchSeqDataset(m, patches, normalizer)
    if arch == "mlp":
        from solar_uq.data import MeanPatchCache
        _mk = mk
        mk = lambda m: MeanPatchCache(_mk(m))
    return dict(
        train=mk(mans["train"]), val=mk(mans["val"]), test=mk(mans["test"]),
        normalizer=normalizer, L=L, d_tab=d_tab,
        baseline_val=eval_persistence(mans["val"], ground, day_threshold),
        baseline_test=eval_persistence(mans["test"], ground, day_threshold),
    )


def build_model(arch: str, bp: dict, patch: int, L: int, d_tab: int = 0,
                k_neighbors: int = 8) -> torch.nn.Module:
    """Instantiate an architecture from its Optuna best_params (on CPU)."""
    if arch == "mlp":
        from solar_uq.models.mlp import FlatMLP
        return FlatMLP(L=L, C=16, n_layers=bp["n_layers"], hidden_dim=bp["hidden_dim"],
                       dropout=bp.get("dropout", 0.0))
    if arch in ("resnet", "fusion_resnet"):
        kw = dict(in_ch=16, base=bp["base"], emb_dim=bp["emb_dim"], hidden_t=bp["hidden_t"],
                  dropout=bp.get("dropout", 0.0), n_lstm_layers=bp.get("n_lstm_layers", 1))
        if arch == "resnet":
            from solar_uq.models.resnet_lstm import ResNetLSTM
            return ResNetLSTM(**kw)
        from solar_uq.models.fusion.fusion_resnet_lstm import FusionResNetLSTM
        return FusionResNetLSTM(d_tab=d_tab, tab_hidden=bp.get("tab_hidden", 64), **kw)
    if arch in ("graphsage", "fusion_graphsage"):
        from solar_uq.models.graphsage_lstm import (
            GraphSAGE_LSTM, build_edge_index_8n, build_weighted_knn_edge_index)
        if "k_neighbors" in bp:
            ei, ew = build_weighted_knn_edge_index(patch, bp["k_neighbors"])
        else:
            ei, ew = build_edge_index_8n(patch), None
        kw = dict(in_dim=16, hidden_g=bp["hidden_g"], n_sage_layers=bp["n_sage_layers"],
                  hidden_t=bp["hidden_t"], n_lstm_layers=bp.get("n_lstm_layers", 1),
                  dropout_head=bp.get("dropout_head", 0.0), input_bn=bp.get("input_bn", False),
                  concat_agg=bp.get("concat_agg", True), edge_index=ei, edge_weight=ew)
        if arch == "graphsage":
            return GraphSAGE_LSTM(**kw)
        from solar_uq.models.fusion.fusion_graphsage_lstm import FusionGraphSAGE_LSTM
        return FusionGraphSAGE_LSTM(d_tab=d_tab, tab_hidden=bp.get("tab_hidden", 64), **kw)
    if arch == "convgraph":
        from solar_uq.models.convgraph_lstm import ConvGraphLSTM
        return ConvGraphLSTM(
            in_ch=16, k_neighbors=k_neighbors, base_c=bp["base_c"], emb_c=bp["emb_c"],
            base_n=bp["base_n"], emb_n=bp["emb_n"], hidden_g=bp["hidden_g"],
            n_sage_layers=bp["n_sage_layers"], concat_agg=bp["concat_agg"],
            hidden_t=bp["hidden_t"], n_lstm_layers=bp["n_lstm_layers"], dropout=bp["dropout"])
    raise ValueError(f"arquitectura desconocida: {arch}")


def load_state(model: torch.nn.Module, ckpt: Optional[Path]) -> bool:
    """Load a tuning-script checkpoint; returns False if there is none."""
    if ckpt is None:
        return False
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj)
    return True

"""Forward-pass smoke tests for the fusion architecture.

Run from project root:
    .venv/bin/python -m pytest tests/test_fusion_forward.py -v

All tests use synthetic random tensors — no real data or GPU required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solar_uq.models.fusion.tabular_projector import TabularProjector
from solar_uq.models.fusion.fusion_resnet_lstm import FusionResNetLSTM
from solar_uq.models.fusion.fusion_graphsage_lstm import FusionGraphSAGE_LSTM
from solar_uq.models.graphsage_lstm import build_weighted_knn_edge_index


# ─────────────────────────────────────────────────────────────────────────────
# TabularProjector
# ─────────────────────────────────────────────────────────────────────────────

def test_tab_projector_residual():
    """Output shape must match d_emb exactly (needed for residual addition)."""
    B, L, d_tab, d_emb = 4, 6, 9, 128
    proj = TabularProjector(d_tab=d_tab, d_emb=d_emb, hidden=64)
    tab  = torch.randn(B, L, d_tab)
    out  = proj(tab)
    assert out.shape == (B, L, d_emb), f"Expected ({B},{L},{d_emb}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in TabularProjector output"


# ─────────────────────────────────────────────────────────────────────────────
# FusionResNetLSTM
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_resnet_forward():
    """End-to-end forward pass: output is (B,) with no NaNs."""
    B, L, C, P = 2, 4, 16, 8
    d_tab = 9
    model = FusionResNetLSTM(
        in_ch=C, base=16, emb_dim=64, d_tab=d_tab,
        tab_hidden=32, hidden_t=32, n_lstm_layers=1, dropout=0.0,
    )
    model.eval()
    x_seq   = torch.randn(B, L, C, P, P)
    tab_seq = torch.randn(B, L, d_tab)
    with torch.no_grad():
        out = model(x_seq, tab_seq)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in FusionResNetLSTM output"


def test_fusion_resnet_freeze_encoder():
    """freeze_encoder=True leaves encoder parameters with requires_grad=False."""
    model = FusionResNetLSTM(
        in_ch=16, base=16, emb_dim=64, d_tab=9,
        tab_hidden=32, hidden_t=32, freeze_encoder=True,
    )
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param {name} should be frozen"
    for name, param in model.tab_proj.named_parameters():
        assert param.requires_grad, f"tab_proj param {name} should be trainable"


# ─────────────────────────────────────────────────────────────────────────────
# FusionGraphSAGE_LSTM
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_graphsage_forward():
    """End-to-end forward pass with B=2, L=4, N=64 nodes: output is (B,), no NaNs."""
    B, L, P, C = 2, 4, 8, 16
    N     = P * P   # 64 nodes
    d_tab = 9
    edge_index, edge_weight = build_weighted_knn_edge_index(P, k=8)

    model = FusionGraphSAGE_LSTM(
        in_dim=C, hidden_g=32, n_sage_layers=2, hidden_t=32,
        n_lstm_layers=1, dropout_head=0.0, input_bn=True, concat_agg=True,
        d_tab=d_tab, tab_hidden=32,
        edge_index=edge_index, edge_weight=edge_weight,
    )
    model.eval()
    x_seq   = torch.randn(B, L, N, C)
    tab_seq = torch.randn(B, L, d_tab)
    with torch.no_grad():
        out = model(x_seq, tab_seq)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in FusionGraphSAGE_LSTM output"


def test_fusion_graphsage_no_edge_weight():
    """Unweighted graph (no edge_weight) should also produce valid output."""
    B, L, P, C = 2, 3, 4, 16
    N = P * P
    from solar_uq.models.graphsage_lstm import build_edge_index_8n
    edge_index = build_edge_index_8n(P)

    model = FusionGraphSAGE_LSTM(
        in_dim=C, hidden_g=16, n_sage_layers=2, hidden_t=16,
        d_tab=5, tab_hidden=8, edge_index=edge_index,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(B, L, N, C), torch.randn(B, L, 5))
    assert out.shape == (B,)
    assert not torch.isnan(out).any()


# ─────────────────────────────────────────────────────────────────────────────
# FusionPatchSeqDataset shape check (synthetic, no real parquet)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_dataset_shapes(tmp_path):
    """FusionPatchSeqDataset returns (sat_seq, tab_seq, y) with correct shapes."""
    from solar_uq.data import TargetNormalizer
    from solar_uq.loaders import FusionPatchSeqDataset, n_tab_features, DEFAULT_FEATURE_COLS

    # ---- synthetic surface parquet ----
    idx  = pd.date_range("2023-01-01", periods=200, freq="10min", tz="UTC")
    ghi  = np.random.uniform(0, 800, len(idx)).astype(np.float32)
    df   = pd.DataFrame({
        "ghi":               ghi,
        "clear_sky_index":   np.clip(ghi / 600, 0, 2).astype(np.float32),
        "air_temperature_c": np.random.normal(25, 5, len(idx)).astype(np.float32),
        "wind_y":            np.random.normal(0, 2, len(idx)).astype(np.float32),
        "wind_x":            np.random.normal(0, 2, len(idx)).astype(np.float32),
        "doy_sin":           np.sin(2 * np.pi * np.arange(len(idx)) / 365).astype(np.float32),
        "doy_cos":           np.cos(2 * np.pi * np.arange(len(idx)) / 365).astype(np.float32),
        "hour_of_day":       (np.arange(len(idx)) % 144 * 10 / 60).astype(np.float32),
    }, index=idx)
    surface_path = tmp_path / "ground.parquet"
    df.to_parquet(surface_path)

    # ---- synthetic .npz patches ----
    P, C, slots_per_file = 8, 16, 6
    patches_root = tmp_path / "patches"
    patch_ts     = pd.Timestamp("2023-01-01 10:00:00", tz="UTC")

    from solar_uq.data import patch_path_for_timestamp
    p_path = patch_path_for_timestamp(patch_ts, patches_root)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(slots_per_file, C, P, P).astype(np.float16)
    np.savez_compressed(str(p_path), patch=arr)   # data.py loads key "patch"

    # ---- synthetic manifest ----
    import json as _json
    history_ts_str = _json.dumps([str(patch_ts)] * 4)  # L=4, must be JSON
    manifest = pd.DataFrame({
        "site":       ["test"],
        "t_label":    [patch_ts],
        "t_target":   [patch_ts + pd.Timedelta(hours=1)],
        "y":          [np.float32(300.0)],
        "history_ts": [history_ts_str],
    })

    normalizer = TargetNormalizer.from_train(np.array([300.0]))
    tab_stats  = FusionPatchSeqDataset.compute_tab_stats(
        manifest, patches_root, normalizer, surface_path, n_samples=1, seed=0,
    )
    ds = FusionPatchSeqDataset(
        manifest, patches_root, normalizer, surface_path, tab_stats=tab_stats,
    )

    D_TAB = n_tab_features(DEFAULT_FEATURE_COLS)
    assert len(ds) == 1

    sat_seq, tab_seq, y = ds[0]
    assert sat_seq.shape  == (4, C, P, P), f"sat_seq shape={sat_seq.shape}"
    assert tab_seq.shape  == (4, D_TAB),   f"tab_seq shape={tab_seq.shape}"
    assert y.shape        == (),            f"y shape={y.shape}"
    assert not torch.isnan(sat_seq).any()
    assert not torch.isnan(tab_seq).any()

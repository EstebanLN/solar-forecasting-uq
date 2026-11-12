"""The fusion datasets' precomputed index returns exactly what the uncached path does.

Runs on the real data when it is present (skipped otherwise): random samples
from both fusion dataset classes are compared tensor by tensor -- satellite
sequence, tabular sequence and target -- for bit-for-bit equality and dtype.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from solar_uq.data import TargetNormalizer, preload_patch_cache
from solar_uq.loaders.fusion_dataset import FusionGraphSeqDataset, FusionPatchSeqDataset

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("FUSION_CACHE_TEST_N", "40"))


def _setup(site, hours):
    site_dir = ROOT / "data" / "datasets" / "manifest_v1" / site / f"h{hours}"
    patches = ROOT / "data" / "patches_v1" / site / "P16"
    surf = ROOT / "data" / "ground_aligned" / f"ground_10min_utc_{site}.parquet"
    if not (site_dir.exists() and patches.exists() and surf.exists()):
        pytest.skip("real data not available")
    tr = pd.read_parquet(site_dir / "manifest_train.parquet")
    va = pd.read_parquet(site_dir / "manifest_val.parquet")
    preload_patch_cache(patches)
    norm = TargetNormalizer.from_train(tr["y"].astype(float).to_numpy())
    stats = FusionPatchSeqDataset.compute_tab_stats(tr, patches, norm, surf, n_samples=2000, seed=0)
    return va, patches, norm, surf, stats


@pytest.mark.parametrize("cls", [FusionPatchSeqDataset, FusionGraphSeqDataset])
@pytest.mark.parametrize("site,hours", [("uniandes", 6), ("elpaso", 1)])
def test_cached_equals_uncached(cls, site, hours):
    va, patches, norm, surf, stats = _setup(site, hours)
    ds = cls(va, patches, norm, surf, tab_stats=stats)
    assert ds._locs is not None
    idx = np.random.default_rng(0).choice(len(ds), size=min(N, len(ds)), replace=False)
    for i in idx:
        fast = ds[int(i)]
        slow = ds._item_uncached(int(i))
        for a, b in zip(fast, slow):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a, b)

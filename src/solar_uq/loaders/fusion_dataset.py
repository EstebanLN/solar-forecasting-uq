"""Fusion datasets: satellite patch sequences + surface tabular features.

Each sample returns a 3-tuple (sat_seq, tab_seq, y) instead of the standard
(x_seq, y) from PatchSeqDataset / GraphSeqDataset, so the training loop must
be called with fusion=True (see solar_uq.train.train_one_model).

Tabular feature vector (D_tab = 9 with DEFAULT_FEATURE_COLS):

    [0]  ghi_norm          — GHI at history step t, normalized by TargetNormalizer
    [1]  clear_sky_index   — kt = GHI / GHI_clearsky  (already in ground parquet)
    [2]  air_temperature_c — surface air temperature (°C)
    [3]  wind_y            — northward wind component (m/s)
    [4]  wind_x            — eastward wind component (m/s)
    [5]  doy_sin           — sin(2π·day_of_year/365)
    [6]  doy_cos           — cos(2π·day_of_year/365)
    [7]  hour_sin          — sin(2π·hour_UTC/24)   derived from hour_of_day
    [8]  hour_cos          — cos(2π·hour_UTC/24)   derived from hour_of_day

``hour_of_day`` expands to two features (sin + cos), so it counts as 2 in D_tab.
Columns absent from the ground parquet default to 0; NaN values are imputed with
the per-column median computed over the slice of data provided at construction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from solar_uq.data import (
    TargetNormalizer,
    _parse_history_ts,
    filter_missing_patches,
    load_patch_npz,
    patch_path_for_timestamp,
    slot_for_timestamp,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_COLS: List[str] = [
    "ghi",             # normalized in __getitem__; placed first
    "clear_sky_index",
    "air_temperature_c",
    "wind_y",
    "wind_x",
    "doy_sin",
    "doy_cos",
    "hour_of_day",     # expands to hour_sin + hour_cos (adds 1 extra dim)
]


def n_tab_features(feature_cols: List[str]) -> int:
    """Return the number of tabular features produced from *feature_cols*.

    ``hour_of_day`` expands to two features (sin + cos); all other columns map
    one-to-one.
    """
    n = len(feature_cols)
    if "hour_of_day" in feature_cols:
        n += 1  # hour_of_day → sin + cos = 2 features instead of 1
    return n


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_surface(surface_parquet: Path) -> pd.DataFrame:
    """Load ground parquet and ensure a UTC-aware DatetimeIndex."""
    df = pd.read_parquet(surface_parquet)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _build_tab_vector(
    t: pd.Timestamp,
    surface_df: pd.DataFrame,
    available_cols: List[str],
    feature_cols: List[str],
    tab_medians: Dict[str, float],
    normalizer: TargetNormalizer,
) -> np.ndarray:
    """Build a D_tab-dimensional feature vector for timestamp *t*.

    If the timestamp is absent from *surface_df*, the column median is used.
    If a requested column is absent from the parquet, 0.0 is used.
    """
    t_key = t.floor("10min")
    row_available = t_key in surface_df.index

    feats: list[float] = []
    for col in feature_cols:
        if col == "hour_of_day":
            # Use the actual UTC hour from t_key (more reliable than the
            # stored integer, which may differ by rounding on the boundary).
            hour = float(t_key.hour) + float(t_key.minute) / 60.0
            feats.append(float(np.sin(2.0 * np.pi * hour / 24.0)))
            feats.append(float(np.cos(2.0 * np.pi * hour / 24.0)))

        elif col == "ghi":
            if col in available_cols and row_available:
                raw = surface_df.at[t_key, col]
                val = float(raw) if not (isinstance(raw, float) and np.isnan(raw)) else tab_medians.get(col, 0.0)
            else:
                if not row_available:
                    logger.debug("Missing surface row at %s; using median for ghi", t_key)
                val = tab_medians.get(col, 0.0)
            feats.append(normalizer.normalize(val))

        else:
            if col in available_cols and row_available:
                raw = surface_df.at[t_key, col]
                val = float(raw) if not (isinstance(raw, float) and np.isnan(raw)) else tab_medians.get(col, 0.0)
            else:
                if not row_available and col in available_cols:
                    logger.debug("Missing surface row at %s; using median for %s", t_key, col)
                val = tab_medians.get(col, 0.0)
            feats.append(val)

    return np.array(feats, dtype=np.float32)


def _compute_medians(surface_df: pd.DataFrame, available_cols: List[str]) -> Dict[str, float]:
    """Column-wise medians over the provided surface slice (used for imputation).

    Callers should restrict *surface_df* to the training period before calling
    (see compute_tab_stats, which does this and ships the result to val/test
    via tab_stats["medians"]) so that imputation constants never see val/test
    data. The full-record fallback in the dataset constructors exists only for
    backward compatibility with tab_stats dicts produced before this fix; the
    difference is a handful of imputation constants for missing rows and is
    numerically negligible either way.
    """
    medians: Dict[str, float] = {}
    for col in available_cols:
        valid = surface_df[col].dropna()
        medians[col] = float(valid.median()) if len(valid) else 0.0
    return medians


# ---------------------------------------------------------------------------
# FusionPatchSeqDataset
# ---------------------------------------------------------------------------

class FusionPatchSeqDataset(Dataset):
    """Augments PatchSeqDataset with surface tabular features per time step.

    Returns a 3-tuple per sample:
        sat_seq  : FloatTensor (L, C=16, P, P) — satellite patch sequence
        tab_seq  : FloatTensor (L, D_tab)       — tabular features
        y        : FloatTensor ()               — normalised GHI target at t+H

    Args:
        manifest        : DataFrame from manifest_*.parquet (columns: y, history_ts, …)
        patches_root    : directory containing pre-extracted .npz patch files
        normalizer      : TargetNormalizer fitted on the training set
        surface_parquet : path to ``ground_10min_utc_{site}.parquet``
        feature_cols    : parquet columns to include; defaults to DEFAULT_FEATURE_COLS.
                          ``hour_of_day`` is replaced by (hour_sin, hour_cos).
        tab_stats       : dict with keys ``mean`` and ``std`` (lists of length D_tab)
                          computed by :meth:`compute_tab_stats` on the training split.
                          Required when ``normalize_tab=True``.
        normalize_tab   : if True, standardise tab_seq using *tab_stats*
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
        surface_parquet: Path,
        feature_cols: Optional[List[str]] = None,
        tab_stats: Optional[Dict] = None,
        normalize_tab: bool = True,
    ) -> None:
        self.patches_root = Path(patches_root)
        self.man = filter_missing_patches(manifest, self.patches_root)
        self.normalizer = normalizer
        self.feature_cols: List[str] = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.normalize_tab = normalize_tab

        surface_df = _load_surface(Path(surface_parquet))
        missing_raw = set(self.feature_cols) - {"hour_of_day"} - set(surface_df.columns)
        if missing_raw:
            logger.warning(
                "Requested feature_cols absent from ground parquet (will use 0): %s",
                missing_raw,
            )
        self.available_cols: List[str] = [
            c for c in self.feature_cols if c != "hour_of_day" and c in surface_df.columns
        ]
        self.surface_df = surface_df[self.available_cols] if self.available_cols else surface_df[[]]
        # Imputation medians: prefer the training-period medians shipped in
        # tab_stats (no val/test data involved); fall back to full-record
        # medians only for legacy tab_stats dicts that predate the fix.
        if tab_stats is not None and "medians" in tab_stats:
            self.tab_medians = dict(tab_stats["medians"])
        else:
            self.tab_medians = _compute_medians(self.surface_df, self.available_cols)

        if normalize_tab:
            if tab_stats is None:
                raise ValueError(
                    "tab_stats must be provided when normalize_tab=True. "
                    "Call FusionPatchSeqDataset.compute_tab_stats() on the training split first."
                )
            self._tab_mean = np.array(tab_stats["mean"], dtype=np.float32)
            self._tab_std  = np.array(tab_stats["std"],  dtype=np.float32)
        else:
            self._tab_mean = None
            self._tab_std  = None

    # ------------------------------------------------------------------

    @property
    def d_tab(self) -> int:
        """Number of tabular features per timestep."""
        return n_tab_features(self.feature_cols)

    def __len__(self) -> int:
        return len(self.man)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.man.iloc[i]
        y_norm     = self.normalizer.normalize(float(row["y"]))
        history_ts = _parse_history_ts(row["history_ts"])

        frames:   list[np.ndarray] = []
        tab_vecs: list[np.ndarray] = []

        for ts_str in history_ts:
            t    = pd.to_datetime(ts_str, utc=True)
            p    = patch_path_for_timestamp(t, self.patches_root)
            slot = slot_for_timestamp(t)

            if not p.exists():
                raise FileNotFoundError(f"Missing patch file: {p}")

            arr   = load_patch_npz(str(p))       # (6, 16, P, P) float16
            frame = arr[slot]                     # (16, P, P)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            frames.append(frame)

            tab_vec = _build_tab_vector(
                t, self.surface_df, self.available_cols,
                self.feature_cols, self.tab_medians, self.normalizer,
            )
            tab_vecs.append(tab_vec)

        x_seq   = np.stack(frames,   axis=0)   # (L, 16, P, P)
        tab_seq = np.stack(tab_vecs, axis=0)   # (L, D_tab)

        if self.normalize_tab and self._tab_mean is not None:
            tab_seq = (tab_seq - self._tab_mean) / self._tab_std

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(tab_seq),
            torch.tensor(y_norm, dtype=torch.float32),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def compute_tab_stats(
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
        surface_parquet: Path,
        feature_cols: Optional[List[str]] = None,
        n_samples: int = 5_000,
        seed: int = 42,
    ) -> Dict:
        """Compute per-feature mean and std over a sample of the training manifest.

        Must be called on the **training split only** and the returned dict
        passed to the val/test dataset constructors to avoid data leakage.
        Imputation medians are likewise computed over the training period only
        (t_label range of the provided manifest, minus the history window) and
        shipped in the returned dict so every split imputes with the same
        train-only constants.

        Returns:
            dict with keys ``mean`` (List[float]), ``std`` (List[float]),
            ``medians`` (Dict[str, float]), ``d_tab`` (int),
            ``feature_cols`` (List[str]).
        """
        ds = FusionPatchSeqDataset(
            manifest=manifest,
            patches_root=patches_root,
            normalizer=normalizer,
            surface_parquet=surface_parquet,
            feature_cols=feature_cols,
            normalize_tab=False,
        )

        # Train-period imputation medians (replaces the constructor's
        # full-record fallback before any feature vector is sampled).
        t_labels = pd.to_datetime(manifest["t_label"], utc=True)
        train_lo = t_labels.min() - pd.Timedelta(hours=4)   # cover history window
        train_hi = t_labels.max()
        train_slice = ds.surface_df.loc[train_lo:train_hi]
        ds.tab_medians = _compute_medians(train_slice, ds.available_cols)

        rng  = np.random.default_rng(seed)
        n    = min(len(ds.man), n_samples)
        idxs = rng.choice(len(ds.man), size=n, replace=False)

        all_vecs: list[np.ndarray] = []
        for i in idxs:
            row = ds.man.iloc[int(i)]
            for ts_str in _parse_history_ts(row["history_ts"]):
                t = pd.to_datetime(ts_str, utc=True)
                all_vecs.append(
                    _build_tab_vector(
                        t, ds.surface_df, ds.available_cols,
                        ds.feature_cols, ds.tab_medians, ds.normalizer,
                    )
                )

        arr  = np.stack(all_vecs)                      # (N_total, D_tab)
        mean = arr.mean(axis=0).astype(np.float32)
        std  = (arr.std(axis=0) + 1e-6).astype(np.float32)
        return {
            "mean":         mean.tolist(),
            "std":          std.tolist(),
            "medians":      ds.tab_medians,
            "d_tab":        ds.d_tab,
            "feature_cols": ds.feature_cols,
        }


# ---------------------------------------------------------------------------
# FusionGraphSeqDataset
# ---------------------------------------------------------------------------

class FusionGraphSeqDataset(Dataset):
    """Augments GraphSeqDataset with surface tabular features per time step.

    Returns a 3-tuple per sample:
        sat_seq  : FloatTensor (L, N=P*P, C=16) — node features for GraphSAGE
        tab_seq  : FloatTensor (L, D_tab)
        y        : FloatTensor ()

    All tabular logic is identical to :class:`FusionPatchSeqDataset`; only the
    satellite frame layout differs ((C, P, P) → (P*P, C) node-feature order).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
        surface_parquet: Path,
        feature_cols: Optional[List[str]] = None,
        tab_stats: Optional[Dict] = None,
        normalize_tab: bool = True,
    ) -> None:
        self.patches_root = Path(patches_root)
        self.man = filter_missing_patches(manifest, self.patches_root)
        self.normalizer = normalizer
        self.feature_cols: List[str] = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.normalize_tab = normalize_tab

        surface_df = _load_surface(Path(surface_parquet))
        missing_raw = set(self.feature_cols) - {"hour_of_day"} - set(surface_df.columns)
        if missing_raw:
            logger.warning(
                "Requested feature_cols absent from ground parquet (will use 0): %s",
                missing_raw,
            )
        self.available_cols: List[str] = [
            c for c in self.feature_cols if c != "hour_of_day" and c in surface_df.columns
        ]
        self.surface_df = surface_df[self.available_cols] if self.available_cols else surface_df[[]]
        # Same train-only imputation medians convention as FusionPatchSeqDataset.
        if tab_stats is not None and "medians" in tab_stats:
            self.tab_medians = dict(tab_stats["medians"])
        else:
            self.tab_medians = _compute_medians(self.surface_df, self.available_cols)

        if normalize_tab:
            if tab_stats is None:
                raise ValueError(
                    "tab_stats must be provided when normalize_tab=True. "
                    "Call FusionPatchSeqDataset.compute_tab_stats() on the training split first."
                )
            self._tab_mean = np.array(tab_stats["mean"], dtype=np.float32)
            self._tab_std  = np.array(tab_stats["std"],  dtype=np.float32)
        else:
            self._tab_mean = None
            self._tab_std  = None

    @property
    def d_tab(self) -> int:
        return n_tab_features(self.feature_cols)

    def __len__(self) -> int:
        return len(self.man)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.man.iloc[i]
        y_norm     = self.normalizer.normalize(float(row["y"]))
        history_ts = _parse_history_ts(row["history_ts"])

        frames:   list[np.ndarray] = []
        tab_vecs: list[np.ndarray] = []

        for ts_str in history_ts:
            t    = pd.to_datetime(ts_str, utc=True)
            p    = patch_path_for_timestamp(t, self.patches_root)
            slot = slot_for_timestamp(t)

            if not p.exists():
                raise FileNotFoundError(f"Missing patch file: {p}")

            arr   = load_patch_npz(str(p))       # (6, 16, P, P) float16
            frame = arr[slot]                     # (16, P, P)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            # Reshape to node-feature layout: (C, P, P) → (P*P, C)
            C, P1, P2 = frame.shape
            node_feats = np.transpose(frame, (1, 2, 0)).reshape(P1 * P2, C)
            frames.append(node_feats)

            tab_vec = _build_tab_vector(
                t, self.surface_df, self.available_cols,
                self.feature_cols, self.tab_medians, self.normalizer,
            )
            tab_vecs.append(tab_vec)

        x_seq   = np.stack(frames,   axis=0)   # (L, N, 16)
        tab_seq = np.stack(tab_vecs, axis=0)   # (L, D_tab)

        if self.normalize_tab and self._tab_mean is not None:
            tab_seq = (tab_seq - self._tab_mean) / self._tab_std

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(tab_seq),
            torch.tensor(y_norm, dtype=torch.float32),
        )

    # compute_tab_stats delegates to FusionPatchSeqDataset (same logic)
    @staticmethod
    def compute_tab_stats(
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
        surface_parquet: Path,
        feature_cols: Optional[List[str]] = None,
        n_samples: int = 5_000,
        seed: int = 42,
    ) -> Dict:
        """Identical to FusionPatchSeqDataset.compute_tab_stats — tabular only."""
        return FusionPatchSeqDataset.compute_tab_stats(
            manifest=manifest,
            patches_root=patches_root,
            normalizer=normalizer,
            surface_parquet=surface_parquet,
            feature_cols=feature_cols,
            n_samples=n_samples,
            seed=seed,
        )

"""Dataset classes and data-loading utilities for satellite patch sequences."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

# ---------------------------------------------------------------------------
# Target normalizer
# ---------------------------------------------------------------------------

@dataclass
class TargetNormalizer:
    """Normalizes GHI targets computed from training data."""
    mean: float
    std: float

    @classmethod
    def from_train(cls, y_train: np.ndarray) -> "TargetNormalizer":
        return cls(
            mean=float(np.mean(y_train)),
            std=float(np.std(y_train) + 1e-6),
        )

    def normalize(self, y: float) -> float:
        return (y - self.mean) / self.std

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean


# ---------------------------------------------------------------------------
# Patch file helpers
# ---------------------------------------------------------------------------

def patch_path_for_timestamp(t: pd.Timestamp, patches_root: Path) -> Path:
    """`patches_root / YYYY / MM / YYYYMMDD_HH_patch.npz`"""
    fname = f"{t.strftime('%Y%m%d')}_{t.strftime('%H')}_patch.npz"
    return patches_root / t.strftime("%Y") / t.strftime("%m") / fname


def slot_for_timestamp(t: pd.Timestamp) -> int:
    """10-min slot index within the hour (0-5)."""
    return int(t.strftime("%M")) // 10


def _load_patch_npz_nocache(path_str: str) -> np.ndarray:
    """Load patch file from disk.  Returns float16 array (6, 16, P, P)."""
    with np.load(Path(path_str)) as d:
        return d["patch"]


@lru_cache(maxsize=16)
def _load_patch_npz_maincache(path_str: str) -> np.ndarray:
    return _load_patch_npz_nocache(path_str)


# Process-wide cache populated by preload_patch_cache().  Left empty unless a
# caller opts in, so scripts that never call preload_patch_cache() keep the
# exact old per-process/main-only lru_cache behavior below.
_PATCH_CACHE: dict[str, np.ndarray] = {}


def preload_patch_cache(patches_root: Path) -> None:
    """Eagerly load every patch .npz file under *patches_root* into RAM.

    A full patch store is <1GB per site (P16), so it fits comfortably next to
    model/optimizer state. Call this once in the main process, before training
    starts, so every load_patch_npz() call becomes a dict lookup instead of a
    disk read + npz decompression on every access, every epoch.

    Note: train_loader deliberately uses num_workers=0 (see the training
    scripts), so this cache is read directly by the main process rather than
    inherited by forked workers via copy-on-write — forking a DataLoader
    worker after the process has touched CUDA can crash with "CUDA error:
    initialization error", so workers are avoided rather than relied upon.
    """
    patches_root = Path(patches_root)
    n_loaded = 0
    for p in sorted(patches_root.rglob("*_patch.npz")):
        key = str(p)
        if key not in _PATCH_CACHE:
            with np.load(p) as d:
                _PATCH_CACHE[key] = d["patch"]
            n_loaded += 1
    print(f"[data] Preloaded {n_loaded} patch files from {patches_root} "
          f"({len(_PATCH_CACHE)} total cached).")


def load_patch_npz(path_str: str) -> np.ndarray:
    """Worker-safe loader.

    Uses the process-wide cache from preload_patch_cache() when populated;
    otherwise falls back to the legacy behavior (cached only in the main
    process, to avoid RAM blowup from uncoordinated per-worker caches).
    """
    if path_str in _PATCH_CACHE:
        return _PATCH_CACHE[path_str]
    if get_worker_info() is None:
        return _load_patch_npz_maincache(path_str)
    return _load_patch_npz_nocache(path_str)


def _parse_history_ts(raw) -> List[str]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    return list(raw)


def filter_missing_patches(manifest: pd.DataFrame, patches_root: Path) -> pd.DataFrame:
    """Drop manifest rows where any history_ts patch file is absent.

    Checks the in-memory cache (if preload_patch_cache() was called for this
    root) instead of hitting the filesystem per row/timestamp.
    """
    patches_root = Path(patches_root)
    use_cache = bool(_PATCH_CACHE)

    def _all_present(row) -> bool:
        for ts_str in _parse_history_ts(row["history_ts"]):
            t = pd.to_datetime(ts_str, utc=True)
            path = patch_path_for_timestamp(t, patches_root)
            if use_cache:
                if str(path) not in _PATCH_CACHE:
                    return False
            elif not path.exists():
                return False
        return True

    mask = manifest.apply(_all_present, axis=1)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"[data] Dropped {n_dropped}/{len(manifest)} samples with missing patches.")
    return manifest[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PatchSeqDataset(Dataset):
    """
    Returns:
        x_seq : FloatTensor (L, C=16, P, P)
        y     : FloatTensor scalar, normalized
    """
    def __init__(
        self,
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
    ):
        self.patches_root = Path(patches_root)
        self.man = filter_missing_patches(manifest, self.patches_root)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.man)

    def __getitem__(self, i: int):
        row = self.man.iloc[i]
        y = self.normalizer.normalize(float(row["y"]))
        history_ts = _parse_history_ts(row["history_ts"])

        frames = []
        for ts_str in history_ts:
            t = pd.to_datetime(ts_str, utc=True)
            p = patch_path_for_timestamp(t, self.patches_root)
            slot = slot_for_timestamp(t)

            if not p.exists():
                raise FileNotFoundError(f"Missing patch file: {p}")

            arr = load_patch_npz(str(p))   # (6, 16, P, P) float16
            frame = arr[slot]              # (16, P, P)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            frames.append(frame)

        x_seq = np.stack(frames, axis=0)  # (L, 16, P, P)
        return torch.from_numpy(x_seq), torch.tensor(y, dtype=torch.float32)


class GraphSeqDataset(Dataset):
    """
    Returns:
        x_seq : FloatTensor (L, N=P*P, C=16)  — node features for graph models
        y     : FloatTensor scalar, normalized
    """
    def __init__(
        self,
        manifest: pd.DataFrame,
        patches_root: Path,
        normalizer: TargetNormalizer,
    ):
        self.patches_root = Path(patches_root)
        self.man = filter_missing_patches(manifest, self.patches_root)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.man)

    def __getitem__(self, i: int):
        row = self.man.iloc[i]
        y = self.normalizer.normalize(float(row["y"]))
        history_ts = _parse_history_ts(row["history_ts"])

        frames = []
        for ts_str in history_ts:
            t = pd.to_datetime(ts_str, utc=True)
            p = patch_path_for_timestamp(t, self.patches_root)
            slot = slot_for_timestamp(t)

            if not p.exists():
                raise FileNotFoundError(f"Missing patch file: {p}")

            arr = load_patch_npz(str(p))   # (6, 16, P, P) float16
            frame = arr[slot]              # (16, P, P)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            # (16, P, P) -> (P*P, 16)  node-feature layout
            C, P1, P2 = frame.shape
            node_feats = np.transpose(frame, (1, 2, 0)).reshape(P1 * P2, C)
            frames.append(node_feats)

        x_seq = np.stack(frames, axis=0)  # (L, N, 16)
        return torch.from_numpy(x_seq), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> DataLoader:
    """Creates a reproducible DataLoader with worker seeding."""
    g = torch.Generator()
    g.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def read_history_steps_from_manifest(manifest: pd.DataFrame) -> int:
    """Infer L from the mode of history_ts list lengths (authoritative source)."""
    def _hist_len(x):
        if isinstance(x, str):
            x = json.loads(x)
        elif isinstance(x, np.ndarray):
            x = x.tolist()
        return len(x)
    return int(manifest["history_ts"].map(_hist_len).mode().iloc[0])

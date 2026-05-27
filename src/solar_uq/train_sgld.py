"""SGLD training loop: burn-in phase + posterior sampling phase.

Two-phase protocol
------------------
Phase 1 — Burn-in (epochs 1 … burn_in):
    Run SGLD but discard all parameter samples.  The chain must travel far
    enough from its initialisation (the Optuna best checkpoint or random init)
    before samples are representative of the posterior.  Typical: 300–500 epochs.

Phase 2 — Sampling (epochs burn_in+1 … burn_in + n_samples*sample_every):
    Every `sample_every` epochs, save a checkpoint.  These checkpoints are
    (approximately) i.i.d. draws from p(θ | D_train).  Typical: 10 checkpoints
    spaced 100 epochs apart (1 000 additional epochs).

Ensemble inference
------------------
After sampling, each checkpoint is loaded and run on the test set.  The ensemble
mean is the point prediction (reported to 08_results_table.py via summary.json).
The per-sample standard deviation is the empirical posterior uncertainty.

AMP
---
Disabled intentionally.  SGLD noise must be injected at the same fp32 scale as
the gradients; mixing fp16 loss scaling with fp32 noise injection produces
incorrectly scaled posteriors.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import TargetNormalizer
from .metrics import metrics_from_arrays
from .sgld import SGLD
from .train import eval_model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_sgld(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    normalizer: TargetNormalizer,
    run_dir: Path,
    *,
    sgld_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    l1_reg: float = 0.0,
    burn_in: int = 500,
    sample_every: int = 100,
    n_samples: int = 10,
    grad_clip_norm: float = 1.0,
    day_threshold: float = 20.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run SGLD burn-in + sampling and evaluate the posterior ensemble on test.

    Args:
        model:          Instantiated nn.Module already on ``device``.
        train_loader:   DataLoader for training split (yields (x_seq, y)).
        val_loader:     DataLoader for validation split.
        test_loader:    DataLoader for test split.
        normalizer:     TargetNormalizer fitted on training targets.
        run_dir:        Directory where checkpoints and logs are saved.
        sgld_lr:        SGLD step size ε (see sgld.py).
        weight_decay:   Gaussian prior precision (same as Optuna best_params value).
        l1_reg:         L1 penalty coefficient from Optuna (0.0 = disabled).
        burn_in:        Epochs to run before saving any checkpoint.
        sample_every:   Epochs between consecutive checkpoint saves.
        n_samples:      Number of checkpoints to collect.
        grad_clip_norm: Max-norm gradient clipping (same convention as train.py).
        day_threshold:  GHI threshold (W/m²) for daytime-only metrics.
        device:         "cuda" or "cpu".

    Returns:
        dict with keys:
            checkpoint_paths     : list of saved checkpoint paths (strings)
            train_log            : list of per-epoch metric dicts
            train_seconds_total  : float
            ensemble_test        : metrics dict from ensemble mean on test set
            n_samples_collected  : int (== len(checkpoint_paths))
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    opt = SGLD(model.parameters(), lr=sgld_lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    total_epochs = burn_in + n_samples * sample_every
    checkpoint_paths: List[str] = []
    train_log: List[dict] = []

    t_total = time.time()

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses: List[float] = []

        for x_seq, y in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            yhat = model(x_seq)
            loss = loss_fn(yhat, y)
            if l1_reg > 0.0:
                loss = loss + l1_reg * sum(p.abs().sum() for p in model.parameters())
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            opt.step()
            tr_losses.append(loss.item())

        phase = "burn-in" if epoch <= burn_in else "sample"
        val_m = eval_model(model, val_loader, normalizer, day_threshold, device)

        entry: dict = {
            "epoch":          epoch,
            "phase":          phase,
            "train_mse_norm": float(np.mean(tr_losses)),
            "val_rmse":       float(val_m["rmse"]),
            "val_rmse_day":   float(val_m["rmse_day"]),
            "epoch_seconds":  float(time.time() - t0),
        }

        # Checkpoint during sampling phase
        sampling_epoch = epoch > burn_in
        is_save = sampling_epoch and (epoch - burn_in) % sample_every == 0
        if is_save:
            sample_idx = (epoch - burn_in) // sample_every
            ckpt_path  = run_dir / f"checkpoint_e{epoch:04d}.pt"
            torch.save(
                {
                    "epoch":       epoch,
                    "sample_idx":  sample_idx,
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                },
                ckpt_path,
            )
            checkpoint_paths.append(str(ckpt_path))
            entry["checkpoint_saved"] = str(ckpt_path)

        train_log.append(entry)

        ckpt_tag = f" | CKPT #{(epoch - burn_in) // sample_every}" if is_save else ""
        print(
            f"[{phase}] {epoch:04d}/{total_epochs} | "
            f"train={entry['train_mse_norm']:.5f} | "
            f"val_rmse_day={entry['val_rmse_day']:.2f} | "
            f"{entry['epoch_seconds']:.1f}s"
            f"{ckpt_tag}"
        )

    train_seconds_total = float(time.time() - t_total)

    # Ensemble inference
    print(f"\nRunning ensemble inference over {len(checkpoint_paths)} checkpoints ...")
    ensemble_preds, y_true = _ensemble_predict(
        model, checkpoint_paths, test_loader, normalizer, device
    )
    ensemble_test = _ensemble_metrics(y_true, ensemble_preds, day_threshold)
    print(
        f"Ensemble test: RMSE_day={ensemble_test['rmse_day']:.2f} | "
        f"skill_day={ensemble_test.get('skill_day_vs_persistence', float('nan')):.3f} | "
        f"std_mean={ensemble_test['ensemble_std_mean']:.2f}"
    )

    # Persist training log
    with open(run_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    return {
        "checkpoint_paths":    checkpoint_paths,
        "train_log":           train_log,
        "train_seconds_total": train_seconds_total,
        "ensemble_test":       ensemble_test,
        "n_samples_collected": len(checkpoint_paths),
    }


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ensemble_predict(
    model: nn.Module,
    checkpoint_paths: List[str],
    loader: DataLoader,
    normalizer: TargetNormalizer,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load each checkpoint and collect test predictions.

    Returns:
        ensemble_preds : shape (n_checkpoints, N) in physical units (W/m²)
        y_true         : shape (N,) in physical units
    """
    all_preds: List[np.ndarray] = []
    y_true: np.ndarray | None = None

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ys: List[np.ndarray] = []
        yhats: List[np.ndarray] = []
        for x_seq, y in loader:
            x_seq = x_seq.to(device, non_blocking=True)
            yhat  = model(x_seq)
            ys.append(y.numpy())
            yhats.append(yhat.detach().cpu().numpy())

        y_phys    = normalizer.denormalize(np.concatenate(ys))
        yhat_phys = normalizer.denormalize(np.concatenate(yhats))

        if y_true is None:
            y_true = y_phys
        all_preds.append(yhat_phys)

    assert y_true is not None, "No checkpoints found — nothing to evaluate."
    return np.stack(all_preds, axis=0), y_true  # (n_ckpt, N), (N,)


def _ensemble_metrics(
    y_true: np.ndarray,
    ensemble_preds: np.ndarray,
    day_threshold: float,
) -> Dict[str, Any]:
    """Point metrics from ensemble mean + empirical uncertainty statistics.

    The returned dict is intentionally compatible with the ``final_test`` block
    expected by 08_results_table.py (rmse, rmse_day, mae, mae_day, skill_*).
    Additional keys (ensemble_std_*) carry UQ information.
    """
    mean_pred = ensemble_preds.mean(axis=0)   # (N,)
    std_pred  = ensemble_preds.std(axis=0)    # (N,)

    metrics = metrics_from_arrays(y_true, mean_pred, day_threshold=day_threshold)

    day_mask = y_true >= day_threshold
    metrics["ensemble_std_mean"]     = float(std_pred.mean())
    metrics["ensemble_std_day_mean"] = (
        float(std_pred[day_mask].mean()) if day_mask.any() else None
    )
    metrics["n_checkpoints"] = int(ensemble_preds.shape[0])

    return metrics

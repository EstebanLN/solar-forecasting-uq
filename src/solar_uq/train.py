"""Seeding, training loop, and evaluation utilities."""
from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import TargetNormalizer
from .metrics import metrics_from_arrays


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    normalizer: TargetNormalizer,
    day_threshold: float = 20.0,
    device: str = "cpu",
    fusion: bool = False,
) -> Dict[str, float]:
    """Run inference, denormalize, and compute metrics in physical units (W/m²).

    Both ResNetLSTM and GraphSAGE_LSTM expose the same ``model(x_seq)``
    signature — GraphSAGE stores its edge_index as a registered buffer.

    When ``fusion=True`` the loader is expected to yield 3-tuples
    ``(sat_seq, tab_seq, y)`` and the model is called as
    ``model(sat_seq, tab_seq)``.
    """
    model.eval()
    ys: list = []
    yhats: list = []

    for batch in loader:
        if fusion:
            x_seq, tab_seq, y = batch
            x_seq   = x_seq.to(device, non_blocking=True)
            tab_seq = tab_seq.to(device, non_blocking=True)
            y       = y.to(device, non_blocking=True)
            yhat    = model(x_seq, tab_seq)
        else:
            x_seq, y = batch
            x_seq = x_seq.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)
            yhat  = model(x_seq)
        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())

    y_arr    = np.concatenate(ys)
    yhat_arr = np.concatenate(yhats)

    y_phys    = normalizer.denormalize(y_arr)
    yhat_phys = normalizer.denormalize(yhat_arr)

    return metrics_from_arrays(y_phys, yhat_phys, day_threshold=day_threshold)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    normalizer: TargetNormalizer,
    device: str = "cpu",
    fusion: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays in physical units (W/m²).

    Companion to eval_model — use this when you need the raw arrays
    (e.g. for conformal calibration) rather than aggregated metrics.

    When ``fusion=True`` the loader yields ``(sat_seq, tab_seq, y)`` 3-tuples.
    """
    model.eval()
    ys: list = []
    yhats: list = []
    for batch in loader:
        if fusion:
            x_seq, tab_seq, y = batch
            x_seq   = x_seq.to(device, non_blocking=True)
            tab_seq = tab_seq.to(device, non_blocking=True)
            yhat    = model(x_seq, tab_seq)
        else:
            x_seq, y = batch
            x_seq = x_seq.to(device, non_blocking=True)
            yhat  = model(x_seq)
        ys.append(y.numpy())
        yhats.append(yhat.detach().cpu().numpy())
    y_phys    = normalizer.denormalize(np.concatenate(ys))
    yhat_phys = normalizer.denormalize(np.concatenate(yhats))
    return y_phys, yhat_phys


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: TargetNormalizer,
    lr: float,
    weight_decay: float,
    l1_reg: float = 0.0,
    optimizer: str = "adamw",      # "adamw" | "adam"
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
    epochs: int = 30,
    patience: int = 8,
    min_delta: float = 0.0,
    day_threshold: float = 20.0,
    device: str = "cpu",
    fusion: bool = False,
    optuna_trial: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train model with early stopping on val RMSE_day.

    Best weights are saved in-memory and reloaded before returning.

    When optuna_trial is given, reports val_rmse_day to it after every epoch
    and raises optuna.TrialPruned() if the trial should be pruned — lets a
    configured pruner (e.g. MedianPruner) actually stop clearly-bad trials
    early instead of always running the full early-stopping loop.

    Returns:
        model               : model with best weights loaded
        best_epoch          : epoch at which best val RMSE_day was achieved
        best_val_rmse_day   : best val RMSE_day (W/m²)
        final_val           : metrics dict from eval on val set with best model
        train_log           : list of per-epoch dicts
        train_seconds_total : float
    """
    opt_cls = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_log: list = []
    best_val_rmse_day = float("inf")
    best_state: Optional[dict] = None
    best_epoch = 0
    bad_epochs = 0

    t_train0 = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses: list = []

        for batch in train_loader:
            if fusion:
                x_seq, tab_seq, y = batch
                x_seq   = x_seq.to(device, non_blocking=True)
                tab_seq = tab_seq.to(device, non_blocking=True)
                y       = y.to(device, non_blocking=True)
            else:
                x_seq, y = batch
                x_seq = x_seq.to(device, non_blocking=True)
                y     = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                yhat = model(x_seq, tab_seq) if fusion else model(x_seq)
                loss = loss_fn(yhat, y)
                if l1_reg > 0.0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_reg * l1

            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            tr_losses.append(loss.item())

        val_metrics  = eval_model(model, val_loader, normalizer, day_threshold, device, fusion=fusion)
        val_rmse     = float(val_metrics["rmse"])
        val_rmse_day = float(val_metrics["rmse_day"])

        scheduler.step(val_rmse_day)

        epoch_out = {
            "epoch": epoch,
            "train_mse_norm": float(np.mean(tr_losses)),
            "val_rmse_phys": val_rmse,
            "val_mae_phys": float(val_metrics["mae"]),
            "val_rmse_day_phys": val_rmse_day,
            "val_mae_day_phys": float(val_metrics["mae_day"]),
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_seconds": float(time.time() - t0),
        }
        train_log.append(epoch_out)

        improved = (best_val_rmse_day - val_rmse_day) > min_delta
        if improved:
            best_val_rmse_day = val_rmse_day
            best_epoch        = epoch
            bad_epochs        = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if optuna_trial is not None:
            import optuna
            optuna_trial.report(val_rmse_day, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        print(
            f"Epoch {epoch:02d} | "
            f"train_mse_norm={epoch_out['train_mse_norm']:.5f} | "
            f"val_rmse={val_rmse:.2f} | val_rmse_day={val_rmse_day:.2f} | "
            f"lr={epoch_out['lr']:.2e} | "
            f"time={epoch_out['epoch_seconds']:.1f}s | "
            f"best={best_val_rmse_day:.2f} (ep {best_epoch}) | "
            f"bad={bad_epochs}/{patience}"
        )

        if bad_epochs >= patience:
            print(f"Early stop at epoch {epoch}. Best: ep {best_epoch}, val_rmse_day={best_val_rmse_day:.2f}")
            break

    total_train_seconds = float(time.time() - t_train0)

    assert best_state is not None, "Training finished with no improvement recorded."
    model.load_state_dict(best_state)

    final_val = eval_model(model, val_loader, normalizer, day_threshold, device, fusion=fusion)

    return {
        "model": model,
        "best_epoch": int(best_epoch),
        "best_val_rmse_day": float(best_val_rmse_day),
        "final_val": final_val,
        "train_log": train_log,
        "train_seconds_total": total_train_seconds,
    }

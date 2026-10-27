"""Canonical selection of a tuned Optuna run for a (site, horizon, seed) cell.

Why this exists
---------------
A cell can hold more than one completed Optuna run for the same seed: elpaso h6
seed 42 has two, with hidden_t 256 (val_rmse_day 260.2) and hidden_t 128
(250.3). Before this module each consumer broke the tie on its own, and in
opposite directions:

  12_variance_net.py, 08_sgld.py    return the FIRST match in sorted order
  08b_deep_ensemble.py, 08f_...py   keep the LAST match in sorted order

So the variance network and SGLD warm-started from the WORSE model while the
deep ensemble used the better one, and 08f then tried to load a variance net
trained on a 256-wide model into a 128-wide one and died on every parameter.

The rule here is the same criterion the search itself optimises: lowest
validation RMSE wins, with the newer run breaking an exact tie. Sort order is
an accident of file naming and is never the criterion.
"""
from __future__ import annotations

import json
from pathlib import Path


def _val_rmse_day(meta: dict) -> float | None:
    fv = (meta.get("best_model") or {}).get("final_val") or {}
    v = fv.get("rmse_day")
    return float(v) if isinstance(v, (int, float)) else None


def scan_cell_runs(runs_root: Path, site: str, horizon_hours: float, seed=None) -> list[dict]:
    """Completed runs for the cell that have best_params and a checkpoint on disk.

    Each entry: {run_dir, name, seed, best_params, ckpt, val_rmse_day}.
    """
    found = []
    for d in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            m = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("site") != site:
            continue
        if m.get("temporal", {}).get("horizon_hours") != horizon_hours:
            continue
        if seed is not None and m.get("seed") != seed:
            continue
        bp = m.get("optuna", {}).get("best_params")
        ck = (m.get("best_model") or {}).get("best_ckpt_path")
        if not bp or not ck or not Path(ck).exists():
            continue
        found.append({"run_dir": d, "name": d.name, "seed": int(m["seed"]),
                      "best_params": bp, "ckpt": ck, "val_rmse_day": _val_rmse_day(m)})
    return found


def pick_best(runs: list[dict], label: str = "") -> dict | None:
    """Lowest validation RMSE; newest run breaks a tie. Announces a real tie-break."""
    if not runs:
        return None
    if len(runs) > 1:
        scored = [r for r in runs if r["val_rmse_day"] is not None]
        if scored:
            best = min(scored, key=lambda r: (r["val_rmse_day"], r["name"]))
        else:
            best = runs[-1]
        losers = [r for r in runs if r is not best]
        print(f"  [run_select] {label}: {len(runs)} runs for seed {best['seed']}, "
              f"keeping {best['name']} (val_rmse_day={best['val_rmse_day']})")
        for r in losers:
            print(f"  [run_select]   discarding {r['name']} (val_rmse_day={r['val_rmse_day']})")
        return best
    return runs[0]


def best_members(runs_root: Path, site: str, horizon_hours: float) -> dict:
    """{seed: (best_params, ckpt)} using the canonical rule, one entry per seed."""
    by_seed: dict[int, list] = {}
    for r in scan_cell_runs(runs_root, site, horizon_hours):
        by_seed.setdefault(r["seed"], []).append(r)
    out = {}
    for seed, runs in sorted(by_seed.items()):
        b = pick_best(runs, label=f"{site} h{horizon_hours:g}")
        if b:
            out[seed] = (b["best_params"], b["ckpt"])
    return out

"""Locks the canonical rule for choosing a cell's tuned run.

Run from project root:
    .venv/bin/python -m pytest tests/test_run_select.py -v

Why these exist. A cell can hold two completed Optuna runs for one seed, and
before 2026-10-27 each consumer broke that tie on its own: the variance network
and SGLD kept the FIRST match in sorted order, the deep ensemble and the
combined band kept the LAST. For elpaso h6 seed 42 the two runs differ in
architecture (hidden_t 256 against 128), so the variance net ended up fitted to
a model nobody else used and the combined band died on a shape mismatch across
every parameter.

Every test builds its own fixture on disk; none reads the real runs/ tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from solar_uq.run_select import best_members, pick_best, scan_cell_runs  # noqa: E402


def _run(root: Path, name: str, *, seed: int, site="elpaso", horizon=6.0,
         hidden_t=128, val_rmse_day=250.0, with_ckpt=True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    ck = d / "best_model.pt"
    if with_ckpt:
        ck.write_bytes(b"not-a-real-checkpoint")
    (d / "summary.json").write_text(json.dumps({
        "site": site, "seed": seed,
        "temporal": {"horizon_hours": horizon},
        "optuna": {"best_params": {"hidden_t": hidden_t}},
        "best_model": {"best_ckpt_path": str(ck),
                       "final_val": {"rmse_day": val_rmse_day}},
    }))
    return d


def test_lower_validation_rmse_wins_not_sort_order(tmp_path):
    """The real elpaso h6 seed 42 case: the better run sorts FIRST by name."""
    _run(tmp_path, "cell_seed42_20260806_131802", seed=42, hidden_t=256, val_rmse_day=260.2)
    _run(tmp_path, "cell_seed42_20260809_234130", seed=42, hidden_t=128, val_rmse_day=250.3)
    members = best_members(tmp_path, "elpaso", 6.0)
    bp, ck = members[42]
    assert bp["hidden_t"] == 128, "picked the worse run; sort order is not the criterion"
    assert "20260809" in ck


def test_better_run_wins_when_it_sorts_last(tmp_path):
    """Mirror image, so the test cannot pass by preferring 'first' or 'last'."""
    _run(tmp_path, "cell_seed7_20260101_000000", seed=7, hidden_t=256, val_rmse_day=300.0)
    _run(tmp_path, "cell_seed7_20270101_000000", seed=7, hidden_t=128, val_rmse_day=200.0)
    bp, _ = best_members(tmp_path, "elpaso", 6.0)[7]
    assert bp["hidden_t"] == 128


def test_exact_tie_breaks_to_the_newer_run(tmp_path):
    _run(tmp_path, "cell_seed1_20260101_000000", seed=1, hidden_t=64, val_rmse_day=111.0)
    _run(tmp_path, "cell_seed1_20270101_000000", seed=1, hidden_t=192, val_rmse_day=111.0)
    bp, _ = best_members(tmp_path, "elpaso", 6.0)[1]
    assert bp["hidden_t"] == 192


def test_runs_without_a_checkpoint_are_not_members(tmp_path):
    _run(tmp_path, "cell_seed13_20260101_000000", seed=13, val_rmse_day=100.0, with_ckpt=False)
    _run(tmp_path, "cell_seed42_20260101_000000", seed=42, val_rmse_day=500.0)
    members = best_members(tmp_path, "elpaso", 6.0)
    assert sorted(members) == [42], "a recorded checkpoint that is absent must not be selected"


def test_a_missing_checkpoint_does_not_mask_a_usable_duplicate(tmp_path):
    """Best-scoring run has no checkpoint: fall back to the usable one, not to nothing."""
    _run(tmp_path, "cell_seed7_20260101_000000", seed=7, hidden_t=64,
         val_rmse_day=10.0, with_ckpt=False)
    _run(tmp_path, "cell_seed7_20260202_000000", seed=7, hidden_t=128, val_rmse_day=99.0)
    bp, _ = best_members(tmp_path, "elpaso", 6.0)[7]
    assert bp["hidden_t"] == 128


def test_other_cells_and_sites_are_not_mixed_in(tmp_path):
    _run(tmp_path, "a_seed42_20260101_000000", seed=42, site="elpaso", horizon=6.0)
    _run(tmp_path, "b_seed42_20260101_000000", seed=42, site="elpaso", horizon=1.0)
    _run(tmp_path, "c_seed42_20260101_000000", seed=42, site="uniandes", horizon=6.0)
    assert len(scan_cell_runs(tmp_path, "elpaso", 6.0)) == 1
    assert len(scan_cell_runs(tmp_path, "elpaso", 1.0)) == 1
    assert len(scan_cell_runs(tmp_path, "uniandes", 6.0)) == 1


def test_runs_missing_a_validation_score_still_resolve(tmp_path):
    """Older summaries have no final_val; the rule must not crash on them."""
    d = _run(tmp_path, "cell_seed1_20260101_000000", seed=1, hidden_t=64)
    m = json.loads((d / "summary.json").read_text())
    del m["best_model"]["final_val"]
    (d / "summary.json").write_text(json.dumps(m))
    _run(tmp_path, "cell_seed1_20260202_000000", seed=1, hidden_t=192, val_rmse_day=400.0)
    bp, _ = best_members(tmp_path, "elpaso", 6.0)[1]
    assert bp["hidden_t"] == 192, "a scored run must beat an unscored one"


def test_empty_and_missing_roots_are_not_errors(tmp_path):
    assert best_members(tmp_path / "does_not_exist", "elpaso", 6.0) == {}
    assert scan_cell_runs(tmp_path, "elpaso", 6.0) == []
    assert pick_best([]) is None

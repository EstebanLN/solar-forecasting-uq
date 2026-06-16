#!/usr/bin/env python
"""
compare_fusion_vs_satellite.py — Compare fusion vs satellite-only runs.

Loads summary.json files from runs/resnet_lstm_optuna/, runs/graphsage_lstm_optuna/,
runs/fusion_resnet_lstm/, and runs/fusion_graphsage_lstm/, then produces:

  results/fusion_comparison.csv   — tidy table per arch/site/horizon/modality
  results/fusion_comparison.tex   — LaTeX booktabs table with ΔSkill column

Only completed runs (those with a summary.json containing final_test) are included.

Usage (from project root):
    python scripts/compare_fusion_vs_satellite.py
    python scripts/compare_fusion_vs_satellite.py --runs_root runs/ --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


ARCH_DIRS = {
    "ResNetLSTM":          "resnet_lstm_optuna",
    "FusionResNetLSTM":    "fusion_resnet_lstm",
    "GraphSAGE_LSTM":      "graphsage_lstm_optuna",
    "FusionGraphSAGE_LSTM": "fusion_graphsage_lstm",
}

SATELLITE_ARCH = {
    "FusionResNetLSTM":    "ResNetLSTM",
    "FusionGraphSAGE_LSTM": "GraphSAGE_LSTM",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion vs satellite-only comparison table")
    p.add_argument("--runs_root", default=None,
                   help="Root directory containing run subdirs (default: PROJECT_ROOT/runs)")
    p.add_argument("--out",       default=None,
                   help="Output directory (default: PROJECT_ROOT/results)")
    p.add_argument("--metric",    default="rmse_day",
                   choices=["rmse_day", "rmse", "mae_day", "mae"],
                   help="Primary metric to report")
    p.add_argument("--skill_ref", default="persistence",
                   help="Baseline reference for skill (default: persistence)")
    return p.parse_args()


def _load_runs(runs_root: Path) -> list[dict]:
    """Scan all arch dirs under runs_root and collect summary rows."""
    rows: list[dict] = []
    for arch, subdir in ARCH_DIRS.items():
        arch_dir = runs_root / subdir
        if not arch_dir.exists():
            continue
        for run_dir in arch_dir.iterdir():
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            bm = s.get("best_model", {})
            ft = bm.get("final_test")
            if ft is None:
                continue

            temporal = s.get("temporal", {})
            optuna   = s.get("optuna",   {})
            rows.append({
                "arch":          arch,
                "fusion":        arch.startswith("Fusion"),
                "site":          s.get("site", "?"),
                "hours_ahead":   temporal.get("horizon_hours", float("nan")),
                "seed":          s.get("seed", -1),
                "run_name":      s.get("run_name", run_dir.name),
                "rmse":          ft.get("rmse"),
                "rmse_day":      ft.get("rmse_day"),
                "mae":           ft.get("mae"),
                "mae_day":       ft.get("mae_day"),
                "skill_day":     ft.get("skill_day_vs_persistence"),
                "n_params":      bm.get("n_params"),
                "best_epoch":    bm.get("best_epoch"),
                "val_rmse_day":  optuna.get("best_value_val_rmse_day"),
            })
    return rows


def _build_comparison(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a tidy table with ΔMetric and ΔSkill_day columns."""
    # Mean over seeds
    grp = df.groupby(["arch", "site", "hours_ahead"], as_index=False)[
        [metric, "skill_day", "n_params"]
    ].mean()

    rows = []
    for (site, h), sub in grp.groupby(["site", "hours_ahead"]):
        sat_rows = sub[~sub["arch"].str.startswith("Fusion")]
        fus_rows = sub[sub["arch"].str.startswith("Fusion")]

        for _, frow in fus_rows.iterrows():
            sat_arch = SATELLITE_ARCH.get(frow["arch"])
            sat_match = sat_rows[sat_rows["arch"] == sat_arch]
            if sat_match.empty:
                d_metric   = float("nan")
                d_skill    = float("nan")
                sat_metric = float("nan")
            else:
                sat_r      = sat_match.iloc[0]
                sat_metric = sat_r[metric]
                d_metric   = frow[metric] - sat_metric          # negative = improvement
                d_skill    = frow["skill_day"] - sat_r["skill_day"]

            rows.append({
                "arch":           frow["arch"],
                "sat_arch":       sat_arch or "?",
                "site":           site,
                "hours_ahead":    h,
                f"{metric}":      round(frow[metric], 2) if pd.notna(frow[metric]) else None,
                f"{metric}_sat":  round(sat_metric, 2)   if pd.notna(sat_metric)   else None,
                f"Δ{metric}":     round(d_metric, 2)     if pd.notna(d_metric)     else None,
                "skill_day":      round(frow["skill_day"], 4) if pd.notna(frow["skill_day"]) else None,
                "Δskill_day":     round(d_skill, 4)      if pd.notna(d_skill)      else None,
                "n_params":       int(frow["n_params"])   if pd.notna(frow["n_params"]) else None,
            })

        for _, srow in sat_rows.iterrows():
            rows.append({
                "arch":           srow["arch"],
                "sat_arch":       srow["arch"],
                "site":           site,
                "hours_ahead":    h,
                f"{metric}":      round(srow[metric], 2) if pd.notna(srow[metric]) else None,
                f"{metric}_sat":  None,
                f"Δ{metric}":     None,
                "skill_day":      round(srow["skill_day"], 4) if pd.notna(srow["skill_day"]) else None,
                "Δskill_day":     None,
                "n_params":       int(srow["n_params"]) if pd.notna(srow["n_params"]) else None,
            })

    return pd.DataFrame(rows).sort_values(["site", "hours_ahead", "arch"])


def _to_latex(df: pd.DataFrame, metric: str) -> str:
    col_order = ["arch", "site", "hours_ahead", metric, f"Δ{metric}", "skill_day", "Δskill_day"]
    col_order = [c for c in col_order if c in df.columns]
    sub = df[col_order].copy()
    sub.columns = [c.replace("_", r"\_") for c in sub.columns]
    return sub.to_latex(index=False, na_rep="—", escape=False,
                        float_format="%.3f",
                        caption="Fusion vs satellite-only performance (test set, mean over seeds).",
                        label="tab:fusion_comparison")


def main() -> None:
    args    = parse_args()
    runs_root = Path(args.runs_root) if args.runs_root else PROJECT_ROOT / "runs"
    out_dir   = Path(args.out)       if args.out       else PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_runs(runs_root)
    if not rows:
        print(f"No completed runs found under {runs_root}. Nothing to compare.")
        return

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs from {runs_root}")
    print(df.groupby(["arch", "site"])["hours_ahead"].count().to_string())

    comp = _build_comparison(df, args.metric)

    csv_path = out_dir / "fusion_comparison.csv"
    tex_path = out_dir / "fusion_comparison.tex"
    comp.to_csv(csv_path, index=False)
    tex_path.write_text(_to_latex(comp, args.metric), encoding="utf-8")

    print(f"\n--- Fusion vs Satellite ({args.metric}) ---")
    print(comp.to_string(index=False))
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {tex_path}")


if __name__ == "__main__":
    main()

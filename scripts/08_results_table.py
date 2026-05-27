#!/usr/bin/env python
"""
08_results_table.py — Aggregate all run results into a comparison table.

Usage (from project root):
    .venv/bin/python scripts/08_results_table.py
    .venv/bin/python scripts/08_results_table.py --latex
    .venv/bin/python scripts/08_results_table.py --out results/summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT    = PROJECT_ROOT / "runs"

HORIZON_LABEL = {1.0: "1 h", 2.0: "2 h", 3.0: "3 h", 6.0: "6 h"}

MODEL_ORDER = [
    "Persistence",
    "SARIMA",
    "ResNet+LSTM",
    "GraphSAGE+LSTM",
    "MLP (Optuna)",
    "ResNet+LSTM (Optuna)",
    "GraphSAGE+LSTM (Optuna)",
    "ResNet+LSTM (Optuna v2)",
    "GraphSAGE+LSTM (Optuna v2)",
]

# Expected seeds per model family (for progress tracking)
EXPECTED_SEEDS: dict[str, int] = {
    "ResNet+LSTM":                 5,
    "GraphSAGE+LSTM":              5,
    "MLP (Optuna)":                4,
    "ResNet+LSTM (Optuna)":        4,
    "GraphSAGE+LSTM (Optuna)":     4,
    "ResNet+LSTM (Optuna v2)":     4,
    "GraphSAGE+LSTM (Optuna v2)":  4,
}


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────

def _load_nn_runs(runs_dir: Path, model_label: str) -> list[dict]:
    """Load summary.json files from a neural-net runs directory.

    Optuna runs store metrics under 'best_model'; baselines under 'model'.
    """
    records: list[dict] = []
    if not runs_dir.exists():
        return records

    for run_dir in sorted(runs_dir.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        with sj.open() as f:
            d = json.load(f)

        ms   = d.get("best_model") or d.get("model") or {}
        test = ms.get("final_test")
        if not test:
            continue

        temporal  = d.get("temporal", {})
        pers_test = d.get("baselines", {}).get("persistence_test", {})

        optuna = d.get("optuna", {})
        records.append({
            "model":         model_label,
            "site":          d.get("site"),
            "horizon_hours": temporal.get("horizon_hours"),
            "seed":          d.get("seed"),
            "rmse":          test.get("rmse"),
            "rmse_day":      test.get("rmse_day"),
            "mae_day":       test.get("mae_day"),
            "skill":         test.get("skill_vs_persistence"),
            "skill_day":     test.get("skill_day_vs_persistence"),
            "pers_rmse_day": pers_test.get("rmse_day"),
            "best_trial":    optuna.get("best_trial_number"),
        })
    return records


def _load_sarima_runs(runs_dir: Path) -> list[dict]:
    """Load SARIMA results (one file contains all horizons for a site)."""
    records: list[dict] = []
    if not runs_dir.exists():
        return records

    for run_dir in sorted(runs_dir.iterdir()):
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue
        with sj.open() as f:
            d = json.load(f)

        site = d.get("site")
        for key, res in d.get("results", {}).items():
            tm   = res.get("test_metrics", {})
            pers = res.get("persistence_test", {})
            records.append({
                "model":         "SARIMA",
                "site":          site,
                "horizon_hours": float(key),
                "seed":          0,
                "rmse":          tm.get("rmse"),
                "rmse_day":      tm.get("rmse_day"),
                "skill":         tm.get("skill_vs_persistence"),
                "skill_day":     tm.get("skill_day_vs_persistence"),
                # SARIMA evaluates on hourly sub-samples; persistence differs
                # from full-resolution neural-model persistence — kept separate
                "pers_rmse_day": None,
            })
    return records


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

def _aggregate(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return (
        df.groupby(["model", "site", "horizon_hours"], sort=False)
        .agg(
            rmse_mean      = ("rmse",          "mean"),
            rmse_std       = ("rmse",          "std"),
            rmse_day_mean  = ("rmse_day",      "mean"),
            rmse_day_std   = ("rmse_day",      "std"),
            skill_day_mean = ("skill_day",     "mean"),
            skill_day_std  = ("skill_day",     "std"),
            pers_rmse_day  = ("pers_rmse_day", "mean"),
            n_seeds        = ("seed",          "count"),
        )
        .reset_index()
    )


def _persistence_rows(all_records: list[dict]) -> pd.DataFrame:
    """Derive one Persistence row per (site, horizon) from embedded baselines."""
    df = pd.DataFrame(all_records)
    if df.empty:
        return pd.DataFrame()
    grp = (
        df[df["pers_rmse_day"].notna()]
        .groupby(["site", "horizon_hours"])["pers_rmse_day"]
        .mean()
        .reset_index()
    )
    rows = []
    for _, r in grp.iterrows():
        rows.append({
            "model":         "Persistence",
            "site":          r["site"],
            "horizon_hours": r["horizon_hours"],
            "rmse_day_mean": r["pers_rmse_day"],
            "rmse_day_std":  np.nan,
            "skill_day_mean": 0.0,
            "skill_day_std": np.nan,
            "pers_rmse_day": r["pers_rmse_day"],
            "n_seeds":       1,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

def _fmt(mean, std=None, fmt: str = ".1f", nan_str: str = "—") -> str:
    if pd.isna(mean):
        return nan_str
    s = f"{mean:{fmt}}"
    if std is not None and not pd.isna(std) and std > 0:
        s += f" ±{std:{fmt}}"
    return s


def _seeds_status(n: int, model: str) -> str:
    exp = EXPECTED_SEEDS.get(model)
    if exp is None:
        return "—"
    return f"{n}/{exp} ✓" if n >= exp else f"{n}/{exp}"


# ──────────────────────────────────────────────────────────────────────
# Console table
# ──────────────────────────────────────────────────────────────────────

def _print_console(table: pd.DataFrame, sites: list, horizons: list) -> None:
    col = dict(model=28, rmse=22, skill=12, seeds=8)
    header = (
        f"  {'Model':<{col['model']}} "
        f"{'RMSE_day (W/m²)':>{col['rmse']}} "
        f"{'Skill_day':>{col['skill']}} "
        f"{'Seeds':>{col['seeds']}}"
    )
    sep = "  " + "─" * (len(header) - 2)

    for site in sites:
        bar = "═" * (len(header) - 2)
        print(f"\n  {bar}")
        print(f"  {site.upper()}")
        print(f"  {bar}")

        for h in horizons:
            hl  = HORIZON_LABEL.get(h, f"{h:.0f}h")
            sub = table[(table["site"] == site) & (table["horizon_hours"] == h)]
            if sub.empty:
                continue

            print(f"\n    ── {hl} ──────────────────────────────────────────")
            print(header)
            print(sep)

            for model in MODEL_ORDER:
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                r = row.iloc[0]

                rmse_s  = _fmt(r.get("rmse_day_mean"), r.get("rmse_day_std"), ".1f")
                skill_s = _fmt(r.get("skill_day_mean"), r.get("skill_day_std"), ".3f")
                seeds_s = _seeds_status(int(r.get("n_seeds", 0)), model)

                print(
                    f"  {model:<{col['model']}} "
                    f"{rmse_s:>{col['rmse']}} "
                    f"{skill_s:>{col['skill']}} "
                    f"{seeds_s:>{col['seeds']}}"
                )

    print()


# ──────────────────────────────────────────────────────────────────────
# LaTeX table
# ──────────────────────────────────────────────────────────────────────

def _print_latex(table: pd.DataFrame, sites: list, horizons: list) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llp{4.5cm}rr}",
        r"\toprule",
        r"Site & Horizon & Model "
        r"& RMSE$_\text{day}$ (W/m$^2$) "
        r"& Skill$_\text{day}$ \\",
        r"\midrule",
    ]

    for site in sites:
        first_site = True
        for h in sorted(horizons):
            hl  = HORIZON_LABEL.get(h, f"{h:.0f}h")
            sub = table[(table["site"] == site) & (table["horizon_hours"] == h)]
            if sub.empty:
                continue
            first_h = True
            for model in MODEL_ORDER:
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                r = row.iloc[0]
                rmse_s  = _fmt(r.get("rmse_day_mean"), r.get("rmse_day_std"), ".1f")
                skill_s = _fmt(r.get("skill_day_mean"), r.get("skill_day_std"), ".3f")
                sc = site if (first_site and first_h) else ""
                hc = hl   if first_h else ""
                lines.append(
                    f"{sc} & {hc} & {model} & ${rmse_s}$ & ${skill_s}$ \\\\"
                )
                first_site = False
                first_h    = False
        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Test-set forecasting results. "
        r"RMSE$_\text{day}$ and Skill$_\text{day}$ are daytime-only metrics "
        r"(GHI\,$\geq$\,20\,W/m$^2$). "
        r"Skill$_\text{day}$ is relative to naive persistence. "
        r"Results averaged over available random seeds ($\pm$std).}",
        r"\label{tab:results}",
        r"\end{table}",
    ]

    print("\n% ── LaTeX ──────────────────────────────────────────────────────")
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# HPO progress summary
# ──────────────────────────────────────────────────────────────────────

def _print_progress(all_records: list[dict]) -> None:
    df = pd.DataFrame(all_records)
    if df.empty:
        return

    optuna_models = [
        "MLP (Optuna)",
        "ResNet+LSTM (Optuna)",
        "GraphSAGE+LSTM (Optuna)",
        "ResNet+LSTM (Optuna v2)",
        "GraphSAGE+LSTM (Optuna v2)",
    ]
    sub = df[df["model"].isin(optuna_models)]
    if sub.empty:
        return

    print("\n  HPO Progress")
    print("  " + "─" * 60)
    grp = sub.groupby(["model", "site", "horizon_hours"])["seed"].count().reset_index()
    grp.columns = ["model", "site", "horizon_hours", "done"]
    grp["expected"] = grp["model"].map(EXPECTED_SEEDS)
    grp["hl"] = grp["horizon_hours"].map(lambda h: HORIZON_LABEL.get(h, f"{h:.0f}h"))

    for _, r in grp.sort_values(["model", "site", "horizon_hours"]).iterrows():
        bar = "✓" if r["done"] >= r["expected"] else "·" * int(r["done"])
        print(
            f"  {r['model']:<28} {r['site']:<10} {r['hl']:<5} "
            f"{r['done']}/{r['expected']}  {bar}"
        )

    total_done = int(sub.shape[0])
    total_exp  = sum(EXPECTED_SEEDS[m] for m in optuna_models) * 2 * len(
        sub["horizon_hours"].unique()
    )
    print(f"\n  Total HPO: {total_done}/{total_exp} runs completed\n")


# ──────────────────────────────────────────────────────────────────────
# Markdown output
# ──────────────────────────────────────────────────────────────────────

def _save_flat_csv(records: list[dict], path: Path) -> None:
    """One row per run: arch, site, horizon, seed, rmse_day, skill_day, mae_day, best_trial."""
    rows = []
    for r in records:
        if r.get("model") in ("Persistence",):
            continue
        rows.append({
            "arch":        r["model"],
            "site":        r.get("site"),
            "horizon":     r.get("horizon_hours"),
            "seed":        r.get("seed"),
            "rmse_day":    r.get("rmse_day"),
            "skill_day":   r.get("skill_day"),
            "mae_day":     r.get("mae_day"),
            "best_trial":  r.get("best_trial"),
        })
    df = pd.DataFrame(rows).sort_values(["arch", "site", "horizon", "seed"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")


def _save_markdown(table: pd.DataFrame, sites: list, horizons: list, path: Path) -> None:
    lines = ["# Results Summary\n"]
    for site in sites:
        lines.append(f"\n## {site.upper()}\n")
        for h in horizons:
            hl  = HORIZON_LABEL.get(h, f"{h:.0f}h")
            sub = table[(table["site"] == site) & (table["horizon_hours"] == h)]
            if sub.empty:
                continue
            lines.append(f"\n### {hl}\n")
            lines.append("| Model | RMSE_day (W/m²) | Skill_day | Seeds |")
            lines.append("|---|---|---|---|")
            for model in MODEL_ORDER:
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                r = row.iloc[0]
                rmse_s  = _fmt(r.get("rmse_day_mean"), r.get("rmse_day_std"), ".1f")
                skill_s = _fmt(r.get("skill_day_mean"), r.get("skill_day_std"), ".3f")
                seeds_s = _seeds_status(int(r.get("n_seeds", 0)), model)
                lines.append(f"| {model} | {rmse_s} | {skill_s} | {seeds_s} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Results comparison table")
    ap.add_argument("--latex",       action="store_true", help="Print LaTeX table")
    ap.add_argument("--no-progress", action="store_true", help="Skip HPO progress section")
    ap.add_argument("--flat",        action="store_true",
                    help="Also save per-seed flat CSV to results/master.csv")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "summary.csv"),
                    help="Save aggregated CSV (default: results/summary.csv)")
    args = ap.parse_args()

    all_records: list[dict] = []
    all_records += _load_nn_runs(RUNS_ROOT / "resnet_lstm",                "ResNet+LSTM")
    all_records += _load_nn_runs(RUNS_ROOT / "graphsage_lstm",             "GraphSAGE+LSTM")
    all_records += _load_nn_runs(RUNS_ROOT / "mlp_optuna",                 "MLP (Optuna)")
    all_records += _load_nn_runs(RUNS_ROOT / "resnet_lstm_optuna",         "ResNet+LSTM (Optuna)")
    all_records += _load_nn_runs(RUNS_ROOT / "graphsage_lstm_optuna",      "GraphSAGE+LSTM (Optuna)")
    all_records += _load_nn_runs(RUNS_ROOT / "resnet_lstm_optuna_v2",      "ResNet+LSTM (Optuna v2)")
    all_records += _load_nn_runs(RUNS_ROOT / "graphsage_lstm_optuna_v2",   "GraphSAGE+LSTM (Optuna v2)")
    all_records += _load_sarima_runs(RUNS_ROOT / "sarima")

    if not all_records:
        print("No runs found under", RUNS_ROOT)
        return

    agg   = _aggregate(all_records)
    pers  = _persistence_rows(all_records)
    table = pd.concat([agg, pers], ignore_index=True) if not pers.empty else agg

    sites    = sorted(table["site"].dropna().unique())
    horizons = sorted(table["horizon_hours"].dropna().unique())

    _print_console(table, sites, horizons)

    if not args.no_progress:
        _print_progress(all_records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    md_out = out.with_suffix(".md")
    _save_markdown(table, sites, horizons, md_out)

    print(f"Saved → {out}")
    print(f"Saved → {md_out}")

    if args.flat:
        flat_out = out.parent / "master.csv"
        _save_flat_csv(all_records, flat_out)
        print(f"Saved → {flat_out}")

    if args.latex:
        _print_latex(table, sites, horizons)


if __name__ == "__main__":
    main()

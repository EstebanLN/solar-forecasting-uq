"""Render the SGLD Markov-chain diagnostic + performance as a single shareable
figure (advisor's request). Three panels:
  (A) chain trace: per-epoch daytime val RMSE over burn-in + sampling, log-y,
      with the persistence and warm-start-backbone references — shows the drift;
  (B) autocorrelation of the collected-sample chain, with ESS / Geweke z;
  (C) test performance (RMSE_day) of the SGLD ensemble vs. backbone vs.
      persistence — shows the estimator is far worse than its own starting point.

Usage:
    .venv/bin/python scripts/09c_sgld_markov_figure.py <sgld_run_dir> [--out path.png]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
mk = import_module("09b_sgld_markov_diagnostics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=str(ROOT / "docs" / "sgld_markov_diagnostic.png"))
    a = ap.parse_args()
    rd = Path(a.run_dir)
    s = json.load(open(rd / "summary.json"))
    sg = s.get("sgld", {})
    burn = int(sg.get("burn_in", 40)); every = int(sg.get("sample_every", 3))
    pers = s.get("baselines", {}).get("persistence_test", {}).get("rmse_day")
    ft = s.get("best_model", {}).get("final_test", {})

    tl = json.load(open(rd / "train_log.json"))
    rows = tl if isinstance(tl, list) else tl.get("train_log", [])
    chain = np.array([r["val_rmse_day"] for r in rows if "val_rmse_day" in r], float)
    ep = np.arange(1, len(chain) + 1)
    sampled = chain[burn::every]                       # collected posterior samples
    dg_full = mk.diagnose(chain, "full")
    dg_samp = mk.diagnose(sampled, "sampled")
    backbone = chain[0]                                # warm-start starting RMSE_day

    C = {"drift": "#EE6677", "samp": "#4477AA", "ref": "#999999", "ok": "#229988"}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (A) trace ---------------------------------------------------------------
    ax[0].axvspan(1, burn, color="#000000", alpha=0.06, label="burn-in")
    ax[0].plot(ep, chain, color=C["drift"], lw=1.4, marker="o", ms=2.6)
    ax[0].plot(ep[burn::every], sampled, "o", color=C["samp"], ms=5,
               label=f"collected samples ($n={len(sampled)}$)", zorder=5)
    if pers:
        ax[0].axhline(pers, ls="--", color=C["ref"], lw=1.2, label=f"persistence ({pers:.0f})")
    ax[0].axhline(backbone, ls=":", color=C["ok"], lw=1.6, label=f"warm-start backbone ({backbone:.0f})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("SGLD epoch"); ax[0].set_ylabel("daytime val RMSE (W/m²)")
    ax[0].set_title("(A) Chain trace — drifts away from the mode", fontsize=10, loc="left")
    ax[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # (B) autocorrelation -----------------------------------------------------
    rho = mk.autocorr(sampled, max_lag=min(20, len(sampled) - 1))
    ax[1].bar(range(len(rho)), rho, color=C["samp"], width=0.7)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("lag"); ax[1].set_ylabel("autocorrelation")
    ax[1].set_title("(B) Sample autocorrelation", fontsize=10, loc="left")
    txt = (f"ESS = {dg_samp['ess']:.1f} / {len(sampled)}\n"
           f"lag-1 $\\rho$ = {dg_samp['lag1_autocorr']:.2f}\n"
           f"Geweke $z$ = {dg_samp['geweke_z']:.1f}\n"
           f"stationary: {'yes' if dg_samp['stationary_|z|<2'] else 'NO'}")
    ax[1].text(0.96, 0.94, txt, transform=ax[1].transAxes, ha="right", va="top",
               fontsize=8.5, family="monospace",
               bbox=dict(boxstyle="round", fc="white", ec="#999999"))

    # (C) performance ---------------------------------------------------------
    labels = ["Persistence", "Backbone\n(Adam)", "SGLD\nensemble"]
    vals = [pers, backbone, ft.get("rmse_day", np.nan)]
    cols = [C["ref"], C["ok"], C["drift"]]
    bars = ax[2].bar(labels, vals, color=cols, edgecolor="black", lw=0.6)
    ax[2].set_ylabel("test daytime RMSE (W/m²)")
    ax[2].set_title("(C) Ensemble performance", fontsize=10, loc="left")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                   va="bottom", fontsize=9)
    sk = ft.get("skill_day_vs_persistence")
    if sk is not None:
        ax[2].text(0.04, 0.96, f"SGLD skill$_{{day}}$ = {sk:+.2f}\n(negative →\nworse than persistence)",
                   transform=ax[2].transAxes, ha="left", va="top", fontsize=8.5,
                   color=C["drift"])

    fig.suptitle(f"SGLD Markov-chain diagnostic — {s.get('site')} 6h, warm-start, "
                 f"{sg.get('total_epochs')} epochs ($\\varepsilon={sg.get('sgld_lr'):.0e}$)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"Saved → {a.out}")
    print("full-chain:", dg_full)
    print("sampled   :", dg_samp)


if __name__ == "__main__":
    main()

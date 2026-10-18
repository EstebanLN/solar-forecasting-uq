"""Diagnostic figures for the multi-chain SGLD run (4 chains x 2500 epochs).

Regenerated from the saved chain_*.npz, without re-running the chains.

Convergence is diagnosed on the PREDICTIVE FUNCTION (64 fixed validation inputs),
not on the RMSE: the RMSE is a scalar aggregate that can mask individual
predictions wandering apart while their errors compensate.

Palette: Okabe-Ito, validated for colour-vision deficiency (all-pairs dE OK).
Because the CVD separation falls in the 6-8 floor band, each chain also carries a
distinct line style (required secondary encoding).

Usage:  .venv/bin/python scripts/10b_sgld_multichain_figure.py
Outputs in docs/:  sgld_predictions.png, sgld_rhat_ess.png, sgld_rmse.png
"""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from solar_uq.mcmc_diag import split_rhat, ess_bulk

run = sorted(glob.glob("runs/graphsage_sgld_multichain/*/"))[-1]
meta = json.load(open(run + "meta.json")); summ = json.load(open(run + "summary.json"))
BURN = summ["protocol"]["burn_in"]; PERS = meta["persistence_val_rmse_day"]
chains = [np.load(f) for f in sorted(glob.glob(run + "chain_*.npz"))]
mon = np.stack([c["mon_preds"][BURN:] for c in chains])          # (4, 2000, 64)
rhat = np.asarray(split_rhat(mon)); ess = np.asarray(ess_bulk(mon))
rhat, ess = rhat[np.isfinite(rhat)], ess[np.isfinite(ess)]
ndraws = mon.shape[0] * mon.shape[1]
veps = chains[0]["val_eps"]; vt = np.stack([c["val_vals"] for c in chains])
ep = np.arange(BURN, BURN + mon.shape[1])

C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]      # Okabe-Ito (validated)
LS = ["-", "--", "-.", ":"]                            # secondary encoding
INK, MUTE, GRID = "#1a1a1a", "#6b6b6b", "#e8e8e8"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bdbdbd",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE})

def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

SUB = "Multi-chain SGLD (4 x 2500 epochs) - GraphSAGE, Uniandes, 6 h"

# ---- 1) Predictions vs physically possible range --------------------------
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.fill_between(ep, mon.min(axis=(0, 2)), mon.max(axis=(0, 2)), color="#c9c9c9",
                alpha=.55, lw=0, zorder=0, label="range over the 64 points (4 chains)")
for i in range(4):
    ax.plot(ep, mon[i].mean(1), color=C[i], ls=LS[i], lw=0.8, alpha=.95, label=f"chain {i}")
ax.axhspan(0, 1400, color="#2e7d32", alpha=.55, zorder=3,
           label="physically possible GHI (0-1400 W/m$^2$)")
ax.set_title(f"{SUB}\nThe predictive function becomes physically impossible",
             fontweight="bold", loc="left")
ax.set_xlabel("epoch"); ax.set_ylabel("mean prediction over the 64 points (W/m$^2$)")
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
tidy(ax); fig.savefig("docs/sgld_predictions.png", dpi=150, bbox_inches="tight", facecolor="white")

# ---- 2) R-hat and ESS -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
a = axes[0]; a.hist(ess, bins=18, color="#0072B2", alpha=.85)
a.set_title("ESS per monitored point", fontweight="bold", loc="left")
a.set_xlabel(f"effective sample size (ESS) - out of {ndraws:,} draws")
a.set_ylabel("number of points"); tidy(a)
b = axes[1]; b.hist(rhat, bins=18, color="#D55E00", alpha=.85)
b.axvline(1.1, color="#B00020", ls="--", lw=1.2, label="threshold 1.1")
b.set_xlim(0.99, max(1.25, rhat.max() * 1.02))
b.set_title(r"split-$\hat{R}$ per monitored point", fontweight="bold", loc="left")
b.set_xlabel(r"split-$\hat{R}$"); b.set_ylabel("number of points")
b.legend(frameon=False, fontsize=9, loc="upper left"); tidy(b)
fig.suptitle(f"{SUB} - diagnostics on the predictions, not on the RMSE",
             fontweight="bold", fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig("docs/sgld_rhat_ess.png", dpi=150, bbox_inches="tight", facecolor="white")

# ---- 3) RMSE_day (secondary reference) ------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i in range(4):
    ax.plot(veps, vt[i], color=C[i], ls=LS[i], lw=1.1, alpha=.95, label=f"chain {i}")
ax.axhline(PERS, color=INK, ls=(0, (4, 2)), lw=1.1, label=f"persistence = {PERS:.0f}")
ax.axvline(BURN, color=MUTE, ls=":", lw=1, label="end of burn-in")
ax.set_title(f"{SUB}\nval RMSE_day per chain (secondary reference)",
             fontweight="bold", loc="left")
ax.set_xlabel("epoch"); ax.set_ylabel("val RMSE_day (W/m$^2$)")
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2); tidy(ax)
fig.savefig("docs/sgld_rmse.png", dpi=150, bbox_inches="tight", facecolor="white")

print(f"R-hat max={rhat.max():.3f} | ESS min={ess.min():.1f} med={np.median(ess):.1f} "
      f"of {ndraws} | pred {mon.min():.0f}..{mon.max():.0f}")

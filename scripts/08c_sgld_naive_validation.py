"""Naive SGLD validation on a tractable posterior (Bayesian linear regression).

Daniel's request (reunión 2026-10-06): before running the full predictive SGLD
chains, *prove the SGLD implementation is correct* on something naive. This is
the gold-standard check: a problem whose posterior is known in closed form, so
we can verify that the SAME optimizer used for the nets (``src.solar_uq.sgld.SGLD``)
actually samples the correct distribution.

Model (Bayesian linear regression, Gaussian likelihood + Gaussian prior):
    y_i = x_i . beta + eps,   eps ~ N(0, sigma^2)        (sigma known)
    beta ~ N(0, (1/alpha) I)                              (prior precision alpha)
Potential sampled by SGLD  (p(beta|D) propto exp(-U)):
    U(beta) = (1/(2 sigma^2)) ||X beta - y||^2  +  (alpha/2) ||beta||^2
so with a FULL-BATCH gradient of  loss = ||X beta - y||^2 / (2 sigma^2)  and the
optimizer's weight_decay = alpha, the SGLD update is exactly unadjusted Langevin
on U. The posterior is Gaussian and known analytically:
    Sigma_post = (X^T X / sigma^2 + alpha I)^{-1}
    mu_post    = Sigma_post X^T y / sigma^2

We run M=4 chains from dispersed inits, discard burn-in, and check:
  * split-R-hat per parameter  (mixing / stationarity; want < 1.01)
  * bulk-ESS per parameter     (effective independent draws; Geyer estimator)
  * empirical posterior mean vs mu_post  (z-score = |emp-mu| / post_sd)
  * empirical posterior sd  vs diag(Sigma_post)^(1/2)  (relative error)

If these pass, the sampler is correct, and the divergence on the big
over-parameterised nets is a property of THOSE loss surfaces, not of the code.

Usage:
    .venv/bin/python scripts/08c_sgld_naive_validation.py                 # defaults: 2500 epochs, 4 chains
    .venv/bin/python scripts/08c_sgld_naive_validation.py --minibatch 32  # also test stochastic-gradient (true SGLD) mode
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from solar_uq.sgld import SGLD  # the SAME optimizer used for the predictive nets
from solar_uq.mcmc_diag import split_rhat, ess_bulk


# ----------------------------------------------------------------------------- experiment
def _make_data(args):
    rng = np.random.default_rng(args.data_seed)
    N, D = args.n_data, args.n_dim
    # Standardized design; choose sigma so the likelihood Hessian X^T X / sigma^2
    # is ~I (well-conditioned, Hessian eigenvalues ~1+alpha). This keeps the
    # unadjusted-Langevin discretization bias small at a usable step size.
    sigma = args.sigma if args.sigma > 0 else float(np.sqrt(N))
    X = rng.standard_normal((N, D))
    X = (X - X.mean(0)) / X.std(0)
    beta_true = rng.standard_normal(D)
    y = X @ beta_true + sigma * rng.standard_normal(N)
    return X, y, sigma


def _analytic_posterior(X, y, sigma, alpha):
    D = X.shape[1]
    A = X.T @ X / sigma ** 2 + alpha * np.eye(D)
    Sigma_post = np.linalg.inv(A)
    mu_post = Sigma_post @ (X.T @ y) / sigma ** 2
    return mu_post, Sigma_post, np.sqrt(np.diag(Sigma_post))


def _run_chains(args, X, y, sigma, alpha, lr, lr_final, epochs, burn_in):
    """Run M SGLD chains; return samples (M, N_draws, D)."""
    N, D = X.shape
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    rng = np.random.default_rng(args.data_seed + 777)
    lrf = lr_final if (lr_final is not None and lr_final > 0) else None
    samples_per_chain = []
    for c in range(args.n_chains):
        torch.manual_seed(args.chain_seed0 + c)
        beta = torch.zeros(D, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            beta += 3.0 / np.sqrt(alpha) * torch.randn(D, dtype=torch.float64)  # dispersed init
        opt = SGLD([beta], lr=lr, weight_decay=alpha,
                   lr_final=lrf, total_steps=epochs if lrf else None,
                   schedule="poly", gamma=args.gamma)
        draws = []
        idx = np.arange(N)
        for t in range(epochs):
            def closure():
                opt.zero_grad()
                if args.minibatch and args.minibatch < N:
                    bi = rng.choice(idx, size=args.minibatch, replace=False)
                    resid = Xt[bi] @ beta - yt[bi]
                    loss = (N / args.minibatch) * 0.5 * (resid ** 2).sum() / sigma ** 2
                else:
                    resid = Xt @ beta - yt
                    loss = 0.5 * (resid ** 2).sum() / sigma ** 2
                loss.backward()
                return loss
            opt.step(closure)
            if t >= burn_in:
                draws.append(beta.detach().numpy().copy())
        samples_per_chain.append(np.array(draws))
    return np.stack(samples_per_chain, axis=0)


def run(args):
    N, D = args.n_data, args.n_dim
    alpha = args.prior_precision
    X, y, sigma = _make_data(args)
    mu_post, Sigma_post, post_sd = _analytic_posterior(X, y, sigma, alpha)

    total_steps = args.epochs
    chains = _run_chains(args, X, y, sigma, alpha, args.lr, args.lr_final,
                         total_steps, args.burn_in)        # (M, N_draws, D)

    rhat = split_rhat(chains)
    ess = ess_bulk(chains)
    flat = chains.reshape(-1, D)
    emp_mean = flat.mean(0)
    emp_sd = flat.std(0, ddof=1)
    z_mean = np.abs(emp_mean - mu_post) / post_sd          # posterior-sd units
    sd_relerr = np.abs(emp_sd - post_sd) / post_sd

    # ---- verdict ----
    ok_rhat = bool(np.all(rhat < 1.01))
    ok_mean = bool(np.all(z_mean < 0.15))                  # within 0.15 post-sd
    ok_sd = bool(np.all(sd_relerr < 0.10))                 # within 10%
    passed = ok_rhat and ok_mean and ok_sd

    print("=" * 70)
    print(f"NAIVE SGLD VALIDATION — Bayesian linear regression (D={D}, N={N})")
    print(f"  mode={'minibatch='+str(args.minibatch) if args.minibatch else 'full-batch (ULA)'}"
          f" | chains={args.n_chains} | epochs={total_steps} | burn_in={args.burn_in}")
    sched = (f"lr {args.lr} -> {args.lr_final} (poly, gamma={args.gamma})"
             if args.lr_final and args.lr_final > 0 else f"lr {args.lr} (constant)")
    print(f"  {sched} | sigma={sigma:.3g} | prior_precision alpha={alpha}")
    print("-" * 70)
    print(f"  split-R-hat:  max={rhat.max():.4f}  mean={rhat.mean():.4f}   -> {'OK' if ok_rhat else 'FAIL'} (<1.01)")
    print(f"  bulk-ESS:     min={ess.min():.0f}   median={np.median(ess):.0f}  (of {args.n_chains*chains.shape[1]} draws)")
    print(f"  mean z-score: max={z_mean.max():.3f} post-sd     -> {'OK' if ok_mean else 'FAIL'} (<0.15)")
    print(f"  post-sd relerr: max={sd_relerr.max()*100:.1f}%         -> {'OK' if ok_sd else 'FAIL'} (<10%)")
    print("-" * 70)
    print(f"  VERDICT: {'PASS — SGLD samples the correct posterior.' if passed else 'FAIL — implementation mismatch.'}")
    print("=" * 70)

    # ---- bias-vs-step sweep: ULA has O(eps) discretization bias in the posterior
    # variance that must vanish as eps -> 0. Demonstrating this scaling is strong
    # evidence the update rule is correct (and motivates W&T's decreasing schedule).
    sweep = []
    if args.sweep:
        # Hold Langevin time T = eps*epochs ~ constant across the sweep so ESS (and
        # hence Monte-Carlo error) stays comparable; then the SIGNED sd bias is the
        # O(eps) discretization error alone. ULA over-estimates the posterior
        # variance, so the signed relerr should be positive and roughly halve as
        # eps halves.
        print("  bias-vs-step sweep (eps*epochs held ~const): expect SIGNED sd-bias > 0, ~ eps")
        for eps in args.sweep_lrs:
            ep = max(2000, int(args.sweep_time / eps))
            ch = _run_chains(args, X, y, sigma, alpha, eps, None, ep, ep // 5)
            sd = ch.reshape(-1, D).std(0, ddof=1)
            signed = float(((sd - post_sd) / post_sd).mean())   # signed (not abs)
            sweep.append({"eps": eps, "epochs": ep, "mean_sd_signed_relerr": signed})
            print(f"    eps={eps:<8g} epochs={ep:<7d} signed sd bias = {signed*100:+6.2f}%")
        # monotone-ish decreasing positive bias is the correctness signature
        biases = [s["mean_sd_signed_relerr"] for s in sweep]
        shrinks = all(biases[i] >= biases[i + 1] - 0.01 for i in range(len(biases) - 1))
        print(f"  -> bias shrinks as eps->0: {'YES' if shrinks else 'noisy'} "
              f"(ULA O(eps) discretization bias; vanishes under the W&T decreasing schedule)")
        print("=" * 70)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        # (a) trace of beta_0 across chains
        for c in range(args.n_chains):
            ax[0, 0].plot(chains[c, :, 0], lw=0.6, alpha=0.8, label=f"chain {c}")
        ax[0, 0].axhline(mu_post[0], color="k", ls="--", lw=1, label="analytic mean")
        ax[0, 0].set_title(f"(a) Trace of $\\beta_0$  (split-R-hat={rhat[0]:.3f})")
        ax[0, 0].set_xlabel("post-burn-in step"); ax[0, 0].legend(fontsize=7)
        # (b) empirical vs analytic posterior for beta_0
        ax[0, 1].hist(flat[:, 0], bins=60, density=True, alpha=0.6, color="tab:blue")
        gx = np.linspace(mu_post[0] - 4 * post_sd[0], mu_post[0] + 4 * post_sd[0], 200)
        ax[0, 1].plot(gx, np.exp(-0.5 * ((gx - mu_post[0]) / post_sd[0]) ** 2)
                      / (post_sd[0] * np.sqrt(2 * np.pi)), "k-", lw=1.5, label="analytic $N(\\mu,\\Sigma)$")
        ax[0, 1].set_title("(b) Posterior of $\\beta_0$: SGLD vs analytic"); ax[0, 1].legend(fontsize=8)
        # (c) emp mean vs analytic mean (all params)
        ax[1, 0].errorbar(mu_post, emp_mean, yerr=emp_sd, fmt="o", ms=4, capsize=2)
        lim = [min(mu_post.min(), emp_mean.min()), max(mu_post.max(), emp_mean.max())]
        ax[1, 0].plot(lim, lim, "k--", lw=1)
        ax[1, 0].set_title("(c) Posterior mean: SGLD vs analytic")
        ax[1, 0].set_xlabel("analytic $\\mu$"); ax[1, 0].set_ylabel("SGLD empirical mean")
        # (d) R-hat and ESS bars
        axd = ax[1, 1]
        axd.bar(np.arange(D) - 0.2, rhat, width=0.4, label="split-R-hat")
        axd.axhline(1.01, color="r", ls=":", lw=1)
        axd.set_ylabel("R-hat"); axd.set_ylim(0.99, max(1.02, rhat.max() * 1.05))
        axt = axd.twinx(); axt.bar(np.arange(D) + 0.2, ess, width=0.4, color="tab:green", alpha=0.6)
        axt.set_ylabel("bulk-ESS", color="tab:green")
        axd.set_title("(d) Per-parameter R-hat / ESS"); axd.set_xlabel("parameter")
        fig.suptitle(f"Naive SGLD validation — {'PASS' if passed else 'FAIL'} "
                     f"(max R-hat {rhat.max():.3f}, min ESS {ess.min():.0f})", fontweight="bold")
        fig.tight_layout()
        out_png = PROJECT_ROOT / "docs" / "sgld_naive_validation.png"
        out_png.parent.mkdir(exist_ok=True)
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        print(f"[fig] {out_png}")
    except Exception as e:
        print(f"[fig] skipped ({e})")

    # ---- summary json ----
    summary = {
        "model": "bayesian_linear_regression",
        "config": {"n_data": N, "n_dim": D, "sigma": sigma, "prior_precision": alpha,
                   "n_chains": args.n_chains, "epochs": total_steps, "burn_in": args.burn_in,
                   "lr": args.lr, "lr_final": args.lr_final, "gamma": args.gamma,
                   "minibatch": args.minibatch},
        "diagnostics": {
            "split_rhat_max": float(rhat.max()), "split_rhat_mean": float(rhat.mean()),
            "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
            "mean_zscore_max": float(z_mean.max()), "post_sd_relerr_max": float(sd_relerr.max()),
            "per_param_rhat": rhat.tolist(), "per_param_ess": ess.tolist(),
        },
        "verdict": {"pass": passed, "ok_rhat": ok_rhat, "ok_mean": ok_mean, "ok_sd": ok_sd},
        "bias_vs_step_sweep": sweep,
    }
    out_json = PROJECT_ROOT / "runs" / "sgld_naive_validation.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[json] {out_json}")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_data", type=int, default=200)
    p.add_argument("--n_dim", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.0, help="0 = auto (sqrt(N), well-conditioned Hessian)")
    p.add_argument("--prior_precision", type=float, default=1.0, help="alpha = 1/prior_var")
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20000, help="Langevin steps per chain")
    p.add_argument("--burn_in", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-2, help="constant step (small, low ULA bias)")
    p.add_argument("--lr_final", type=float, default=0.0, help="0 = constant step; >0 = poly decay to this")
    p.add_argument("--gamma", type=float, default=0.55)
    p.add_argument("--minibatch", type=int, default=0, help="0 = full-batch ULA; >0 = stochastic-gradient SGLD")
    p.add_argument("--sweep", action="store_true", default=True, help="run bias-vs-step sweep")
    p.add_argument("--no-sweep", dest="sweep", action="store_false")
    p.add_argument("--sweep_lrs", type=float, nargs="+", default=[1.6e-1, 8e-2, 4e-2, 2e-2])
    p.add_argument("--sweep_time", type=float, default=600.0, help="eps*epochs held ~const across sweep")
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--chain_seed0", type=int, default=100)
    args = p.parse_args()
    ok = run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

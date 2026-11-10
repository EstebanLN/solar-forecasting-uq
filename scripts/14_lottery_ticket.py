#!/usr/bin/env python
"""
14_lottery_ticket.py — iterative magnitude pruning with rewinding to the
original initialisation (Frankle & Carbin, "The Lottery Ticket Hypothesis",
ICLR 2019, arXiv:1803.03635).

Each round: train the masked network from the SAME initial weights theta_0,
prune a fraction of the smallest-magnitude surviving weights (globally, over
weight matrices and kernels; biases and normalisation parameters are never
pruned), reset the survivors to theta_0, repeat. The output is the skill of the
"winning ticket" as a function of the fraction of weights remaining.

Two purposes, both asked for on 2026-11-10. As a result in itself: how small a
subnetwork keeps the skill of the tuned model. And for SGLD: the chains did not
mix on the full networks (0.5-1M parameters); a ticket with a tenth of the
weights is a far lower-dimensional posterior to sample.

Pruned weights are held at exactly zero by zeroing them and masking their
gradients: AdamW's decoupled weight decay and its moment estimates leave a zero
weight with zero gradient at zero, so solar_uq.train.train_one_model is used
unchanged and the dense round is the ordinary training recipe.

    .venv/bin/python scripts/14_lottery_ticket.py --arch mlp --site uniandes --hours_ahead 6 --seed 42

Output: runs/{arch}_lottery/{site}_H{H}_L{L}_P{patch}_seed{seed}_{ts}/
        summary.json (one row per round) + ticket_round{r}.pt (theta_0, mask, trained weights)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solar_uq import zoo
from solar_uq.data import make_loader
from solar_uq.metrics import skill_score
from solar_uq.train import eval_model, seed_everything, train_one_model


def parse_args():
    p = argparse.ArgumentParser(description="Lottery-ticket iterative magnitude pruning",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(zoo.ARCHS))
    p.add_argument("--site", default="uniandes", choices=["uniandes", "elpaso"])
    p.add_argument("--hours_ahead", type=int, default=6, choices=[1, 3, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--k_neighbors", type=int, default=8)
    p.add_argument("--rounds", type=int, default=10, help="pruning rounds after the dense one")
    p.add_argument("--prune_frac", type=float, default=0.2, help="fraction of surviving weights removed per round")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--day_threshold", type=float, default=20.0)
    return p.parse_args()


def prunable(model):
    """Weight matrices and kernels only: never biases or normalisation parameters."""
    norm_ids = {id(p) for m in model.modules()
                if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm))
                for p in m.parameters()}
    return {n: p for n, p in model.named_parameters() if p.ndim >= 2 and id(p) not in norm_ids}


def main():
    args = parse_args()
    two = zoo.ARCHS[args.arch]["two_input"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cell = zoo.canonical_run(args.arch, args.site, args.hours_ahead, args.seed)
    bp = cell["best_params"]
    D = zoo.build_data(args.arch, args.site, args.hours_ahead, args.seed, args.patch,
                       args.k_neighbors, args.day_threshold)
    norm, L = D["normalizer"], D["L"]
    bs = bp["batch_size"]
    tl = make_loader(D["train"], bs, shuffle=True, num_workers=args.num_workers, seed=args.seed, device=dev)
    vl = make_loader(D["val"], bs, shuffle=False, num_workers=0, seed=args.seed, device=dev)
    te = make_loader(D["test"], bs, shuffle=False, num_workers=0, seed=args.seed, device=dev)
    base = D["baseline_test"]["rmse_day"]

    seed_everything(args.seed)
    model = zoo.build_model(args.arch, bp, args.patch, L, D["d_tab"], args.k_neighbors).to(dev)
    theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    masks = {n: torch.ones_like(p, dtype=torch.bool) for n, p in prunable(model).items()}
    n_prunable = sum(m.numel() for m in masks.values())
    n_total = sum(p.numel() for p in model.parameters())
    # A gradient hook per prunable tensor keeps removed weights at exactly zero.
    for n, p in prunable(model).items():
        p.register_hook(lambda g, n=n: g * masks[n].to(g.dtype))

    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    H = int(round(args.hours_ahead * 6))
    out = PROJECT_ROOT / "runs" / f"{args.arch}_lottery" / f"{args.site}_H{H}_L{L}_P{args.patch}_seed{args.seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nLOTTERY TICKET | {args.arch} {args.site} h{args.hours_ahead} seed{args.seed} dev={dev}")
    print(f"  parametros: {n_total:,} total, {n_prunable:,} podables | {args.rounds} rondas al {args.prune_frac:.0%}")

    rows = []
    for r in range(args.rounds + 1):
        # rewind: survivors back to theta_0, removed weights to zero
        model.load_state_dict(theta0)
        with torch.no_grad():
            for n, p in prunable(model).items():
                p.mul_(masks[n].to(p.dtype))
        seed_everything(args.seed + r)        # data order; the initial weights are fixed by theta_0
        res = train_one_model(model=model, train_loader=tl, val_loader=vl, normalizer=norm,
                              lr=bp["lr"], weight_decay=bp["weight_decay"], l1_reg=bp.get("l1_reg", 0.0),
                              use_amp=(dev == "cuda"), epochs=args.epochs, patience=args.patience,
                              day_threshold=args.day_threshold, device=dev, fusion=two)
        with torch.no_grad():                 # the restored best state: re-assert the mask
            for n, p in prunable(model).items():
                p.mul_(masks[n].to(p.dtype))
        test = eval_model(model, te, norm, args.day_threshold, dev, fusion=two)
        alive = sum(int(m.sum()) for m in masks.values())
        row = dict(round=r, remaining_frac=alive / n_prunable, remaining_weights=alive,
                   best_epoch=res["best_epoch"], val_rmse_day=res["final_val"]["rmse_day"],
                   test_rmse_day=test["rmse_day"], test_skill_day=skill_score(test["rmse_day"], base))
        rows.append(row)
        print(f"  ronda {r:2d}: quedan {row['remaining_frac']:6.1%} ({alive:,}) | ep {row['best_epoch']:2d} | "
              f"val {row['val_rmse_day']:.2f} | test skill_day {row['test_skill_day']:.3f}")
        torch.save(dict(theta0=theta0, masks={k: v.cpu() for k, v in masks.items()},
                        model_state=model.state_dict(), best_params=bp, row=row),
                   out / f"ticket_round{r}.pt")
        json.dump(dict(method="lottery_ticket", arch=args.arch, site=args.site, hours_ahead=args.hours_ahead,
                       seed=args.seed, source_run=cell["run_dir"].name, n_params_total=n_total,
                       n_params_prunable=n_prunable, prune_frac=args.prune_frac,
                       persistence_test_rmse_day=base, rounds=rows),
                  open(out / "summary.json", "w"), indent=2)
        if r == args.rounds:
            break
        # global magnitude pruning of the survivors
        with torch.no_grad():
            mags = torch.cat([p.detach().abs()[masks[n]] for n, p in prunable(model).items()])
            k = int(args.prune_frac * mags.numel())
            thr = torch.kthvalue(mags.float().cpu(), k).values.item() if k > 0 else -1.0
            for n, p in prunable(model).items():
                masks[n] &= (p.detach().abs() > thr)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

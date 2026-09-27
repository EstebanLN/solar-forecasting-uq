#!/usr/bin/env bash
# FlatMLP spatial-ablation baseline, El Paso, corrected data: horizons {1,3,6}h,
# 2 seeds {42,1}. Sequential; skips combos that already have a summary.
set -e
cd "$(dirname "$0")/.." || exit 1   # repo root, machine-agnostic

for seed in 42 1; do
  for h in 1 3 6; do
    Hsteps=$(( h * 6 ))                       # horizon steps (10-min cadence)
    if ls runs/mlp_optuna/elpaso_H${Hsteps}_*_seed${seed}_*/summary.json >/dev/null 2>&1; then
      echo "[skip] mlp elpaso h${h} seed${seed} (summary exists)"
      continue
    fi
    echo "[run ] mlp elpaso h${h} seed${seed}  $(date +%F_%T)"
    .venv/bin/python -u scripts/06_mlp_optuna.py \
      --site elpaso --hours_ahead "${h}" --seed "${seed}" \
      --n_trials 50 --num_workers 4 \
      >> "logs/mlp_elpaso_h${h}_s${seed}.log" 2>&1
  done
done
echo "MLP_ELPASO_QUEUE_DONE $(date +%F_%T)"

#!/usr/bin/env bash
# ConvGraph-LSTM reduced protocol: 6h horizon, both sites, 2 seeds {42,1}.
# Sequential (one heavy K=8 job at a time on the 16 GB GPU), skips combos that
# already have a summary, resumes interrupted ones via persistent Optuna storage.
set -e
cd "$(dirname "$0")/.." || exit 1   # repo root, machine-agnostic

for seed in 42 1; do
  for site in elpaso uniandes; do
    if ls runs/convgraph_lstm/${site}_H36_*_seed${seed}_*/summary.json >/dev/null 2>&1; then
      echo "[skip] convgraph ${site} h6 seed${seed} (summary exists)"
      continue
    fi
    echo "[run ] convgraph ${site} h6 seed${seed}  $(date +%F_%T)"
    .venv/bin/python -u scripts/06_convgraph_lstm_optuna.py \
      --site "${site}" --hours_ahead 6 --seed "${seed}" \
      --n_trials 25 --num_workers 4 \
      >> "logs/convgraph_${site}_h6_s${seed}.log" 2>&1
  done
done
echo "CONVGRAPH_QUEUE_DONE $(date +%F_%T)"

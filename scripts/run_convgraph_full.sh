#!/usr/bin/env bash
# ConvGraph-LSTM FULL symmetric protocol: all horizons {1,3,6} x both sites x
# 4 seeds {42,1,7,13}, K=8, n_trials=25 (same reduced budget as the existing 6h
# runs so ConvGraph is internally comparable). One heavy job at a time (K=8).
# Idempotent: skips any (site,horizon,seed) that already has a summary.json;
# resumes interrupted trials via persistent Optuna storage.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # repo root, machine-agnostic
mkdir -p logs

already_done() {   # 0 if a summary exists for this site/hours/seed
  local site=$1 hours=$2 seed=$3
  .venv/bin/python - "$site" "$hours" "$seed" <<'PYEOF'
import json, glob, sys
site, hours, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
for f in glob.glob("runs/convgraph_lstm/*/summary.json"):
    try:
        d = json.load(open(f))
        if (d.get("site") == site and d.get("seed") == seed
                and abs(d.get("temporal", {}).get("horizon_hours", -1) - hours) < 0.01):
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

for hours in 6 3 1; do
  for site in elpaso uniandes; do
    for seed in 42 1 7 13; do
      if already_done "$site" "$hours" "$seed"; then
        echo "[skip] convgraph ${site} h${hours} seed${seed} (summary exists)"
        continue
      fi
      echo "[run ] convgraph ${site} h${hours} seed${seed}  $(date +%F_%T)"
      .venv/bin/python -u scripts/06_convgraph_lstm_optuna.py \
        --site "${site}" --hours_ahead "${hours}" --seed "${seed}" \
        --n_trials 25 --num_workers 4 \
        >> "logs/convgraph_${site}_h${hours}_s${seed}.log" 2>&1 \
        || echo "[FAIL] convgraph ${site} h${hours} seed${seed}"
    done
  done
done
echo "CONVGRAPH_FULL_DONE $(date +%F_%T)"

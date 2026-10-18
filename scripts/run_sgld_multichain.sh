#!/usr/bin/env bash
# Multi-chain SGLD, 4 chains IN PARALLEL on one GPU (Daniel protocol, 2026-10-06).
# Each chain is its own process (independent CUDA context, ~2.5 GB) writing
# chain_{i}.npz into a shared run dir; after all finish, one --combine pass
# computes cross-chain split-R-hat + bulk-ESS and the diagnostics figure.
#
# Usage: bash scripts/run_sgld_multichain.sh <arch> <site> <warm_ckpt> [epochs]
#   defaults: graphsage uniandes <v2 MAP ckpt> 2500
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

ARCH="${1:-graphsage}"
SITE="${2:-uniandes}"
CK="${3:-runs/graphsage_lstm_optuna_v2/uniandes_H36_L24_P16_seed42_20260816_153529/best_model.pt}"
EPOCHS="${4:-2500}"
NCH=4; SEED=42; H=6; BURN=500; NW=2

if [[ ! -f "$CK" ]]; then echo "[MISSING CKPT] $CK"; exit 1; fi
TS=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR="runs/${ARCH}_sgld_multichain/${SITE}_H36_L24_P16_seed${SEED}_${TS}"
mkdir -p "$RUN_DIR"
echo "[parallel multichain] $ARCH $SITE h6 | $NCH chains x $EPOCHS ep (parallel) -> $RUN_DIR  $(date +%F_%T)"

pids=()
for c in $(seq 0 $((NCH-1))); do
  .venv/bin/python -u scripts/08d_sgld_multichain.py \
    --arch "$ARCH" --site "$SITE" --hours_ahead "$H" --seed "$SEED" \
    --n_chains "$NCH" --epochs "$EPOCHS" --burn_in "$BURN" \
    --chain_only "$c" --run_dir "$RUN_DIR" \
    --warm_start_ckpt "$CK" --num_workers "$NW" \
    > "logs/sgld_mc_${ARCH}_${SITE}_h6_chain${c}.log" 2>&1 &
  pids+=("$!")
  echo "  launched chain $c (pid $!)"
  sleep 25   # stagger so the 4 patch-cache preloads don't hit disk at once
done

echo "[parallel multichain] waiting for ${#pids[@]} chains ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "  chain pid $p FAILED"; fail=1; }; done
echo "[parallel multichain] chains finished (fail=$fail); combining ..."

.venv/bin/python -u scripts/08d_sgld_multichain.py \
  --arch "$ARCH" --site "$SITE" --hours_ahead "$H" --seed "$SEED" \
  --n_chains "$NCH" --epochs "$EPOCHS" --burn_in "$BURN" \
  --combine --run_dir "$RUN_DIR" --warm_start_ckpt "$CK" \
  > "logs/sgld_mc_${ARCH}_${SITE}_h6_combine.log" 2>&1
echo "SGLD_MULTICHAIN_PARALLEL_DONE $RUN_DIR  $(date +%F_%T)"

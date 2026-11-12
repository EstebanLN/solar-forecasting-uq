#!/usr/bin/env bash
# The SGLD protocol, in one place, so every replication is symmetric.
#
#   bash scripts/run_sgld_protocol.sh <arch> <site> <hours> <seed> [ticket_round]
#
# Every architecture, site, horizon and seed is sampled with the SAME settings
# below. A ticket round selects a lottery-ticket subnetwork from
# 14_lottery_ticket.py; all architectures are pruned on the same schedule (20 %
# of survivors per round), so a given round is the same fraction of weights for
# every architecture (round 9 = 13.4 %). Without a round the dense tuned model is
# sampled.
#
# Fixed on 2026-11-12: four chains of 2500 epochs with a 500-epoch burn-in, the
# length the advisor asked for and the one the FlatMLP runs of 2026-11-10/11
# already used, so those stay comparable. (The ResNet and GraphSAGE runs of
# October used 500 epochs; they are not on this protocol.)
#
# Chains run as separate processes (08d --chain_only) sharing one run
# directory, each admitted when there is VRAM, and are combined at the end
# (08d --combine). Sequentially a fusion model would take about 17 days.
#
# Safety: waits for free VRAM, then runs a short smoke test (2 chains, 8
# epochs) and launches the real run only if the smoke test wrote its
# diagnostics. Both go to separate output roots.
set -u
ARCH=$1; SITE=$2; H=$3; SEED=$4; ROUND=${5:-}
cd "$(dirname "$0")/.." || exit 1
source scripts/ram_guard.sh

N_CHAINS=4; EPOCHS=2500; BURN_IN=500
SGLD="--sgld_lr 1e-08 --sgld_lr_final 1e-09 --sgld_prior_precision 10.0 --ckpt_every 25 --num_workers 2"

TICKET_ARG=""; TAG="dense"
if [ -n "$ROUND" ]; then
  T=$(ls -d runs/${ARCH}_lottery/${SITE}_H$((H*6))_*_seed${SEED}_* 2>/dev/null | sort | tail -1)
  [ -f "$T/ticket_round${ROUND}.pt" ] || { echo "!!! no existe $T/ticket_round${ROUND}.pt"; exit 1; }
  TICKET_ARG="--ticket $T/ticket_round${ROUND}.pt"; TAG="ticket_r${ROUND}"
fi
ROOT=runs/sgld_protocol/${ARCH}_${TAG}
LOG=logs/sgld_protocol_${ARCH}_${TAG}_${SITE}_h${H}_s${SEED}

ram_wait 12000 200 86400 || exit 1
gpu_wait 4000 604800 || exit 1

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  echo "=== smoke $ARCH $TAG $SITE h$H s$SEED $(date +%F_%T) ==="
  .venv/bin/python -u scripts/08d_sgld_multichain.py --arch $ARCH --site $SITE --hours_ahead $H --seed $SEED \
    --n_chains 2 --epochs 8 --burn_in 2 --val_every 1 --n_monitor 16 $SGLD $TICKET_ARG \
    --runs_root /tmp/sgld_smoke_${ARCH}_${TAG}_$$ > ${LOG}_smoke.log 2>&1
  if ! ls /tmp/sgld_smoke_${ARCH}_${TAG}_$$/*/summary.json > /dev/null 2>&1; then
    echo "!!! la prueba corta fallo, no lanzo la corrida real: ver ${LOG}_smoke.log"; exit 1
  fi
  rm -rf /tmp/sgld_smoke_${ARCH}_${TAG}_$$
  echo "=== prueba OK $(date +%F_%T) ==="
fi

RUN_DIR=$ROOT/${SITE}_H$((H*6))_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
COMMON="--arch $ARCH --site $SITE --hours_ahead $H --seed $SEED --n_chains $N_CHAINS --epochs $EPOCHS --burn_in $BURN_IN $SGLD $TICKET_ARG --run_dir $RUN_DIR"
echo "=== corrida real en $RUN_DIR $(date +%F_%T) ==="
for c in $(seq 0 $((N_CHAINS-1))); do
  gpu_wait 4000 604800 || exit 1
  ram_wait 12000 200 86400 || exit 1
  echo "--> cadena $c $(date +%F_%T)"
  .venv/bin/python -u scripts/08d_sgld_multichain.py $COMMON --chain_only $c > ${LOG}_chain$c.log 2>&1 &
  sleep 1200          # deja que la cadena construya datos y reserve VRAM antes de medir la siguiente
done
wait
n=$(ls $RUN_DIR/chain_*.npz 2>/dev/null | wc -l)
echo "=== cadenas terminadas: $n de $N_CHAINS $(date +%F_%T) ==="
.venv/bin/python -u scripts/08d_sgld_multichain.py $COMMON --combine > ${LOG}.log 2>&1
grep -E "split-R-hat|bulk-ESS|VERDICT" ${LOG}.log
echo "=== FIN $(date +%F_%T) ==="

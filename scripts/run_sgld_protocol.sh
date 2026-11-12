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
# Fixed on 2026-11-12. The FlatMLP runs of 2026-11-10/11 used 2500 epochs and a
# 500-epoch burn-in; this protocol uses 500 and 100 (the length of the ResNet
# run) so that the fusion models are affordable. Compare architectures only
# within one protocol.
#
# Safety: waits for free VRAM, then runs a short smoke test (2 chains, 8
# epochs) and launches the real run only if the smoke test wrote its
# diagnostics. Both go to separate output roots.
set -u
ARCH=$1; SITE=$2; H=$3; SEED=$4; ROUND=${5:-}
cd "$(dirname "$0")/.." || exit 1
source scripts/ram_guard.sh

N_CHAINS=4; EPOCHS=500; BURN_IN=100
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

echo "=== smoke $ARCH $TAG $SITE h$H s$SEED $(date +%F_%T) ==="
.venv/bin/python -u scripts/08d_sgld_multichain.py --arch $ARCH --site $SITE --hours_ahead $H --seed $SEED \
  --n_chains 2 --epochs 8 --burn_in 2 --val_every 1 --n_monitor 16 $SGLD $TICKET_ARG \
  --runs_root /tmp/sgld_smoke_${ARCH}_${TAG}_$$ > ${LOG}_smoke.log 2>&1
if ! ls /tmp/sgld_smoke_${ARCH}_${TAG}_$$/*/summary.json > /dev/null 2>&1; then
  echo "!!! la prueba corta fallo, no lanzo la corrida real: ver ${LOG}_smoke.log"; exit 1
fi
rm -rf /tmp/sgld_smoke_${ARCH}_${TAG}_$$
echo "=== prueba OK; corrida real $(date +%F_%T) ==="
gpu_wait 4000 604800 || exit 1
.venv/bin/python -u scripts/08d_sgld_multichain.py --arch $ARCH --site $SITE --hours_ahead $H --seed $SEED \
  --n_chains $N_CHAINS --epochs $EPOCHS --burn_in $BURN_IN $SGLD $TICKET_ARG \
  --runs_root $ROOT > ${LOG}.log 2>&1
echo "=== FIN rc=$? $(date +%F_%T) ==="

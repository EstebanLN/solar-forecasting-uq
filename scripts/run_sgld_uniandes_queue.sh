#!/usr/bin/env bash
# Sequential SGLD queue for the three Uniandes rows of Table 4
# (tab:decomposition, epistemic uncertainty). Backbone per horizon follows the
# paper's methodology (ssec:uq_results): best deep-learning backbone at each
# horizon -- ResNet-LSTM at 1h, GraphSAGE-LSTM at 3h and 6h -- all v1 (the
# fixed-graph configuration reported in Table 2 for Uniandes), seed 42.
#
# SGLD retrains from a FRESH initialisation using the Optuna best_params (it
# does NOT load the .pt checkpoint), so it only needs the run's summary.json,
# which is present locally for all Uniandes v1 combos.
#
# Runs one job at a time at nice -n 19 (minimum priority) so it only sips idle
# GPU cycles and never competes with the local El Paso corrected-retraining
# jobs (fusion + mlp), which stay on the critical path for de-daggering the
# paper.
#
# Safe to re-run: any combo that already has a completed run (summary.json
# present) is skipped. A failed combo is retried up to MAX_RETRIES times unless
# the failure is permanent (no matching Optuna run), in which case it is
# skipped immediately.
cd /srv/projects/Proyecto_e_ladino

# (arch, hours) pairs — the exact backbones Table 4 (Uniandes) calls for.
COMBOS=("resnet 1" "graphsage 3" "graphsage 6")
SEED=42
MAX_RETRIES=3
RETRY_DELAY=30

declare -A OUT_DIR=( [resnet]=resnet_lstm_sgld [graphsage]=graphsage_lstm_sgld )
declare -A HCODE=( [1]=6 [3]=18 [6]=36 )

mkdir -p logs
STATUS_LOG="logs/sgld_uniandes_queue_status.log"
echo "[sgld-queue] started $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS_LOG"

is_permanent_failure() {
  local logfile="$1"
  grep -qE "No completed Optuna run found" "$logfile"
}

for combo in "${COMBOS[@]}"; do
  read -r arch h <<< "$combo"
  out_dir="runs/${OUT_DIR[$arch]}"
  hcode="${HCODE[$h]}"
  done_match=$(find "$out_dir" -maxdepth 1 -type d -iname "uniandes_H${hcode}_L24_P16_seed${SEED}_*" 2>/dev/null | \
               while read -r d; do [ -f "$d/summary.json" ] && echo "$d"; done | head -1)

  if [ -n "$done_match" ]; then
    echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') SKIP (already done) arch=$arch h=$h seed=$SEED -> $done_match" | tee -a "$STATUS_LOG"
    continue
  fi

  attempt=1
  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    logfile="logs/sgld_${arch}_uniandes_h${h}_s${SEED}_attempt${attempt}.log"
    echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') START arch=$arch h=$h seed=$SEED (attempt $attempt/$MAX_RETRIES)" | tee -a "$STATUS_LOG"

    nice -n 19 .venv/bin/python scripts/08_sgld.py \
        --arch "$arch" --site uniandes --hours_ahead "$h" --seed "$SEED" \
        --optuna_version v1 > "$logfile" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
      echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') OK   arch=$arch h=$h seed=$SEED (attempt $attempt)" | tee -a "$STATUS_LOG"
      break
    fi

    if is_permanent_failure "$logfile"; then
      echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') SKIP (permanent: no matching Optuna run) arch=$arch h=$h seed=$SEED" | tee -a "$STATUS_LOG"
      break
    fi

    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
      echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') FAIL arch=$arch h=$h seed=$SEED (exit $rc, exhausted $MAX_RETRIES attempts, see $logfile)" | tee -a "$STATUS_LOG"
    else
      echo "[sgld-queue] $(date '+%Y-%m-%d %H:%M:%S') RETRY arch=$arch h=$h seed=$SEED (exit $rc on attempt $attempt, waiting ${RETRY_DELAY}s, see $logfile)" | tee -a "$STATUS_LOG"
      sleep "$RETRY_DELAY"
    fi
    attempt=$((attempt + 1))
  done
done

echo "[sgld-queue] finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$STATUS_LOG"
echo "[sgld-queue] summary:"
grep -c "OK   "               "$STATUS_LOG" | xargs echo "  completed:"
grep -c "FAIL "               "$STATUS_LOG" | xargs echo "  failed (exhausted retries):"
grep -c "SKIP (already done)" "$STATUS_LOG" | xargs echo "  already done:"
grep -c "SKIP (permanent"     "$STATUS_LOG" | xargs echo "  permanently blocked:"

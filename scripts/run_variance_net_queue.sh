#!/usr/bin/env bash
# Sequential queue for the aleatoric variance network (Step 2, cooperative
# BNN-VE) across all Uniandes v1 combos (3 archs x 3 horizons x 4 seeds = 36,
# minus whichever combos lack a mean-network checkpoint on this machine).
#
# More robust than a single pass: a failed combo is retried up to
# MAX_RETRIES times (with a pause in between) before being given up on --
# unless the failure is a PERMANENT one (the mean-network checkpoint only
# exists on the server and was never copied here), in which case retrying
# would just fail identically every time, so it is skipped immediately
# instead of wasting three attempts on it.
#
# Safe to re-run at any time: any combo that already has a completed run
# (summary.json present) is skipped without re-attempting it.
cd /srv/projects/Proyecto_e_ladino

ARCHS=(resnet graphsage mlp)
HOURS=(1 3 6)
SEEDS=(42 1 7 13)
MAX_RETRIES=3
RETRY_DELAY=30   # seconds between retry attempts

declare -A OUT_DIR=( [resnet]=resnet_lstm_variance [graphsage]=graphsage_lstm_variance [mlp]=mlp_variance )
declare -A HCODE=( [1]=6 [3]=18 [6]=36 )

mkdir -p logs
STATUS_LOG="logs/variance_net_queue_status.log"
echo "[queue] started $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS_LOG"

# Error signatures that mean "retrying will not help": the checkpoint this
# combo needs is recorded in summary.json but the .pt file itself only
# exists on a different machine (e.g. trained on the server, never synced
# locally -- .pt files are gitignored, so summary.json can travel via git
# without the checkpoint coming with it).
is_permanent_failure() {
  local logfile="$1"
  grep -qE "no usable best_ckpt_path|No completed Optuna run found" "$logfile"
}

for arch in "${ARCHS[@]}"; do
  for h in "${HOURS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out_dir="runs/${OUT_DIR[$arch]}"
      hcode="${HCODE[$h]}"
      done_match=$(find "$out_dir" -maxdepth 1 -type d -iname "uniandes_H${hcode}_L24_P16_seed${seed}_*" 2>/dev/null | \
                   while read -r d; do [ -f "$d/summary.json" ] && echo "$d"; done | head -1)

      if [ -n "$done_match" ]; then
        echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') SKIP (already done) arch=$arch h=$h seed=$seed -> $done_match" | tee -a "$STATUS_LOG"
        continue
      fi

      attempt=1
      while [ "$attempt" -le "$MAX_RETRIES" ]; do
        logfile="logs/variance_net_${arch}_uniandes_h${h}_s${seed}_attempt${attempt}.log"
        echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') START arch=$arch h=$h seed=$seed (attempt $attempt/$MAX_RETRIES)" | tee -a "$STATUS_LOG"

        .venv/bin/python scripts/12_variance_net.py \
            --arch "$arch" --site uniandes --hours_ahead "$h" --seed "$seed" \
            --optuna_version v1 > "$logfile" 2>&1
        rc=$?

        if [ $rc -eq 0 ]; then
          echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') OK   arch=$arch h=$h seed=$seed (attempt $attempt)" | tee -a "$STATUS_LOG"
          break
        fi

        if is_permanent_failure "$logfile"; then
          echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') SKIP (permanent: mean-network checkpoint unavailable locally) arch=$arch h=$h seed=$seed" | tee -a "$STATUS_LOG"
          break
        fi

        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
          echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') FAIL arch=$arch h=$h seed=$seed (exit $rc, exhausted $MAX_RETRIES attempts, see $logfile)" | tee -a "$STATUS_LOG"
        else
          echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') RETRY arch=$arch h=$h seed=$seed (exit $rc on attempt $attempt, waiting ${RETRY_DELAY}s, see $logfile)" | tee -a "$STATUS_LOG"
          sleep "$RETRY_DELAY"
        fi
        attempt=$((attempt + 1))
      done
    done
  done
done

echo "[queue] finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$STATUS_LOG"
echo "[queue] summary:"
grep -c "OK   "                        "$STATUS_LOG" | xargs echo "  completed:"
grep -c "FAIL "                        "$STATUS_LOG" | xargs echo "  failed (exhausted retries):"
grep -c "SKIP (already done)"          "$STATUS_LOG" | xargs echo "  already done:"
grep -c "SKIP (permanent"              "$STATUS_LOG" | xargs echo "  permanently blocked (checkpoint unavailable):"

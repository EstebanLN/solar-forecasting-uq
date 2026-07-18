#!/usr/bin/env bash
# Sequential runner for the GraphSAGE-LSTM P=64 single-seed probe.
# Runs one combo at a time (a single P64 trial can use the whole 11.5GB GPU;
# running >1 concurrently caused CUDA OOM crashes).
set -e
cd /srv/projects/Proyecto_e_ladino

for combo in "elpaso 1" "elpaso 3" "elpaso 6" "uniandes 1" "uniandes 3" "uniandes 6"; do
    set -- $combo
    site=$1; hours=$2
    logfile="logs/gsage_p64_${site}_h${hours}_s42.log"
    echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') lanzando $site h$hours ..."
    PYTHONUNBUFFERED=1 .venv/bin/python scripts/06_graphsage_lstm_optuna.py \
        --site "$site" --hours_ahead "$hours" --seed 42 --patch 64 --n_trials 20 \
        --runs_root runs/graphsage_lstm_p64 > "$logfile" 2>&1
    echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') terminado $site h$hours (exit $?)"
done

echo "[queue] todos los combos P64 completados"

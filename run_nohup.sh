#!/usr/bin/env bash
# ================================================================
# run_nohup.sh — lanza todos los experimentos en background
# Uso: bash run_nohup.sh [resnet|gsage|optuna|all]  (default: all)
# ================================================================
set -euo pipefail

GROUP="${1:-all}"
mkdir -p logs

run_resnet() {
    echo "[resnet] Lanzando ResNet+LSTM baseline..."

    # -- uniandes --
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 42  > logs/resnet_uniandes_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 1   > logs/resnet_uniandes_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 7   > logs/resnet_uniandes_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 13  > logs/resnet_uniandes_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 100 > logs/resnet_uniandes_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 42  > logs/resnet_uniandes_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 1   > logs/resnet_uniandes_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 7   > logs/resnet_uniandes_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 13  > logs/resnet_uniandes_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 100 > logs/resnet_uniandes_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 42  > logs/resnet_uniandes_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 1   > logs/resnet_uniandes_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 7   > logs/resnet_uniandes_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 13  > logs/resnet_uniandes_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 100 > logs/resnet_uniandes_h6_s100.log 2>&1 &

    # -- elpaso --
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 42  > logs/resnet_elpaso_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 1   > logs/resnet_elpaso_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 7   > logs/resnet_elpaso_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 13  > logs/resnet_elpaso_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 100 > logs/resnet_elpaso_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 42  > logs/resnet_elpaso_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 1   > logs/resnet_elpaso_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 7   > logs/resnet_elpaso_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 13  > logs/resnet_elpaso_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 100 > logs/resnet_elpaso_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 42  > logs/resnet_elpaso_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 1   > logs/resnet_elpaso_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 7   > logs/resnet_elpaso_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 13  > logs/resnet_elpaso_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_resnet_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 100 > logs/resnet_elpaso_h6_s100.log 2>&1 &

    echo "[resnet] 30 procesos lanzados."
}

run_gsage() {
    echo "[gsage] Lanzando GraphSAGE+LSTM baseline..."

    # -- uniandes --
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 42  > logs/gsage_uniandes_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 1   > logs/gsage_uniandes_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 7   > logs/gsage_uniandes_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 13  > logs/gsage_uniandes_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 1 --seed 100 > logs/gsage_uniandes_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 42  > logs/gsage_uniandes_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 1   > logs/gsage_uniandes_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 7   > logs/gsage_uniandes_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 13  > logs/gsage_uniandes_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 3 --seed 100 > logs/gsage_uniandes_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 42  > logs/gsage_uniandes_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 1   > logs/gsage_uniandes_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 7   > logs/gsage_uniandes_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 13  > logs/gsage_uniandes_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site uniandes --hours_ahead 6 --seed 100 > logs/gsage_uniandes_h6_s100.log 2>&1 &

    # -- elpaso --
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 42  > logs/gsage_elpaso_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 1   > logs/gsage_elpaso_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 7   > logs/gsage_elpaso_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 13  > logs/gsage_elpaso_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 1 --seed 100 > logs/gsage_elpaso_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 42  > logs/gsage_elpaso_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 1   > logs/gsage_elpaso_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 7   > logs/gsage_elpaso_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 13  > logs/gsage_elpaso_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 3 --seed 100 > logs/gsage_elpaso_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 42  > logs/gsage_elpaso_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 1   > logs/gsage_elpaso_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 7   > logs/gsage_elpaso_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 13  > logs/gsage_elpaso_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/05_graphsage_lstm_baseline.py --site elpaso --hours_ahead 6 --seed 100 > logs/gsage_elpaso_h6_s100.log 2>&1 &

    echo "[gsage] 30 procesos lanzados."
}

run_optuna() {
    echo "[optuna] Lanzando Optuna HPO..."

    # -- uniandes resnet --
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 42  --n_trials 50 > logs/optuna_resnet_uniandes_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 1   --n_trials 50 > logs/optuna_resnet_uniandes_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 7   --n_trials 50 > logs/optuna_resnet_uniandes_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 13  --n_trials 50 > logs/optuna_resnet_uniandes_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 100 --n_trials 50 > logs/optuna_resnet_uniandes_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 42  --n_trials 50 > logs/optuna_resnet_uniandes_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 1   --n_trials 50 > logs/optuna_resnet_uniandes_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 7   --n_trials 50 > logs/optuna_resnet_uniandes_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 13  --n_trials 50 > logs/optuna_resnet_uniandes_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 100 --n_trials 50 > logs/optuna_resnet_uniandes_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 42  --n_trials 50 > logs/optuna_resnet_uniandes_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 1   --n_trials 50 > logs/optuna_resnet_uniandes_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 7   --n_trials 50 > logs/optuna_resnet_uniandes_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 13  --n_trials 50 > logs/optuna_resnet_uniandes_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 100 --n_trials 50 > logs/optuna_resnet_uniandes_h6_s100.log 2>&1 &

    # -- uniandes gsage --
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 42  --n_trials 50 > logs/optuna_gsage_uniandes_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 1   --n_trials 50 > logs/optuna_gsage_uniandes_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 7   --n_trials 50 > logs/optuna_gsage_uniandes_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 13  --n_trials 50 > logs/optuna_gsage_uniandes_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 1 --seed 100 --n_trials 50 > logs/optuna_gsage_uniandes_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 42  --n_trials 50 > logs/optuna_gsage_uniandes_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 1   --n_trials 50 > logs/optuna_gsage_uniandes_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 7   --n_trials 50 > logs/optuna_gsage_uniandes_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 13  --n_trials 50 > logs/optuna_gsage_uniandes_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 3 --seed 100 --n_trials 50 > logs/optuna_gsage_uniandes_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 42  --n_trials 50 > logs/optuna_gsage_uniandes_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 1   --n_trials 50 > logs/optuna_gsage_uniandes_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 7   --n_trials 50 > logs/optuna_gsage_uniandes_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 13  --n_trials 50 > logs/optuna_gsage_uniandes_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site uniandes --hours_ahead 6 --seed 100 --n_trials 50 > logs/optuna_gsage_uniandes_h6_s100.log 2>&1 &

    # -- elpaso resnet --
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 42  --n_trials 50 > logs/optuna_resnet_elpaso_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 1   --n_trials 50 > logs/optuna_resnet_elpaso_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 7   --n_trials 50 > logs/optuna_resnet_elpaso_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 13  --n_trials 50 > logs/optuna_resnet_elpaso_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 100 --n_trials 50 > logs/optuna_resnet_elpaso_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 42  --n_trials 50 > logs/optuna_resnet_elpaso_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 1   --n_trials 50 > logs/optuna_resnet_elpaso_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 7   --n_trials 50 > logs/optuna_resnet_elpaso_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 13  --n_trials 50 > logs/optuna_resnet_elpaso_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 100 --n_trials 50 > logs/optuna_resnet_elpaso_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 42  --n_trials 50 > logs/optuna_resnet_elpaso_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 1   --n_trials 50 > logs/optuna_resnet_elpaso_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 7   --n_trials 50 > logs/optuna_resnet_elpaso_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 13  --n_trials 50 > logs/optuna_resnet_elpaso_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_resnet_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 100 --n_trials 50 > logs/optuna_resnet_elpaso_h6_s100.log 2>&1 &

    # -- elpaso gsage --
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 42  --n_trials 50 > logs/optuna_gsage_elpaso_h1_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 1   --n_trials 50 > logs/optuna_gsage_elpaso_h1_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 7   --n_trials 50 > logs/optuna_gsage_elpaso_h1_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 13  --n_trials 50 > logs/optuna_gsage_elpaso_h1_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 1 --seed 100 --n_trials 50 > logs/optuna_gsage_elpaso_h1_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 42  --n_trials 50 > logs/optuna_gsage_elpaso_h3_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 1   --n_trials 50 > logs/optuna_gsage_elpaso_h3_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 7   --n_trials 50 > logs/optuna_gsage_elpaso_h3_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 13  --n_trials 50 > logs/optuna_gsage_elpaso_h3_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 3 --seed 100 --n_trials 50 > logs/optuna_gsage_elpaso_h3_s100.log 2>&1 &

    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 42  --n_trials 50 > logs/optuna_gsage_elpaso_h6_s42.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 1   --n_trials 50 > logs/optuna_gsage_elpaso_h6_s1.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 7   --n_trials 50 > logs/optuna_gsage_elpaso_h6_s7.log   2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 13  --n_trials 50 > logs/optuna_gsage_elpaso_h6_s13.log  2>&1 &
    nohup .venv/bin/python scripts/06_graphsage_lstm_optuna.py --site elpaso --hours_ahead 6 --seed 100 --n_trials 50 > logs/optuna_gsage_elpaso_h6_s100.log 2>&1 &

    echo "[optuna] 60 procesos lanzados."
}

case "$GROUP" in
    resnet)  run_resnet ;;
    gsage)   run_gsage  ;;
    optuna)  run_optuna ;;
    all)
        run_resnet
        run_gsage
        run_optuna
        echo ""
        echo "Total: 120 procesos en background."
        ;;
    *)
        echo "Uso: bash run_nohup.sh [resnet|gsage|optuna|all]"
        exit 1
        ;;
esac

echo ""
echo "Verificar con:"
echo "  jobs -l"
echo "  nvidia-smi"
echo "  tail -f logs/resnet_uniandes_h6_s42.log"

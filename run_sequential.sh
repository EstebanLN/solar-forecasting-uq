#!/usr/bin/env bash
# ================================================================
# run_sequential.sh — corre los experimentos faltantes uno a uno
# Evita el OOM que ocurre al lanzar 120 procesos en paralelo.
# Saltea automáticamente cualquier run que ya tenga summary.json.
#
# Uso:
#   bash run_sequential.sh [baseline|optuna|resnet_optuna|gsage_optuna|all] [uniandes|elpaso|all]
#   bash run_sequential.sh resnet_optuna uniandes   # solo resnet, solo uniandes
# ================================================================
set -uo pipefail   # sin -e: un run fallido loguea el error y sigue al siguiente

GROUP="${1:-all}"
SITE_FILTER="${2:-all}"   # "uniandes", "elpaso", o "all"
mkdir -p logs
FAILED_RUNS=()   # acumula los que fallen para reportar al final

# ----------------------------------------------------------------
# Helper: devuelve 0 si ya existe un run completo para esta combo
# ----------------------------------------------------------------
already_done() {
    local runs_dir=$1 site=$2 hours=$3 seed=$4
    python3 - "$runs_dir" "$site" "$hours" "$seed" <<'PYEOF'
import json, glob, sys
runs_dir, site, hours, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
for f in glob.glob(f"{runs_dir}/*/summary.json"):
    try:
        d = json.load(open(f))
        if (d.get("site") == site
                and d.get("seed") == seed
                and abs(d.get("temporal", {}).get("horizon_hours", -1) - hours) < 0.01):
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

# ----------------------------------------------------------------
# Wrapper que saltea o corre
# ----------------------------------------------------------------
run_one() {
    local label=$1 runs_dir=$2 script=$3 site=$4 hours=$5 seed=$6
    shift 6
    local extra=("$@")

    if already_done "$runs_dir" "$site" "$hours" "$seed"; then
        echo "[SKIP] ${label} site=${site} h${hours} seed${seed} — ya existe summary.json"
        return 0
    fi

    local logfile="logs/${label}_${site}_h${hours}_s${seed}.log"
    echo "[RUN ] ${label} site=${site} h${hours} seed${seed} → ${logfile}"
    if .venv/bin/python "$script" \
            --site "$site" --hours_ahead "$hours" --seed "$seed" \
            "${extra[@]}" \
            > "$logfile" 2>&1; then
        echo "[DONE] ${label} site=${site} h${hours} seed${seed}"
    else
        echo "[FAIL] ${label} site=${site} h${hours} seed${seed} — ver ${logfile}"
        FAILED_RUNS+=("${label} ${site} h${hours} seed${seed}")
    fi
}

# ----------------------------------------------------------------
# Baseline: ResNet+LSTM  (30 runs)
# ----------------------------------------------------------------
run_baseline_resnet() {
    echo ""
    echo "=== ResNet+LSTM baseline (30 runs) ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13 100; do
                run_one "resnet" "runs/resnet_lstm" \
                    "scripts/05_resnet_lstm_baseline.py" \
                    "$site" "$hours" "$seed"
            done
        done
    done
}

# ----------------------------------------------------------------
# Baseline: GraphSAGE+LSTM  (30 runs)
# ----------------------------------------------------------------
run_baseline_gsage() {
    echo ""
    echo "=== GraphSAGE+LSTM baseline (30 runs) ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13 100; do
                run_one "gsage" "runs/graphsage_lstm" \
                    "scripts/05_graphsage_lstm_baseline.py" \
                    "$site" "$hours" "$seed"
            done
        done
    done
}

# ----------------------------------------------------------------
# Optuna: ResNet+LSTM  (24 runs × 20 trials, n_jobs=2)
# ----------------------------------------------------------------
run_optuna_resnet() {
    echo ""
    echo "=== ResNet+LSTM Optuna [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "optuna_resnet" "runs/resnet_lstm_optuna" \
                    "scripts/06_resnet_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --n_trials 20
            done
        done
    done
}

# ----------------------------------------------------------------
# Optuna: GraphSAGE+LSTM  (24 runs × 20 trials, n_jobs=2)
# ----------------------------------------------------------------
run_optuna_gsage() {
    echo ""
    echo "=== GraphSAGE+LSTM Optuna [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "optuna_gsage" "runs/graphsage_lstm_optuna" \
                    "scripts/06_graphsage_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --n_trials 20
            done
        done
    done
}

# ----------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------
case "$GROUP" in
    baseline)
        run_baseline_resnet
        run_baseline_gsage
        ;;
    optuna)
        run_optuna_resnet
        run_optuna_gsage
        ;;
    resnet_optuna)
        run_optuna_resnet
        ;;
    gsage_optuna)
        run_optuna_gsage
        ;;
    all)
        run_baseline_resnet
        run_baseline_gsage
        run_optuna_resnet
        run_optuna_gsage
        ;;
    *)
        echo "Uso: bash run_sequential.sh [baseline|optuna|resnet_optuna|gsage_optuna|all] [uniandes|elpaso|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Completado ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Runs con error (${#FAILED_RUNS[@]}):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
else
    echo "Sin errores."
fi

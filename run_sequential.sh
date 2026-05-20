#!/usr/bin/env bash
# ================================================================
# run_sequential.sh — corre los experimentos faltantes uno a uno
# Evita el OOM que ocurre al lanzar 120 procesos en paralelo.
# Saltea automáticamente cualquier run que ya tenga summary.json.
#
# Uso:
#   bash run_sequential.sh <grupo> [uniandes|elpaso|all]
#
# Grupos disponibles:
#   baseline          ResNet+LSTM y GraphSAGE+LSTM sin HPO (30 runs c/u)
#   optuna            ResNet Optuna v1 + GraphSAGE Optuna v1 + MLP (24 runs c/u, 20 trials)
#   resnet_optuna     Solo ResNet Optuna v1
#   gsage_optuna      Solo GraphSAGE Optuna v1  (COMPLETO — 24/24)
#   gsage_optuna_v2   GraphSAGE Optuna v2 — grafo ponderado, 100 trials, hidden hasta 256,
#                     l1_reg en búsqueda → runs/graphsage_lstm_optuna_v2/
#   resnet_optuna_v2  ResNet Optuna v2   — 100 trials, hidden hasta 256, n_lstm_layers,
#                     l1_reg en búsqueda → runs/resnet_lstm_optuna_v2/
#   mlp_optuna        FlatMLP Optuna (50 trials)
#   sarima            SARIMA baseline
#   all               Todo lo anterior
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
    .venv/bin/python - "$runs_dir" "$site" "$hours" "$seed" <<'PYEOF'
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

# Devuelve 0 si ya existe un run SARIMA completo (3 horizontes) para este sitio
sarima_done() {
    local site=$1
    .venv/bin/python - "runs/sarima" "$site" <<'PYEOF'
import json, glob, sys
runs_dir, site = sys.argv[1], sys.argv[2]
for f in glob.glob(f"{runs_dir}/*/summary.json"):
    try:
        d = json.load(open(f))
        if d.get("site") == site and len(d.get("results", {})) >= 3:
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
# Optuna: ResNet+LSTM v1  (24 runs × 20 trials) — YA COMPLETO
# ----------------------------------------------------------------
run_optuna_resnet() {
    echo ""
    echo "=== ResNet+LSTM Optuna v1 [site=${SITE_FILTER}] (COMPLETO — solo skip) ==="
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
# Optuna: ResNet+LSTM v2  (24 runs × 100 trials)
#   - emb_dim / hidden_t hasta 256
#   - n_lstm_layers {1, 2}
#   - l1_reg en búsqueda: {0, 1e-5, 1e-4, 1e-3}
#   - Resultado en runs/resnet_lstm_optuna_v2/
# ----------------------------------------------------------------
run_optuna_resnet_v2() {
    echo ""
    echo "=== ResNet+LSTM Optuna v2 — 100 trials, hidden hasta 256 [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "optuna_resnet_v2" "runs/resnet_lstm_optuna_v2" \
                    "scripts/06_resnet_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --n_trials 100 \
                    --runs_root runs/resnet_lstm_optuna_v2
            done
        done
    done
}

# ----------------------------------------------------------------
# Optuna: GraphSAGE+LSTM v1  (24 runs × 20 trials) — YA COMPLETO
# ----------------------------------------------------------------
run_optuna_gsage() {
    echo ""
    echo "=== GraphSAGE+LSTM Optuna v1 [site=${SITE_FILTER}] (COMPLETO — solo skip) ==="
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
# Optuna: GraphSAGE+LSTM v2  (24 runs × 100 trials)
#   - Grafo 8-conectado con pesos 1/distancia
#   - hidden_g / hidden_t hasta 256
#   - l1_reg en búsqueda: {0, 1e-5, 1e-4, 1e-3}
#   - Resultado en runs/graphsage_lstm_optuna_v2/
# ----------------------------------------------------------------
run_optuna_gsage_v2() {
    echo ""
    echo "=== GraphSAGE+LSTM Optuna v2 — grafo ponderado, 100 trials [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "optuna_gsage_v2" "runs/graphsage_lstm_optuna_v2" \
                    "scripts/06_graphsage_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --n_trials 100 \
                    --runs_root runs/graphsage_lstm_optuna_v2
            done
        done
    done
}

# ----------------------------------------------------------------
# Optuna: FlatMLP  (24 runs × 50 trials, n_jobs=2)
# ----------------------------------------------------------------
run_optuna_mlp() {
    echo ""
    echo "=== FlatMLP Optuna [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "optuna_mlp" "runs/mlp_optuna" \
                    "scripts/06_mlp_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --n_trials 50
            done
        done
    done
}

# ----------------------------------------------------------------
# SARIMA baseline (una ejecución por sitio cubre h1, h3, h6)
# ----------------------------------------------------------------
run_sarima() {
    echo ""
    echo "=== SARIMA baseline [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        if sarima_done "$site"; then
            echo "[SKIP] sarima site=${site} — ya existe summary.json con 3 horizontes"
            continue
        fi
        local logfile="logs/sarima_${site}.log"
        echo "[RUN ] sarima site=${site} → ${logfile}"
        if .venv/bin/python scripts/05_sarima_baseline.py \
                --site "$site" --hours_ahead 1 3 6 \
                > "$logfile" 2>&1; then
            echo "[DONE] sarima site=${site}"
        else
            echo "[FAIL] sarima site=${site} — ver ${logfile}"
            FAILED_RUNS+=("sarima ${site}")
        fi
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
        run_optuna_mlp
        ;;
    optuna_v2)
        run_optuna_resnet_v2
        run_optuna_gsage_v2
        ;;
    resnet_optuna)
        run_optuna_resnet
        ;;
    resnet_optuna_v2)
        run_optuna_resnet_v2
        ;;
    gsage_optuna)
        run_optuna_gsage
        ;;
    gsage_optuna_v2)
        run_optuna_gsage_v2
        ;;
    mlp_optuna)
        run_optuna_mlp
        ;;
    sarima)
        run_sarima
        ;;
    all)
        run_baseline_resnet
        run_baseline_gsage
        run_optuna_resnet
        run_optuna_gsage
        run_optuna_resnet_v2
        run_optuna_gsage_v2
        run_optuna_mlp
        run_sarima
        ;;
    *)
        echo "Uso: bash run_sequential.sh [baseline|optuna|optuna_v2|resnet_optuna|resnet_optuna_v2|gsage_optuna|gsage_optuna_v2|mlp_optuna|sarima|all] [uniandes|elpaso|all]"
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

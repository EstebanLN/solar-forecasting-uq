#!/usr/bin/env bash
# ================================================================
# run_sequential.sh — gestor único de experimentos solar UQ
#
# Uso:
#   bash run_sequential.sh                   → mostrar estado (default)
#   bash run_sequential.sh --launch          → lanzar pipeline completo en tmux
#   bash run_sequential.sh --force           → matar procesos, limpiar incompletos y lanzar
#   bash run_sequential.sh <grupo> [site]    → correr un grupo manualmente
#
# Grupos disponibles:
#   baseline          ResNet+LSTM y GraphSAGE+LSTM sin HPO
#   resnet_optuna     ResNet Optuna v1 (COMPLETO — solo skip)
#   gsage_optuna      GraphSAGE Optuna v1 (COMPLETO — solo skip)
#   resnet_optuna_v2  ResNet Optuna v2 — 100 trials, hidden hasta 256, n_lstm_layers
#   gsage_optuna_v2   GraphSAGE Optuna v2 — 100 trials, k-NN ponderado
#   mlp_optuna        FlatMLP Optuna — 50 trials
#   fusion_resnet     FusionResNet Optuna — 100 trials (satellite + surface)
#   fusion_gsage      FusionGraphSAGE Optuna — 100 trials (satellite + surface)
#   fusion            fusion_resnet + fusion_gsage
#   sarima            SARIMA baseline (re-run con GHI cruda)
#   sgld_resnet       SGLD ResNet+LSTM (requiere v2 completo)
#   sgld_gsage        SGLD GraphSAGE+LSTM (requiere v2 completo)
#   sgld_mlp          SGLD FlatMLP
#   sgld              sgld_resnet + sgld_gsage + sgld_mlp
#
# Pipeline tmux (4 ventanas — v2 → mlp → fusion):
#   [0] resnet_ep  : resnet_optuna_v2 elpaso   → mlp_optuna elpaso   → fusion_resnet elpaso
#   [1] resnet_uni : resnet_optuna_v2 uniandes → mlp_optuna uniandes → fusion_resnet uniandes
#   [2] gsage_ep   : gsage_optuna_v2 elpaso    → fusion_gsage elpaso
#   [3] gsage_uni  : gsage_optuna_v2 uniandes  → fusion_gsage uniandes
# MLP va en las ventanas de resnet porque es independiente de la arquitectura gsage.
# ================================================================
set -uo pipefail   # sin -e para que un run fallido no mate el script

# ── Configuración ─────────────────────────────────────────────────
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$PROJECT/run_sequential.sh"
SESSION="solar_runs"
cd "$PROJECT"
mkdir -p logs

SITE_FILTER="${2:-all}"
FAILED_RUNS=()

# ================================================================
# Helpers
# ================================================================

# Devuelve 0 si ya existe un run completo (con summary.json) para esta combo
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

# Devuelve 0 si existe un run SARIMA válido (metodología post-2026-06-07 con GHI cruda)
sarima_done() {
    local site=$1
    .venv/bin/python - "runs/sarima" "$site" <<'PYEOF'
import json, glob, sys
runs_dir, site = sys.argv[1], sys.argv[2]
for f in glob.glob(f"{runs_dir}/*/summary.json"):
    try:
        d = json.load(open(f))
        target_ok = d.get("model", {}).get("target") == "ghi_raw_wm2"
        if d.get("site") == site and target_ok and len(d.get("results", {})) >= 3:
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

# Saltea si ya está completo; si no, corre y loguea en logs/<label>_<site>_h<h>_s<seed>.log
run_one() {
    local label=$1 runs_dir=$2 script=$3 site=$4 hours=$5 seed=$6
    shift 6
    local extra=("$@")

    if already_done "$runs_dir" "$site" "$hours" "$seed"; then
        echo "[SKIP] ${label} site=${site} h${hours} seed${seed}"
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

# ================================================================
# Grupos de experimentos
# ================================================================

run_baseline_resnet() {
    echo ""; echo "=== ResNet+LSTM baseline (30 runs) ==="
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

run_baseline_gsage() {
    echo ""; echo "=== GraphSAGE+LSTM baseline (30 runs) ==="
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

# v1 completos: se llaman sólo para verificar/skip
run_optuna_resnet() {
    echo ""; echo "=== ResNet Optuna v1 [COMPLETO — solo skip] ==="
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

run_optuna_gsage() {
    echo ""; echo "=== GraphSAGE Optuna v1 [COMPLETO — solo skip] ==="
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

run_optuna_resnet_v2() {
    echo ""; echo "=== ResNet Optuna v2 — 100 trials [site=${SITE_FILTER}] ==="
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

run_optuna_gsage_v2() {
    echo ""; echo "=== GraphSAGE Optuna v2 — 100 trials [site=${SITE_FILTER}] ==="
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

run_optuna_mlp() {
    echo ""; echo "=== FlatMLP Optuna — 50 trials [site=${SITE_FILTER}] ==="
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

run_fusion_resnet() {
    echo ""; echo "=== FusionResNetLSTM Optuna — 100 trials [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "fusion_resnet" "runs/fusion_resnet_lstm" \
                    "scripts/06_resnet_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --fusion
            done
        done
    done
}

run_fusion_gsage() {
    echo ""; echo "=== FusionGraphSAGE_LSTM Optuna — 100 trials [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "fusion_gsage" "runs/fusion_graphsage_lstm" \
                    "scripts/06_graphsage_lstm_optuna.py" \
                    "$site" "$hours" "$seed" \
                    --fusion
            done
        done
    done
}

run_sgld_resnet() {
    echo ""; echo "=== SGLD ResNet+LSTM [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "sgld_resnet" "runs/resnet_lstm_sgld" \
                    "scripts/08_sgld.py" \
                    "$site" "$hours" "$seed" \
                    --arch resnet
            done
        done
    done
}

run_sgld_gsage() {
    echo ""; echo "=== SGLD GraphSAGE+LSTM [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "sgld_gsage" "runs/graphsage_lstm_sgld" \
                    "scripts/08_sgld.py" \
                    "$site" "$hours" "$seed" \
                    --arch graphsage
            done
        done
    done
}

run_sgld_mlp() {
    echo ""; echo "=== SGLD FlatMLP [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        for hours in 1 3 6; do
            for seed in 42 1 7 13; do
                run_one "sgld_mlp" "runs/mlp_sgld" \
                    "scripts/08_sgld.py" \
                    "$site" "$hours" "$seed" \
                    --arch mlp
            done
        done
    done
}

run_sarima() {
    echo ""; echo "=== SARIMA baseline [site=${SITE_FILTER}] ==="
    for site in uniandes elpaso; do
        [[ "$SITE_FILTER" != "all" && "$SITE_FILTER" != "$site" ]] && continue
        if sarima_done "$site"; then
            echo "[SKIP] sarima site=${site} — ya existe summary.json con metodología GHI cruda"
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

# ================================================================
# Estado
# ================================================================

show_status() {
    echo "=== Estado de runs ($(date '+%Y-%m-%d %H:%M')) ==="
    echo ""
    while IFS=: read -r path expected; do
        local count=0
        [[ -d "$path" ]] && count=$(find "$path" -name "summary.json" 2>/dev/null | wc -l)
        local mark="  "
        [[ "$count" -ge "$expected" ]] && mark="✓ "
        printf "  %s%-40s %2d / %2d\n" "$mark" "$path" "$count" "$expected"
    done <<'DIRS'
runs/resnet_lstm:30
runs/graphsage_lstm:30
runs/resnet_lstm_optuna:24
runs/graphsage_lstm_optuna:24
runs/resnet_lstm_optuna_v2:24
runs/graphsage_lstm_optuna_v2:24
runs/mlp_optuna:24
runs/fusion_resnet_lstm:24
runs/fusion_graphsage_lstm:24
runs/resnet_lstm_sgld:24
runs/graphsage_lstm_sgld:24
runs/mlp_sgld:24
runs/sarima:2
DIRS
    echo ""
    echo "=== Sesión tmux '$SESSION' ==="
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux list-windows -t "$SESSION" | sed 's/^/  /'
    else
        echo "  No existe. Lanzar con: bash $SELF --launch"
    fi
    echo ""
    echo "=== Procesos de experimentos activos ==="
    local procs
    procs=$(ps aux | grep -E "scripts/0[5-9]_|scripts/1[0-9]_" | grep -v grep \
            | awk '{printf "  PID=%-7s %s %s %s %s %s\n", $2, $11, $12, $13, $14, $15}')
    if [[ -n "$procs" ]]; then
        echo "$procs"
    else
        echo "  Ninguno."
    fi
}

# ================================================================
# Matar procesos y limpiar incompletos (solo con --force)
# ================================================================

kill_experiments() {
    echo "Matando procesos de experimentos en curso..."
    local killed=0
    for pat in \
        "scripts/06_resnet_lstm_optuna" \
        "scripts/06_graphsage_lstm_optuna" \
        "scripts/06_mlp_optuna" \
        "scripts/05_resnet_lstm_baseline" \
        "scripts/05_graphsage_lstm_baseline" \
        "scripts/05_sarima_baseline" \
        "scripts/08_sgld"; do
        if pkill -f "$pat" 2>/dev/null; then
            echo "  killed: $pat"
            killed=$((killed + 1))
        fi
    done
    [[ $killed -gt 0 ]] && sleep 3
    echo "  Total: $killed scripts terminados."
}

cleanup_incomplete() {
    echo "Limpiando directorios sin summary.json..."
    for dir in \
        runs/resnet_lstm_optuna_v2 \
        runs/graphsage_lstm_optuna_v2 \
        runs/mlp_optuna \
        runs/fusion_resnet_lstm \
        runs/fusion_graphsage_lstm \
        runs/resnet_lstm_sgld \
        runs/graphsage_lstm_sgld \
        runs/mlp_sgld; do
        [[ -d "$dir" ]] || continue
        local count=0
        for d in "$dir"/*/; do
            [[ -d "$d" ]] || continue
            if [[ ! -f "${d}summary.json" ]]; then
                rm -rf "$d"
                count=$((count + 1))
            fi
        done
        [[ $count -gt 0 ]] && echo "  $dir: $count eliminados"
    done
}

# ================================================================
# Lanzar sesión tmux
# ================================================================

launch_tmux() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "ERROR: sesión '$SESSION' ya existe."
        echo "  Para verla:      tmux attach -t $SESSION"
        echo "  Para reemplazar: bash $SELF --force"
        exit 1
    fi

    echo "Creando sesión tmux '$SESSION'..."

    tmux new-session -d -s "$SESSION" -n "resnet_ep" -x 220 -y 50
    tmux send-keys -t "$SESSION:0" "
cd '$PROJECT'
bash '$SELF' resnet_optuna_v2 elpaso   &&
bash '$SELF' mlp_optuna       elpaso   &&
bash '$SELF' fusion_resnet    elpaso
echo '=== [resnet_ep] COMPLETADO ===' " Enter

    tmux new-window -t "$SESSION" -n "resnet_uni"
    tmux send-keys -t "$SESSION:1" "
cd '$PROJECT'
bash '$SELF' resnet_optuna_v2 uniandes &&
bash '$SELF' mlp_optuna       uniandes &&
bash '$SELF' fusion_resnet    uniandes
echo '=== [resnet_uni] COMPLETADO ===' " Enter

    tmux new-window -t "$SESSION" -n "gsage_ep"
    tmux send-keys -t "$SESSION:2" "
cd '$PROJECT'
bash '$SELF' gsage_optuna_v2 elpaso   &&
bash '$SELF' fusion_gsage    elpaso
echo '=== [gsage_ep] COMPLETADO ===' " Enter

    tmux new-window -t "$SESSION" -n "gsage_uni"
    tmux send-keys -t "$SESSION:3" "
cd '$PROJECT'
bash '$SELF' gsage_optuna_v2 uniandes &&
bash '$SELF' fusion_gsage    uniandes
echo '=== [gsage_uni] COMPLETADO ===' " Enter

    tmux select-window -t "$SESSION:0"

    echo ""
    echo "  [0] resnet_ep  : resnet_optuna_v2 elpaso   → mlp_optuna elpaso   → fusion_resnet elpaso"
    echo "  [1] resnet_uni : resnet_optuna_v2 uniandes → mlp_optuna uniandes → fusion_resnet uniandes"
    echo "  [2] gsage_ep   : gsage_optuna_v2 elpaso    → fusion_gsage elpaso"
    echo "  [3] gsage_uni  : gsage_optuna_v2 uniandes  → fusion_gsage uniandes"
    echo ""
    echo "  tmux attach -t $SESSION     → conectarse / ver logs en vivo"
    echo "  Ctrl+B, d                   → desconectarse (runs siguen)"
    echo "  Ctrl+B, n/p  o  0-4        → navegar ventanas"
    echo "  bash $SELF --status         → ver estado sin conectarse"
}

# ================================================================
# Dispatch
# ================================================================

CMD="${1:-}"

case "$CMD" in
    ""|--status)
        show_status
        exit 0
        ;;
    --launch)
        launch_tmux
        exit 0
        ;;
    --force)
        tmux kill-session -t "$SESSION" 2>/dev/null && echo "Sesión tmux anterior eliminada." || true
        kill_experiments
        cleanup_incomplete
        echo ""
        launch_tmux
        exit 0
        ;;
    baseline)
        run_baseline_resnet
        run_baseline_gsage
        ;;
    resnet_optuna)     run_optuna_resnet ;;
    resnet_optuna_v2)  run_optuna_resnet_v2 ;;
    gsage_optuna)      run_optuna_gsage ;;
    gsage_optuna_v2)   run_optuna_gsage_v2 ;;
    mlp_optuna)        run_optuna_mlp ;;
    fusion_resnet)     run_fusion_resnet ;;
    fusion_gsage)      run_fusion_gsage ;;
    fusion)
        run_fusion_resnet
        run_fusion_gsage
        ;;
    sarima)            run_sarima ;;
    sgld_resnet)       run_sgld_resnet ;;
    sgld_gsage)        run_sgld_gsage ;;
    sgld_mlp)          run_sgld_mlp ;;
    sgld)
        run_sgld_resnet
        run_sgld_gsage
        run_sgld_mlp
        ;;
    *)
        echo "Uso: bash run_sequential.sh [--status|--launch|--force|<grupo>] [uniandes|elpaso|all]"
        echo ""
        echo "Grupos:"
        echo "  baseline          resnet_optuna     resnet_optuna_v2"
        echo "  gsage_optuna      gsage_optuna_v2   mlp_optuna"
        echo "  fusion_resnet     fusion_gsage      fusion"
        echo "  sarima            sgld_resnet       sgld_gsage  sgld_mlp  sgld"
        exit 1
        ;;
esac

# Resumen final (solo ejecución manual de grupo, no para --flags)
echo ""
echo "=== Completado ==="
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "Runs con error (${#FAILED_RUNS[@]}):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
else
    echo "Sin errores."
fi

#!/usr/bin/env bash
# ================================================================
# launch_experiments.sh — lanza todos los experimentos pendientes
# en una sesión tmux persistente que sobrevive al cierre de sesión.
#
# Uso:
#   bash launch_experiments.sh          # lanzar (falla si ya existe)
#   bash launch_experiments.sh --force  # matar sesión anterior y relanzar
#   bash launch_experiments.sh --status # ver estado sin lanzar
#
# Para conectarse a los logs en vivo:
#   tmux attach -t solar_runs
#   Ctrl+B, n/p     — siguiente/anterior ventana
#   Ctrl+B, 0-3     — ir a ventana N directamente
#   Ctrl+B, d       — desconectarse (runs siguen corriendo)
#   tmux kill-session -t solar_runs  — detener todo
#
# Pipeline por ventana (máx. 4 procesos GPU simultáneos):
#   0 resnet_ep  : resnet_v2 elpaso   → sgld_resnet elpaso   → sgld_mlp elpaso
#   1 resnet_uni : resnet_v2 uniandes → sgld_resnet uniandes → sgld_mlp uniandes
#   2 gsage_ep   : gsage_v2  elpaso   → sgld_gsage  elpaso
#   3 gsage_uni  : gsage_v2  uniandes → sgld_gsage  uniandes
# ================================================================
set -euo pipefail

SESSION="solar_runs"
PROJECT="/srv/projects/Proyecto_e_ladino"

# ----------------------------------------------------------------
# --status: mostrar estado actual de runs sin lanzar nada
# ----------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
    echo "=== Estado de runs ==="
    for dir in resnet_lstm_optuna_v2 graphsage_lstm_optuna_v2 resnet_lstm_sgld graphsage_lstm_sgld mlp_sgld; do
        count=0; [[ -d "$PROJECT/runs/$dir" ]] && count=$(find "$PROJECT/runs/$dir" -name "summary.json" | wc -l)
        echo "  runs/$dir: $count summary.json"
    done
    echo ""
    echo "=== Sesión tmux ==="
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  Activa. Ventanas:"
        tmux list-windows -t "$SESSION" | sed 's/^/    /'
    else
        echo "  No existe."
    fi
    exit 0
fi

# ----------------------------------------------------------------
# --force: matar sesión anterior
# ----------------------------------------------------------------
if [[ "${1:-}" == "--force" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Sesión anterior eliminada." || true
fi

# ----------------------------------------------------------------
# Verificar que no haya sesión activa
# ----------------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: la sesión '$SESSION' ya existe."
    echo "  Para verla:      tmux attach -t $SESSION"
    echo "  Para reemplazar: bash launch_experiments.sh --force"
    exit 1
fi

cd "$PROJECT"
mkdir -p logs

# Wrapper: corre el grupo y loguea; el && encadena solo si el anterior exitó
run_group() {
    local group=$1 site=$2 logfile=$3
    echo "[$(date '+%H:%M:%S')] Iniciando: run_sequential.sh $group $site → $logfile"
    bash run_sequential.sh "$group" "$site" 2>&1 | tee "$logfile"
    echo "[$(date '+%H:%M:%S')] Terminado: $group $site"
}
export -f run_group

echo "Creando sesión tmux '$SESSION'..."

# ── Ventana 0: resnet elpaso ──────────────────────────────────────
tmux new-session -d -s "$SESSION" -n "resnet_ep" -x 220 -y 50
tmux send-keys -t "$SESSION:0" "
cd $PROJECT
run_group() {
    local group=\$1 site=\$2 logfile=\$3
    echo \"[\$(date '+%H:%M:%S')] Iniciando: \$group \$site\"
    bash run_sequential.sh \"\$group\" \"\$site\" 2>&1 | tee \"\$logfile\"
    echo \"[\$(date '+%H:%M:%S')] Terminado: \$group \$site\"
}
run_group resnet_optuna_v2 elpaso   logs/resnet_v2_ep.out   &&
run_group sgld_resnet      elpaso   logs/sgld_resnet_ep.out &&
run_group sgld_mlp         elpaso   logs/sgld_mlp_ep.out
echo '=== [resnet_ep] PIPELINE COMPLETADO ==='
" Enter

# ── Ventana 1: resnet uniandes ───────────────────────────────────
tmux new-window -t "$SESSION" -n "resnet_uni"
tmux send-keys -t "$SESSION:1" "
cd $PROJECT
run_group() {
    local group=\$1 site=\$2 logfile=\$3
    echo \"[\$(date '+%H:%M:%S')] Iniciando: \$group \$site\"
    bash run_sequential.sh \"\$group\" \"\$site\" 2>&1 | tee \"\$logfile\"
    echo \"[\$(date '+%H:%M:%S')] Terminado: \$group \$site\"
}
run_group resnet_optuna_v2 uniandes logs/resnet_v2_uni.out   &&
run_group sgld_resnet      uniandes logs/sgld_resnet_uni.out &&
run_group sgld_mlp         uniandes logs/sgld_mlp_uni.out
echo '=== [resnet_uni] PIPELINE COMPLETADO ==='
" Enter

# ── Ventana 2: gsage elpaso ───────────────────────────────────────
tmux new-window -t "$SESSION" -n "gsage_ep"
tmux send-keys -t "$SESSION:2" "
cd $PROJECT
run_group() {
    local group=\$1 site=\$2 logfile=\$3
    echo \"[\$(date '+%H:%M:%S')] Iniciando: \$group \$site\"
    bash run_sequential.sh \"\$group\" \"\$site\" 2>&1 | tee \"\$logfile\"
    echo \"[\$(date '+%H:%M:%S')] Terminado: \$group \$site\"
}
run_group gsage_optuna_v2 elpaso   logs/gsage_v2_ep.out   &&
run_group sgld_gsage      elpaso   logs/sgld_gsage_ep.out
echo '=== [gsage_ep] PIPELINE COMPLETADO ==='
" Enter

# ── Ventana 3: gsage uniandes ─────────────────────────────────────
tmux new-window -t "$SESSION" -n "gsage_uni"
tmux send-keys -t "$SESSION:3" "
cd $PROJECT
run_group() {
    local group=\$1 site=\$2 logfile=\$3
    echo \"[\$(date '+%H:%M:%S')] Iniciando: \$group \$site\"
    bash run_sequential.sh \"\$group\" \"\$site\" 2>&1 | tee \"\$logfile\"
    echo \"[\$(date '+%H:%M:%S')] Terminado: \$group \$site\"
}
run_group gsage_optuna_v2 uniandes logs/gsage_v2_uni.out   &&
run_group sgld_gsage      uniandes logs/sgld_gsage_uni.out
echo '=== [gsage_uni] PIPELINE COMPLETADO ==='
" Enter

# Ir a la ventana 0 por defecto al conectarse
tmux select-window -t "$SESSION:0"

echo ""
echo "Sesión '$SESSION' activa. Pipeline:"
echo "  [0] resnet_ep  : resnet_v2 elpaso   → sgld_resnet elpaso   → sgld_mlp elpaso"
echo "  [1] resnet_uni : resnet_v2 uniandes → sgld_resnet uniandes → sgld_mlp uniandes"
echo "  [2] gsage_ep   : gsage_v2 elpaso    → sgld_gsage elpaso"
echo "  [3] gsage_uni  : gsage_v2 uniandes  → sgld_gsage uniandes"
echo ""
echo "Comandos:"
echo "  tmux attach -t $SESSION        # conectarse / ver logs en vivo"
echo "  bash launch_experiments.sh --status   # ver estado sin conectarse"
echo "  Ctrl+B, d                      # desconectarse (runs siguen)"
echo "  Ctrl+B, n / p / 0-3            # navegar ventanas"

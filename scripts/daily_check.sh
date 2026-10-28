#!/usr/bin/env bash
# Snapshot diario del estado, para leer al volver de viaje.
# Pensado para cron del sistema: no depende de ninguna sesion de Claude, no lanza
# ni mata nada, solo observa y escribe. Un fallo en cualquier bloque no detiene
# los demas.
#
#   scripts/daily_check.sh            # imprime el informe
#   scripts/daily_check.sh --append   # lo agrega a logs/daily_check.log
LOCAL=/srv/projects/Proyecto_e_ladino
HOST=${SYNC_HOST:-uniandes-personal-daniel}
RDIR=${SYNC_REMOTE_DIR:-Documents/solar-forecasting-uq}
cd "$LOCAL" || exit 1

report() {
  echo "================================================================"
  echo " CHECK DIARIO  $(date '+%F %T %Z')"
  echo "================================================================"

  echo
  echo "--- LOCAL: trabajos vivos ---"
  for p in /proc/[0-9]*; do
    p=${p#/proc/}; [ -r "/proc/$p/cmdline" ] || continue
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in
      *"venv/bin/python"*scripts/*)
        printf '  %-9s %-13s %s\n' "$p" "$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')" \
          "$(printf '%s' "$c" | sed 's|.*scripts/|scripts/|' | cut -c1-52)";;
    esac
  done

  echo
  echo "--- LOCAL: recursos ---"
  uptime | awk -F'load average:' '{print "  carga:"$2}'
  free -m | awk 'NR==2{printf "  RAM %d/%d MB, %d disponibles (%.0f%%)\n",$3,$2,$7,100*$7/$2}
                 NR==3{printf "  swap %d/%d (%.0f%%)\n",$3,$2,100*$3/$2}'
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/  GPU /'
  for u in estebanladino aiaindustrial; do
    printf '  RSS %-16s %5.1f GB\n' "$u" \
      "$(ps -u "$u" -o rss= 2>/dev/null | awk '{s+=$1} END{print s/1048576}')"
  done
  n=$( (dmesg -T 2>/dev/null || true) | grep -ciE 'killed process' )
  echo "  OOM-kills en el log del kernel: ${n:-?}"

  echo
  echo "--- LOCAL: progreso ---"
  [ -f logs/_sgld_full.log ] && echo "  SGLD: $(tail -1 logs/_sgld_full.log | tr -s ' ')"
  for f in logs/surface_*_t75.log; do
    [ -f "$f" ] || continue
    printf '  %-34s %s/75 trials\n' "$(basename "$f" .log)" "$(grep -c 'Trial.*finished' "$f" 2>/dev/null)"
  done
  [ -f logs/_finish_uq.log ] && echo "  cadena UQ: $(tail -1 logs/_finish_uq.log)"

  echo
  echo "--- SERVIDOR ---"
  if ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" true 2>/dev/null; then
    ssh "$HOST" "cd $RDIR 2>/dev/null || exit 1
      for p in /proc/[0-9]*; do p=\${p#/proc/}; [ -r \"/proc/\$p/cmdline\" ] || continue
        c=\$(tr '\\0' ' ' < \"/proc/\$p/cmdline\" 2>/dev/null)
        case \"\$c\" in
          *\"bash -c\"*|*cmdline*) ;;
          *venv/bin/python*scripts/*)
          printf '  %-9s %-13s %s\\n' \"\$p\" \"\$(ps -o etime= -p \$p 2>/dev/null | tr -d ' ')\" \
            \"\$(printf '%s' \"\$c\" | sed 's|.*scripts/|scripts/|' | cut -c1-52)\";; esac
      done
      uptime | awk -F'load average:' '{print \"  carga:\"\$2}'
      free -m | awk 'NR==2{printf \"  RAM %d disponibles (%.0f%%)\\n\",\$7,100*\$7/\$2}'
      for h in 1 3 6; do
        f=logs/surface_elpaso_h\${h}_t75.log
        [ -f \"\$f\" ] && printf '  surface elpaso h%s: %s/75 trials\\n' \"\$h\" \"\$(grep -c 'Trial.*finished' \$f)\"
      done" 2>/dev/null
  else
    echo "  NO SE PUDO CONECTAR al servidor"
  fi

  echo
  echo "--- AUDITORIA ---"
  .venv/bin/python scripts/00_audit_runs.py 2>/dev/null | grep -vE '^  \[run_select\]' \
    | grep -E '^AUDIT|^\[' | sed 's/^/  /'

  echo
  echo "--- SIN SINCRONIZAR DESDE EL SERVIDOR (dry run) ---"
  bash scripts/sync_from_server.sh 2>/dev/null | grep -E 'corridas|solo en el servidor|nada que traer' | sed 's/^/  /'

  echo
  echo "--- GIT ---"
  printf '  %s sin commitear | %s sin empujar | HEAD %s\n' \
    "$(git status --porcelain | wc -l)" \
    "$(git log --oneline origin/main..HEAD 2>/dev/null | wc -l)" \
    "$(git log --oneline -1 | cut -c1-48)"
  echo
}

if [ "$1" = "--append" ]; then
  report >> logs/daily_check.log 2>&1
else
  report
fi

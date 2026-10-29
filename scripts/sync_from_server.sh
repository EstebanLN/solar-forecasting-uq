#!/usr/bin/env bash
# Pull finished runs from the compute server to this machine, which is canonical
# for results. One direction only: it never writes to the server and never
# deletes anything here.
#
# Why a script instead of an ad-hoc rsync. On 2026-10-27 a hand-written
# `rsync -a --relative <host>:.../runs/./<run>/ runs/<run>/` put every one of five
# runs a second time INSIDE itself, and the duplicates were committed before
# anyone noticed. The ./ marker in --relative recreates the whole path under the
# destination. So: no --relative, one run directory at a time, and every file
# checksummed on both sides afterwards.
#
#   scripts/sync_from_server.sh            # dry run: lists what is missing
#   scripts/sync_from_server.sh --apply    # transfers and verifies
#
# Exit code is non-zero if any run failed to verify.

set -o pipefail
HOST=${SYNC_HOST:-uniandes-personal-daniel}
RDIR=${SYNC_REMOTE_DIR:-Documents/solar-forecasting-uq}
LOCAL=/srv/projects/Proyecto_e_ladino
APPLY=0
[ "$1" = "--apply" ] && APPLY=1

cd "$LOCAL" || exit 1
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "=== inventario ==="
find runs -name summary.json | sed 's|^runs/||' | sort > "$TMP/local.txt"
ssh "$HOST" "cd $RDIR && find runs -name summary.json | sed 's|^runs/||' | sort" > "$TMP/remote.txt" || {
  echo "no se pudo listar el servidor" >&2; exit 1; }
[ -s "$TMP/remote.txt" ] || { echo "listado remoto vacio: abortando por seguridad" >&2; exit 1; }
printf '  aqui: %s corridas | servidor: %s corridas\n' \
  "$(wc -l < "$TMP/local.txt")" "$(wc -l < "$TMP/remote.txt")"

comm -13 "$TMP/local.txt" "$TMP/remote.txt" | sed 's|/summary.json$||' > "$TMP/missing.txt"
n=$(wc -l < "$TMP/missing.txt")
if [ "$n" -eq 0 ]; then echo "  nada que traer"; exit 0; fi
echo "=== solo en el servidor ($n) ==="; sed 's/^/  /' "$TMP/missing.txt"

if [ "$APPLY" -eq 0 ]; then
  echo
  echo "  (dry run) vuelve a correrlo con --apply para traerlas y verificarlas"
  exit 0
fi

fail=0
processed=0
while IFS= read -r run <&3; do
  [ -n "$run" ] || continue
  processed=$((processed+1))
  case "$run" in
    *..*|/*) echo "  RECHAZADA ruta sospechosa: $run" >&2; fail=1; continue;;
  esac
  echo "--- $run"
  mkdir -p "runs/$run"
  # Trailing slash on BOTH sides, no --relative: contents land directly in place.
  if ! rsync -a "$HOST:$RDIR/runs/$run/" "runs/$run/" < /dev/null; then
    echo "  rsync fallo" >&2; fail=1; continue
  fi
  # Verify: every remote file must exist here with the same checksum.
  ssh -n "$HOST" "cd $RDIR/runs/$run && find . -type f -exec md5sum {} + | sort -k2" \
    > "$TMP/rsum" 2>/dev/null
  (cd "runs/$run" && find . -type f -exec md5sum {} + | sort -k2) > "$TMP/lsum" 2>/dev/null
  if diff -q "$TMP/rsum" "$TMP/lsum" >/dev/null 2>&1; then
    echo "  OK  $(wc -l < "$TMP/lsum") archivo(s) verificados por checksum"
  else
    echo "  VERIFICACION FALLIDA:" >&2
    diff "$TMP/rsum" "$TMP/lsum" | head -6 >&2
    fail=1
  fi
  # The nesting pathology that started this: a run inside a run.
  if find "runs/$run" -mindepth 2 -name summary.json | grep -q .; then
    echo "  ANIDAMIENTO detectado dentro de runs/$run" >&2; fail=1
  fi
done 3< "$TMP/missing.txt"

# Every line of missing.txt must have been read. On 2026-10-29 the loop body's
# ssh inherited the loop's stdin and swallowed the rest of the list, so a sync
# with 2 missing runs transferred 1 and still printed "todo verificado"; the
# El Paso h3 UQ then ran with 3 variance nets instead of 4. The <&3/3< pair
# fixes the cause, and this counter makes the symptom impossible to miss again.
if [ "$processed" -ne "$n" ]; then
  echo "  INCOMPLETO: $processed de $n corridas procesadas" >&2; fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "=== todo verificado. Revisa con: .venv/bin/python scripts/00_audit_runs.py ==="
else
  echo "=== hubo fallos: NO commitees hasta revisarlos ===" >&2
fi
exit "$fail"

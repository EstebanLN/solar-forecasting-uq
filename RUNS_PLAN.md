# Plan de runs — fuente de verdad

_Actualizado: 2026-08-11 (relanzado local tras desconexión de AIA blanco)_

Este archivo es la **única fuente de verdad** para qué runs faltan, en qué máquina se
lanzan, en qué orden y con qué restricciones. Antes de lanzar cualquier run,
sincronizar `summary.json` entre máquinas (ver §Protocolo antiduplicados) y
recalcular faltantes con `audit_runs.py` (o el bloque Python de este archivo).

Estado de progreso general (memoria conceptual, no operativa) sigue en `PENDIENTES.md`.

---

## Máquinas

- **AIA blanco** = local, RTX 5070 (12 GB VRAM). Working dir `/srv/projects/Proyecto_e_ladino`.
- **Daniel** = servidor Uniandes, GPU 16 GB (`ssh uniandes-personal-daniel`,
  working dir `~/Documents/solar-forecasting-uq`, tmux `solar_runs`).

---

## Lanzar en AIA blanco SIEMPRE dentro de tmux (lección 11-ago)

Los `nohup` locales murieron dos veces al caerse la sesión de VS Code Remote-SSH
(systemd tumba el scope del login). Daniel nunca se cae porque corre en tmux.
**Regla:** en AIA blanco lanzar dentro de la sesión tmux `solar_local`
(ventanas `fusion`, `mlp`), no con `nohup` suelto. Además se activó
`loginctl enable-linger` (systemd ya no mata procesos de usuario al desconectar).

```bash
tmux new-session -d -s solar_local -n fusion   # crear si no existe
tmux send-keys -t solar_local:fusion "<comando> > logs/<x>.log 2>&1" C-m
```

Reconectar a ver progreso: `tmux attach -t solar_local` (Ctrl-b d para salir).

---

## Cómo cancelar un run sin gatillar el siguiente (lección 10-ago)

`run_sequential.sh` es un for-loop en bash: si matas solo el proceso Python child,
el shell padre continúa con el siguiente combo del loop — así arrancan duplicados
"por inercia". Para cancelar limpio:

```bash
# Encuentra el PPID del python
ps -eo pid,ppid,cmd | grep <script>
# Mata el bash padre PRIMERO (si es el run_sequential.sh de la ventana que
# quieres detener — NO uses pkill -f run_sequential.sh, mata todos los shells
# de todas las ventanas)
kill <PPID_bash>
# Luego mata el python child (se re-parenta a init si el padre murió antes)
kill <PID_python>
```

Verificar después: `ps -eo pid,ppid,cmd | grep -E "python.*\.py|run_sequential"`.

---

## Protocolo antiduplicados (obligatorio antes de lanzar)

El bug del duplicado (23 h de gsage_v2 elpaso h6 s42 desperdiciadas el 9–10 ago)
NO fue un bug de `already_done()`. Fue **desincronización**: AIA blanco tenía
`summary.json` del combo, Daniel no lo tenía → su `already_done()` correctamente
respondió "no está hecho".

**Regla:** antes de `--launch` en cualquier máquina, correr **rsync bidireccional
de summary.json/trials.csv** para que ambas vean lo mismo:

```bash
# Pull Daniel → AIA blanco
rsync -avz --update --include='*/' --include='summary.json' --include='trials.csv' \
  --exclude='*' uniandes-personal-daniel:~/Documents/solar-forecasting-uq/runs/ \
  /srv/projects/Proyecto_e_ladino/runs/

# Push AIA blanco → Daniel
rsync -avz --update --include='*/' --include='summary.json' --include='trials.csv' \
  --exclude='*' /srv/projects/Proyecto_e_ladino/runs/ \
  uniandes-personal-daniel:~/Documents/solar-forecasting-uq/runs/
```

Ejecutar **antes** de cada `bash run_sequential.sh --launch <grupo> <sitio>` cuando
se cambia de máquina, y al menos 1×/día si ambas máquinas trabajan en paralelo.

**Convención de dir:** `--hours_ahead 1 → H6`, `--hours_ahead 3 → H18`,
`--hours_ahead 6 → H36` (H = pasos de 10 min).

---

## Estado (tras sync 2026-08-10)

Meta v2 = 12 combos (2 sitios × 3 horizontes × 2 seeds {42, 1}).
Meta mlp = 24 (4 seeds {42, 1, 7, 13}, protocolo original).

| Grupo | Hecho/Meta | Faltantes |
|---|---|---|
| resnet_v2 | 12 / 12 ✓ | — (cerrado: `elp h6 s1` terminó 2026-08-11 05:33) |
| gsage_v2 | 7 / 12 | `uni h1 s1`, `uni h3 s1`, `uni h3 s42`, `uni h6 s1`, `uni h6 s42` |
| mlp | 13 / 24 | El Paso × todos: `h1 s{1,7,13,42}` (s42 solo si no está), `h3 s{1,7,13,42}`, `h6 s{1,7,13,42}` |
| fusion_resnet | 4 / 12 | `elp h1 s{42,1}`, `elp h3 s{42,1}`, `elp h6 s42`, `uni h1 s{42,1}`, `uni h6 s{42,1}` |
| fusion_gsage | 0 / 12 | Todos — **bloqueado hasta gsage_v2 = 12/12** |
| SGLD (3 arches) | 0 / 72 | **Bloqueado hasta reunión con asesor** — método pendiente |

Extras que no forman parte de la meta (no borrar, quedan como suplementario):
resnet_v2 tiene runs Uniandes con seeds 7/13 del protocolo viejo (4×3=12 → cuenta 15/12).

---

## Restricciones de recursos (OOM safety)

Perfiles observados por trial (n_jobs=1):
- `resnet` / `fusion_resnet` / `mlp`: 2–4 GB.
- `gsage` / `fusion_gsage`: hasta 5.5 GB con `hidden_g=256, k_neighbors=16`.

**Reglas:**
- **AIA blanco (12 GB)**: máximo 1 job GPU a la vez con 4 spawn workers (los workers
  consumen ~5 CPU cores; dos jobs contendieron por CPU en la prueba del 2026-08-10).
  Coexistieron sin OOM (~1 GB c/u) pero con contención. No lanzar `nvidia-smi` en
  simultáneo con jobs (driver/library mismatch histórico, hoy responde OK — vigilar).
- **Daniel (16 GB)**: máximo 2 jobs GPU en simultáneo, y **nunca 2 de gsage** — el peor caso combinado (2 × 5.5 GB = 11 GB) deja solo 5 GB de headroom, insuficiente para spikes.
- Si un job actual usa ≥ 6 GB, **no** lanzar otro gsage/fusion_gsage encima.

---

## Colas activas (ejecutar en orden)

### Cola A — AIA blanco: fusion_resnet (prioridad del usuario)
Un job a la vez. Cada uno **~3 días** con `TRAIN_NUM_WORKERS=4` (validado 2026-08-10:
~9 min/epoch, 2.7× vs `=0`). Sync antes de lanzar el siguiente.

**Edit necesario en el script (aplicado 2026-08-10):** `scripts/06_resnet_lstm_optuna.py`
respeta la env var `TRAIN_NUM_WORKERS` (default 4, previo hardcoded a 4).
Lanzar con `env TRAIN_NUM_WORKERS=4 .venv/bin/python -u scripts/06_...`.

**Cola A paralelizada en 2 máquinas (2026-08-11):** AIA blanco toma seed42, Daniel[0]
toma seed1 del mismo horizonte, para no duplicar combos y cerrar el par h3 al doble.

1. `fusion_resnet elpaso h3 seed42` ← ⚡ **corriendo AIA blanco** (PID 9394, 4 workers)
2. `fusion_resnet elpaso h3 seed1`  ← ⚡ **corriendo Daniel[0]** (PID 238428, 4 workers hardcoded)
3. `fusion_resnet elpaso h1 seed42`
4. `fusion_resnet elpaso h1 seed1`
5. `fusion_resnet elpaso h6 seed42`
6. `fusion_resnet uniandes h1 seed42`
7. `fusion_resnet uniandes h1 seed1`
8. `fusion_resnet uniandes h6 seed42`
9. `fusion_resnet uniandes h6 seed1`

Comando plantilla:
```bash
cd /srv/projects/Proyecto_e_ladino
.venv/bin/python scripts/06_resnet_lstm_optuna.py --fusion \
  --site <sitio> --hours_ahead <H> --seed <S> \
  --n_trials 75 --runs_root runs/fusion_resnet_lstm \
  > logs/fusion_resnet_<sitio>_h<H>_s<S>.log 2>&1
```

### Cola E — AIA blanco en paralelo: mlp_optuna elpaso
Modelo FlatMLP muy ligero. Complementa la Cola A aprovechando los huecos de GPU
del fusion. 11 combos faltantes (`elpaso × {h1,h3,h6} × seeds {1,7,13,42→42 hecho}`).

**⚠ DEADLOCK con spawn workers (11-ago):** el mlp hardcodeaba `num_workers=4` spawn
en el objective de Optuna. Co-corriendo con el fusion (que también usa 4 spawn
workers), el mlp se colgó 2 h en `futex_do_wait` con 0 epochs, y sus 9 workers
girando a ~47 % CPU c/u **starvaron al fusion** (GPU cayó a 0 %). Fix: se parcheó
`06_mlp_optuna.py` (línea ~94) para respetar la env var `TRAIN_NUM_WORKERS`.
**Regla: en AIA blanco lanzar el mlp SIEMPRE con `TRAIN_NUM_WORKERS=0`** mientras
el fusion (spawn workers) corra en paralelo. Sin workers el mlp es liviano de
verdad y no pelea CPU.

Comando (dentro de tmux `solar_local:mlp`):
```bash
env TRAIN_NUM_WORKERS=0 bash run_sequential.sh mlp_optuna elpaso > logs/run_mlp_elpaso.out 2>&1
```

Estado 2026-08-11: ⏸ **DIFERIDO.** Con 0 workers el deadlock de spawn desapareció
(0 hijos), pero el mlp se colgó igual co-corriendo con el fusion: ambos son procesos
torch que abren ~16 threads OMP c/u → 32 threads sobre 16 cores (oversubscription),
y el mlp (ops cortas) se ahoga en `futex`. Al matarlo, el fusion aceleró 632→500 s/epoch.
**AIA blanco corre SOLO el fusion.** Para el mlp El Paso: correrlo (a) solo, cuando no
haya fusion en AIA, o (b) en Daniel cuando se libere una ventana, o (c) con
`OMP_NUM_THREADS=4` en ambos procesos si se quiere paralelizar (sin validar aún).

### Cola B — Daniel ventana [0] — resnet_v2 ✓ cerrado, ahora offload de fusion
- `resnet_v2 elpaso h6 seed1` terminó 2026-08-11 05:33 → resnet_v2 12/12.
- **Ventana [0] repurposada a fusion_resnet (offload de Cola A):** corriendo
  `fusion_resnet elpaso h3 seed1` (PID 238428) desde 2026-08-11. Datos verificados:
  `data/patches_v1/elpaso/P16` poblado (887 MB). NO poner `mlp_optuna elpaso` aquí
  (corre en AIA blanco Cola E → sería duplicado entre máquinas).
- Al terminar h3 s1: sync + encadenar el siguiente seed1 de fusion pendiente
  (`elpaso h1 s1`) mientras AIA blanco lleva los seed42.

### CAMINO CRÍTICO (plan 11-ago, ejecutado 12-ago con watchers tmux)
Cerrar gsage_v2 Uniandes (4 combos) desbloquea fusion_gsage (12). Solo se
paraleliza con 2 workers (AIA + Daniel); Daniel NO corre 2 gsage (OOM).
Partición **sin colisión**: Daniel = {h3 s42, h3 s1}, AIA = {h6 s42, h6 s1}.

**Implementado con watchers auto-encadenados en tmux (sobreviven desconexión),
en vez de monitores frágiles que morían al cerrar sesión:**
- **Daniel**: se mató el bash de run_sequential (PID 207105) para cortar la
  cadena en h3; el python h3 s42 (316973) sobrevivió reparentado. Ventana tmux
  `solar_runs:gs_h3s1` espera a 316973 y corre `gsage uni h3 s1`, luego para.
- **Daniel `solar_runs:fus_next`**: espera al fusion h3 s1 (238428) y encadena
  `fusion_resnet elpaso h1 s1` (para no dejar Daniel[0] ocioso; no colisiona).
- **AIA `solar_local:gs_h6`**: espera al fusion h3 s42 (29237), hace sync pull,
  corre `gsage uni h6 s42` → push → `gsage uni h6 s1` → push.

Logs de los watchers: `logs/gsage_watch_*.log`, `logs/fusion_watch_*.log`.
Progreso al 12-ago: AIA fusion h3 s42 41/75 (ETA ~16h); Daniel fusion h3 s1
53/75; gsage h3 s42 12/75. Estado combinado gsage uni: 2/6 (h1 s42, h1 s1).

### Cola C — Daniel ventana [1] (vacía tras kill del duplicado)
Lanzar en orden (uno a la vez, se encadenan solos con `run_sequential.sh`):

1. `gsage_v2 uniandes h1 seed42`
2. `gsage_v2 uniandes h1 seed1`
3. `gsage_v2 uniandes h3 seed42`
4. `gsage_v2 uniandes h3 seed1`
5. `gsage_v2 uniandes h6 seed42`
6. `gsage_v2 uniandes h6 seed1`

Comando:
```bash
ssh uniandes-personal-daniel 'cd ~/Documents/solar-forecasting-uq && \
  tmux send-keys -t solar_runs:1 "bash run_sequential.sh gsage_optuna_v2 uniandes" C-m'
```

Al llegar a 12/12 → arrancar `fusion_gsage` (Cola D).

### Cola D — desbloqueada cuando gsage_v2 = 12/12
Fusion_gsage 12 combos (2 sitios × 3 H × 2 seeds).
**Solo uno a la vez por máquina** (pico 5.5 GB por trial).

### Bloqueado
- **SGLD** (72 runs pendientes): NO arrancar hasta reunión con asesor —
  método epistémico aún abierto (fresh-init y warm-start descartados).

---

## Cómo actualizar este archivo

Después de cada sync + audit, actualizar la tabla "Estado" y tachar/reordenar
la cola. Regla: **si aparece un combo en curso o completado que no está en la
cola, hubo una desviación → documentarla en `PENDIENTES.md` y arreglar el
proceso, no el registro.**

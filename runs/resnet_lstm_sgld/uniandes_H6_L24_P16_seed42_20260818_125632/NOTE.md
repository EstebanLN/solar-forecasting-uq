# SGLD run — resnet · uniandes · h1 (H6) · seed42 — INTERRUMPIDO

**Fecha:** 2026-08-18 · **Estado:** interrumpido externamente en época 66/70
(sin `summary.json`). 9 checkpoints válidos guardados (`checkpoint_e0025.pt` …
`checkpoint_e0065.pt`); falta el 10º (`e0070`) y la inferencia de ensamble final.

## Configuración
- Warm-start: `runs/resnet_lstm_optuna_v2/uniandes_H6_L24_P16_seed42_20260608_211124/best_model.pt`
- `sgld_lr=1e-7` · `burn_in=20` · `sample_every=5` · `n_samples=10`
- Persistence test RMSE_day = 293.9 (referencia)
- Código: `src/solar_uq/train_sgld.py` refinado (monitorea la media del ensamble;
  emite coverage 80/90/95, NLL, CRPS, error-vs-cuantil-σ, `weight_l2_from_prev_sample`,
  norma de gradiente). La interrupción ocurrió antes de escribir el `summary.json`,
  así que esos diagnósticos post-hoc no quedaron persistidos.

## Hallazgo: la cadena DIVERGE (no converge lento)
Media del ensamble (`ens_val_rmse_day`) por checkpoint:

| checkpoint | n_ens | ens_val_rmse_day |
|---|---|---|
| e0030 | 2 | 379.0 |
| e0050 | 6 | 436.0 |
| e0055 | 7 | 436.5 |
| e0060 | 8 | 439.7 |
| e0065 | 9 | 432.0 |

La media **empeora** al agregar muestras y se mantiene **por encima de persistence
(294)** y del backbone (~250–290). El `val_rmse_day` de las muestras individuales
vaga entre 380 y 522, y el `train` loss sube de 0.9 a 1.6. Esto es divergencia de
la cadena (abandona la cuenca del warm-start), no exploración estacionaria (que
oscilaría alrededor de un centro estable).

## Causa de la interrupción
El log (`logs/sgld_resnet_uniandes_h1_s42.log`) termina limpio en época 66/70 sin
traceback; AIA no rebooteó (uptime > runtime). Interrupción externa (probable OOM
del sistema al lanzar otra tarea, o cierre de la ventana tmux). No es un fallo del
método en sí — pero el método ya mostraba divergencia mucho antes de cortarse.

## Qué sigue (decisión con el asesor)
El SGLD es el método propuesto por el asesor. Opciones:
1. **Reconfinar la cadena:** `sgld_prior_precision`↑ (default 100 insuficiente para
   confinar sobre 70 épocas), `sgld_lr`↓ a 1e-8.
2. **Pivotar a deep ensembles** sobre las semillas — también da incertidumbre
   epistémica; el título/abstract del artículo ("epistemic uncertainty", sin atarse
   a SGLD) ya lo soportan.

Los 9 checkpoints se conservan como evidencia; no borrar.

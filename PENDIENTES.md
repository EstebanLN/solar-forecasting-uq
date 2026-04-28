# Pendientes y estado del proyecto
_Actualizado: 2026-04-26_

---

## Estado actual de runs (servidor: /srv/projects/Proyecto_e_ladino/)

| Modelo               | Runs completos | Observaciones                                          |
|----------------------|---------------|--------------------------------------------------------|
| resnet_lstm          | 30 / 30       | 2 sitios × 3 horizontes × 5 semillas. Solo val metrics en summary.json (test metrics en servidor pero no subidas a git) |
| graphsage_lstm       | 30 / 30       | Ídem                                                   |
| resnet_lstm_optuna   | 6 / 24        | 2 sitios × 3H × 4 seeds = 24 objetivo. Sí tienen final_test en summary.json |
| graphsage_lstm_optuna| 5 / 24        | Ídem                                                   |
| sarima               | 1 / 2         | Solo elpaso. Falta uniandes.                           |

**Comando para seguir corriendo Optuna en el servidor:**
```bash
bash run_sequential.sh optuna           # ambas archs, ambos sitios
bash run_sequential.sh resnet_optuna uniandes   # solo resnet, solo uniandes
bash run_sequential.sh gsage_optuna elpaso      # solo gsage, solo elpaso
```

---

## Pendiente de implementar (UQ)

### Fase 1 — Conformal Prediction (empezada)
- [x] `src/solar_uq/conformal.py` — clase `SplitCP` con calibrate/predict/evaluate
- [x] `src/solar_uq/train.py` — función `collect_predictions(model, loader, normalizer, device)`
- [x] `scripts/07_conformal_explore.py` — exploración Split CP sobre un run existente
- [ ] Correr `07_conformal_explore.py` en el servidor sobre runs representativos
- [ ] Evaluar si cobertura_day ≈ target_coverage (si no → motiva métodos adaptativos)
- [ ] Decidir si implementar CQR (Conformalized Quantile Regression) para intervalos adaptativos

### Fase 2 — Variance Networks
- [ ] Implementar cabeza de varianza (output: μ, σ²) con loss NLL gaussiana
- [ ] Script de entrenamiento `05_variance_net_baseline.py`
- [ ] Evaluación: calibración, sharpness, CRPS

### Fase 3 — Cooperative Bayesian
- [ ] Definir qué variante: MC Dropout, Deep Ensemble, o Bayesian layers (BNN)
- [ ] Implementar y entrenar
- [ ] Evaluación: descomposición aleatoria/epistémica

### Fase 4 — Comparación y paper
- [ ] `scripts/08_results_table.py` — leer todos los summary.json y producir tabla
  comparativa (mean ± std por arch/site/horizonte) → CSV + LaTeX
- [ ] Script de figuras de barras comparativas (matplotlib estilo publicación)
- [ ] Guardar y_true / y_pred del test set en disco para plots detallados
  (scatter, serie de tiempo, distribución de errores por hora del día)
- [ ] SARIMA para uniandes (falta)

---

## Cambios recientes en scripts (2026-04-26)

### Scripts 05 y 06 — guardado de arch_hparams en checkpoint
Los cuatro scripts (`05_resnet_lstm_baseline.py`, `05_graphsage_lstm_baseline.py`,
`06_resnet_lstm_optuna.py`, `06_graphsage_lstm_optuna.py`) ahora guardan
`"arch_hparams"` en el meta del checkpoint `.pt`.

**Por qué:** `07_conformal_explore.py` necesita reconstruir el modelo exacto
al cargar un `.pt`. Sin arch_hparams el script cae en defaults hardcodeados
que no coinciden con los hiperparámetros reales de los runs Optuna.

**Impacto en runs ya hechos:** los 72 runs existentes NO tienen arch_hparams
en su .pt. Para esos, `07_conformal_explore.py` usa los defaults del CLI
(que sí coinciden con los baseline runs, pero NO con los Optuna).
Solución: pasar los hparams del Optuna manualmente vía CLI, o re-entrenar.

---

## Notas de resultados clave (Optuna, test set)

| Arch          | Sitio    | H   | skill_day |
|---------------|----------|-----|-----------|
| graphsage     | elpaso   | 6h  | **0.631** |
| resnet        | elpaso   | 3h  | 0.479     |
| graphsage     | uniandes | 6h  | 0.428     |
| graphsage     | uniandes | 1h  | 0.144–0.166 |
| resnet        | elpaso   | 1h  | negativo  |

H=1h es el horizonte más difícil (skill_day mixto o negativo).
SARIMA peor que persistencia en todos los horizontes (skill_day < 0 en elpaso).

---

## Notas de arquitectura

- **Split:** train 2022-2023 | val 2024-H1 | test 2024-H2→2025
- **Métrica primaria:** rmse_day (RMSE muestras diurnas, GHI ≥ 20 W/m²)
- **Baseline:** persistencia — ŷ(t+H) = GHI(t)
- **Frecuencia:** 10 min | Historia L=24 pasos (4h) | Semillas: 42, 1, 7, 13, 100
- **Datos en el servidor, nunca en git** (ver .gitignore)

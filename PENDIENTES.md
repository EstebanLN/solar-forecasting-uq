# Pendientes y estado del proyecto
_Actualizado: 2026-05-11_

---

## Estado actual de runs

| Modelo                | Runs completos | Observaciones                                                    |
|-----------------------|---------------|------------------------------------------------------------------|
| resnet_lstm           | 30 / 30       | 2 sitios × 3H × 5 seeds. Val metrics en summary.json            |
| graphsage_lstm        | 30 / 30       | Ídem                                                             |
| resnet_lstm_optuna    | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| graphsage_lstm_optuna | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| sarima                | **2 / 2**     | COMPLETO ✓ — uniandes corrido el 2026-05-11                      |
| mlp_optuna            | 0 / 24        | EN CURSO — 2 procesos paralelos lanzados 2026-05-11              |

`run_sequential.sh` MLP corriendo: `logs/run_mlp_uniandes.out` y `logs/run_mlp_elpaso.out`.

---

## Pendiente de implementar — Incertidumbre y métodos bayesianos

### Fase 1 — Conformal Prediction (implementada, falta correr)
- [x] `src/solar_uq/conformal.py` — clase `SplitCP` con calibrate/predict/evaluate
- [x] `src/solar_uq/train.py` — `collect_predictions(model, loader, normalizer, device)`
- [x] `scripts/07_conformal_explore.py` — exploración Split CP sobre un run existente
- [ ] Correr `07_conformal_explore.py` en runs representativos (graphsage elpaso h6 seed42)
- [ ] Evaluar cobertura_day ≈ target_coverage → decide si implementar CQR

### Fase 2 — Variance Networks (NLL)
- [ ] Cabeza de varianza (output μ, σ²) con loss NLL gaussiana en `src/solar_uq/`
- [ ] Script `09_variance_net.py` — entrenamiento y evaluación (calibración, sharpness, CRPS)

### Fase 3 — SGLD (Stochastic Gradient Langevin Dynamics) ← PRÓXIMO
- [ ] `scripts/10_sgld_train.py` — entrenamiento con inyección de ruido de Langevin
  - 1 000 épocas totales; guardar checkpoint cada 100 épocas (10 muestras de la posterior)
  - Guardar: epoch, loss_train, loss_val, params snapshot → `runs/sgld/{site}_{H}_{seed}/`
  - Estructura de guardado: `checkpoint_ep{epoch:04d}.pt` + `sgld_trace.json` con métricas
- [ ] Inferencia: ensemble sobre las 10 muestras → media (predicción) + std (incertidumbre epistémica)
- [ ] Evaluación: CRPS, cobertura de intervalos, skill_day con predicción SGLD
- [ ] Notebook `notebooks/06_sgld_posterior.ipynb` — visualizar cadenas de parámetros (convergencia, autocorrelación)
- [ ] Extraer parámetros de la distribución a posteriori:
  - Media y varianza por capa/parámetro
  - Diagnósticos de convergencia (R̂ de Gelman-Rubin si se corren varias cadenas)
  - Correlación entre parámetros relevantes (pesos de cabeza de salida)

### Fase 4 — Baseline MLP + Optuna
- [ ] `scripts/06_mlp_optuna.py` — red completamente conectada (sin graph, con/sin LSTM)
  - Hiperparámetros: n_layers, hidden_dim, dropout, lr, weight_decay
  - Mismo protocolo Optuna (n_trials=50, 4 seeds, 2 sitios, 3 horizontes)
- [ ] Integrar resultados en `runs/mlp_optuna/` con mismo formato summary.json

---

## Pendiente — Resultados y paper

### Tabla comparativa y figuras
- [ ] Correr `scripts/08_results_table.py` con todos los runs actuales → `results/summary.csv`
- [ ] Figuras de barras comparativas (matplotlib estilo publicación) por horizonte/sitio
- [ ] Guardar y_true / y_pred del test set en disco (scatter, serie de tiempo, distribución por hora)
- [x] SARIMA para uniandes (skill_day: h1=0.081, h3=0.353, h6=0.435)

### Documento de artículo
- [ ] Definir venue objetivo (NeurIPS workshop, Solar Energy journal, Applied Energy, etc.)
- [ ] Estructura base del LaTeX: Abstract, Intro, Related Work, Metodología, Experimentos, Conclusión
- [ ] Sección de datos: descripción El Paso / Uniandes, preprocesamiento, split temporal
- [ ] Sección de modelos: ResNet-LSTM, GraphSAGE-LSTM, MLP, SARIMA, persistencia
- [ ] Sección UQ: Conformal, Variance Net, SGLD posterior
- [ ] Tabla principal de resultados (RMSE, MAE, skill_day, CRPS) por arch/site/horizon
- [ ] Figuras: intervalos de predicción, descomposición incertidumbre

---

## Notas de resultados clave (Optuna, test set)

| Arch      | Sitio    | H   | skill_day      |
|-----------|----------|-----|----------------|
| graphsage | elpaso   | 6h  | **0.631**      |
| resnet    | elpaso   | 3h  | 0.479          |
| graphsage | uniandes | 6h  | 0.428          |
| graphsage | uniandes | 1h  | 0.144–0.166    |
| resnet    | elpaso   | 1h  | negativo       |

SARIMA peor que persistencia en todos los horizontes (skill_day < 0 en elpaso).

---

## Notas de arquitectura

- **Split:** train 2022-2023 | val 2024-H1 | test 2024-H2→2025
- **Métrica primaria:** rmse_day (RMSE muestras diurnas, GHI ≥ 20 W/m²)
- **Baseline:** persistencia — ŷ(t+H) = GHI(t)
- **Frecuencia:** 10 min | Historia L=24 pasos (4h) | Seeds Optuna: 42, 1, 7, 13
- **Datos en el servidor, nunca en git** (ver .gitignore)

---

## Cambios recientes en scripts (2026-05-11)

### SARIMA bug fix — `fit_and_forecast()` en `05_sarima_baseline.py`
statsmodels `forecast()` retorna `RangeIndex` cuando `k_train` no tiene frecuencia inferida.
Fix: detectar `isinstance(fc.index, pd.DatetimeIndex)` y reconstruir el índice desde `k_train.index[-1] + 1h`.

### FlatMLP baseline — nuevos archivos
- `src/solar_uq/models/mlp.py` — `FlatMLP`: spatial avg-pool sobre P×P, flatten L×C, MLP con LayerNorm
- `scripts/06_mlp_optuna.py` — HPO idéntico al de ResNet/GraphSAGE (50 trials, 2 sitios, 3H, 4 seeds)
- `run_sequential.sh` — nuevo grupo `mlp_optuna`

---

## Cambios recientes en scripts (2026-04-26)

Los cuatro scripts baseline/optuna guardan `"arch_hparams"` en el meta del checkpoint `.pt`
para permitir reconstrucción exacta del modelo en `07_conformal_explore.py`.
Los 72 runs anteriores al 2026-04-26 no tienen este campo → usar defaults CLI o re-entrenar.

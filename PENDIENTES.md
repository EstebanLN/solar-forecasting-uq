# Pendientes y estado del proyecto
_Actualizado: 2026-05-19_

---

## Estado actual de runs

| Modelo                | Runs completos | Observaciones                                                    |
|-----------------------|---------------|------------------------------------------------------------------|
| resnet_lstm           | 30 / 30       | 2 sitios × 3H × 5 seeds. Val metrics en summary.json            |
| graphsage_lstm        | 30 / 30       | Ídem                                                             |
| resnet_lstm_optuna    | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| graphsage_lstm_optuna | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| sarima                | **2 / 2**     | COMPLETO ✓ — uniandes corrido el 2026-05-11                      |
| mlp_optuna            | 9 / 24        | EN CURSO — H6 (1h) elpaso y uniandes completos (4 seeds); corriendo H18 (3h) |

`run_sequential.sh` MLP corriendo: elpaso H18 seed 42, uniandes H18 seed 1 (activos a 2026-05-19).
Completos: elpaso_H6 (seeds 42,1,7,13) + uniandes_H6 (seeds 42,1,7,13) + uniandes_H18 (seed 42).

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
- [x] Correr `scripts/08_results_table.py` → `results/summary.csv` (48 Optuna runs + SARIMA ambos sitios)
- [x] `scripts/09_paper_figures.py` — fig1 skill_day, fig2 RMSE_day, fig3 serie temporal (GraphSAGE elpaso 6h)
- [x] Figuras integradas en `docs/Artículo___Investigación/sections/results.tex`
- [ ] Guardar y_true / y_pred del test set en disco para todos los modelos (scatter, distribución por hora)
- [x] SARIMA para uniandes (skill_day: h1=0.081, h3=0.353, h6=0.435)

### Documento de artículo (`docs/Artículo___Investigación/`)
- [ ] Definir venue objetivo (NeurIPS workshop, Solar Energy journal, Applied Energy, etc.)
- [x] Estructura base del LaTeX: Abstract, Intro, Related Work, Metodología, Experimentos, Conclusión
- [x] Sección de datos: descripción El Paso / Uniandes, preprocesamiento, split temporal
- [x] Sección de modelos: ResNet-LSTM, GraphSAGE-LSTM, MLP (pending), SARIMA, persistencia
- [ ] Sección UQ: Conformal, Variance Net, SGLD posterior (cuando se implementen)
- [x] Tabla principal de resultados (RMSE_day, skill_day) por arch/site/horizon — valores 4 seeds definitivos
- [x] Figuras publicación: fig1 skill_day, fig2 RMSE_day, fig3 serie temporal
- [ ] Compilar PDF localmente (pdflatex no disponible en servidor)
- [ ] Revisar y actualizar sección MLP cuando runs terminen
- [ ] Añadir figura scatter y_true vs y_pred por modelo (opcional)

---

## Cambios pendientes — GraphSAGE v2 (sugeridos por asesor, 2026-05-19)

> Diseño detallado de la red a discutir en próxima reunión. Lo que sigue es el alcance acordado para implementar antes de esa reunión.

### 1. Grafo con pesos inversos a la distancia
**Archivo:** `src/solar_uq/models/graphsage_lstm.py`

- [x] Nueva función `build_weighted_edge_index(patch)` que reemplaza `build_edge_index_8n`:
  - Misma topología 8-conectada (sin cambio de vecinos)
  - Retorna `(edge_index, edge_weight)` donde `edge_weight[e] = 1 / d(u, v)`
  - Vecinos cardinales (↑↓←→): `d = 1.0`, peso `= 1.0`
  - Vecinos diagonales (×4): `d = √2`, peso `= 1/√2 ≈ 0.707`
  - `edge_weight` se registra como buffer en el modelo (`register_buffer`)

- [x] Modificar `GraphSAGELayer.forward()` para agregación ponderada:
  - Antes: `nei_mean = Σ x_j / count`
  - Después: `nei_mean = Σ(w_ij · x_j) / Σ(w_ij)` donde `w_ij` viene de `edge_weight`
  - Signature: `forward(x, edge_index, edge_weight=None)` — retrocompatible si `None`

- [x] Modificar `GraphSAGE_LSTM.forward()` para batching de `edge_weight`:
  - Función `_batch_edge_weight(edge_weight, batch_size)` → repite el vector E veces
  - Se pasa a cada `GraphSAGELayer` en el loop temporal

- [x] `_batch_edge_index` ya existe — añadir `_batch_edge_weight` análogo

### 2. Espacio de búsqueda Optuna ampliado
**Archivo:** `scripts/06_graphsage_lstm_optuna.py`

| Hiperparámetro | Antes | Después |
|---|---|---|
| `hidden_g` (encoder SAGE) | `{64, 96, 128, 192}` | `{64, 96, 128, 192, 256}` |
| `hidden_t` (LSTM hidden) | `{64, 96, 128}` | `{64, 96, 128, 192, 256}` |
| `l1_reg` | — (no existía) | `suggest_float(1e-6, 1e-3, log=True)` |
| `n_trials` default | 50 | **100** |

- [x] Añadir `l1_reg` al `objective(trial)` y pasarlo a `train_one_model`
- [x] Guardar `l1_reg` en `arch_hparams` del checkpoint y en `summary.json`

### 3. L1 regularization en el loop de entrenamiento
**Archivo:** `src/solar_uq/train.py`

- [x] Añadir parámetro `l1_reg: float = 0.0` a `train_one_model()`
- [x] En el loop: `loss = mse_loss + l1_reg * sum(p.abs().sum() for p in model.parameters())`
- [x] Retrocompatible: si `l1_reg=0.0` el comportamiento es idéntico al actual

### Pendiente de discutir en próxima reunión con asesor
- Topología del grafo: ¿k-NN con k como hiperparámetro (4–16) vs. 8-conectado ponderado?
- "Vecino óptimo" como hiperparámetro de Optuna
- Posibles cambios en la cabeza de salida o en el readout del grafo

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

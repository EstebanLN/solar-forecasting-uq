# Pendientes y estado del proyecto
_Actualizado: 2026-05-20_

---

## Estado actual de runs

| Modelo                | Runs completos | Observaciones                                                    |
|-----------------------|---------------|------------------------------------------------------------------|
| resnet_lstm           | 30 / 30       | 2 sitios × 3H × 5 seeds. Val metrics en summary.json            |
| graphsage_lstm        | 30 / 30       | Ídem                                                             |
| resnet_lstm_optuna    | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| graphsage_lstm_optuna | **24 / 24**   | COMPLETO ✓ — test metrics en summary.json                        |
| sarima                | **2 / 2**     | COMPLETO ✓ — uniandes corrido el 2026-05-11                      |
| mlp_optuna            | 10 / 24       | EN CURSO — H6 (1h) completo ambos sitios; H18 (3h) elpaso seed42 + uniandes seed42 done |
| resnet_lstm_optuna_v2 | 0 / 24        | PENDIENTE — lanzar `bash run_sequential.sh resnet_optuna_v2` |
| graphsage_lstm_optuna_v2 | 0 / 24     | PENDIENTE — lanzar `bash run_sequential.sh gsage_optuna_v2`  |

`run_sequential.sh` MLP corriendo: H18 seeds 1,7,13 y H36 para ambos sitios (activos a 2026-05-20).
Completos: elpaso_H6 (seeds 42,1,7,13) + uniandes_H6 (seeds 42,1,7,13) + elpaso_H18 seed42 + uniandes_H18 seed42.

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
- [x] `scripts/06_mlp_optuna.py` — `FlatMLP`: spatial avg-pool, flatten L×C, MLP con LayerNorm
  - Hiperparámetros: n_layers, hidden_dim, dropout, lr, weight_decay (n_trials=50)
  - Protocolo Optuna: 4 seeds, 2 sitios, 3 horizontes → `runs/mlp_optuna/`
- [ ] Esperar a 24/24 runs completos → actualizar tabla de resultados y sección MLP del artículo

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

## Cambios implementados — v2 (ambas arquitecturas, 2026-05-20)

> Sugeridos por asesor. Implementados completamente. Runs v2 pendientes de lanzar.

### GraphSAGE-LSTM v2 — `src/solar_uq/models/graphsage_lstm.py`

- [x] **`build_weighted_knn_edge_index(patch, k)`** — grafo k-NN ponderado:
  - Cada píxel conecta sus `k` vecinos más cercanos (distancia euclídea en la grilla 2D)
  - `edge_weight[e] = 1 / d(u, v)` — peso inverso a la distancia
  - Retrocompatibles: `build_edge_index_8n` y `build_weighted_edge_index` (k=8) se mantienen
- [x] **`_batch_edge_weight(edge_weight, batch_size, device)`** — replica los pesos para batch disjunto
- [x] **`GraphSAGELayer.forward(x, edge_index, edge_weight=None)`** — agregación ponderada:
  - `nei_mean = Σ(w_ij · x_j) / Σ(w_ij)` si `edge_weight` no es `None`; sin pesos si `None`
- [x] **`GraphSAGE_LSTM`** — acepta y registra `edge_weight` como buffer; lo batchea y pasa a cada capa SAGE

### ResNet-LSTM v2 — `src/solar_uq/models/resnet_lstm.py`

- [x] **`n_lstm_layers`** como parámetro del constructor (antes fijo en 1)
  - `nn.LSTM(num_layers=n_lstm_layers)`, `dropout` activo sólo si `n_lstm_layers > 1`

### L1 regularización — `src/solar_uq/train.py`

- [x] **`train_one_model(..., l1_reg: float = 0.0)`** — suma L1 al MSE loss:
  - `loss = mse_loss + l1_reg * Σ|w|` en cada paso de entrenamiento
  - Retrocompatible: `l1_reg=0.0` → comportamiento idéntico al anterior

### Optuna GraphSAGE v2 — `scripts/06_graphsage_lstm_optuna.py`

| Hiperparámetro    | v1 (viejo)              | v2 (implementado)                    |
|-------------------|-------------------------|--------------------------------------|
| `hidden_g`        | `{64, 96, 128, 192}`    | `{64, 96, 128, 192, 256}`            |
| `hidden_t`        | `{64, 96, 128}`         | `{64, 96, 128, 192, 256}`            |
| `n_lstm_layers`   | fijo en 1               | `{1, 2}`                             |
| `k_neighbors`     | no existía (k=8 fijo)   | `{4, 8, 12, 16}` ← **k-NN Optuna**  |
| `l1_reg`          | no existía              | `{0.0, 1e-5, 1e-4, 1e-3}` categórico|
| `n_trials` default| 50                      | **100**                              |
| `--runs_root`     | no existía              | CLI arg (default `graphsage_lstm_optuna`) |

- Grafo construido **por trial** según `k_neighbors` con `build_weighted_knn_edge_index(patch, k)`
- Resultado en `runs/graphsage_lstm_optuna_v2/` (no interfiere con v1)

### Optuna ResNet v2 — `scripts/06_resnet_lstm_optuna.py`

| Hiperparámetro    | v1 (viejo)           | v2 (implementado)                    |
|-------------------|----------------------|--------------------------------------|
| `emb_dim`         | `{64, 128, 192}`     | `{64, 128, 192, 256}`                |
| `hidden_t`        | `{64, 128, 192}`     | `{64, 128, 192, 256}`                |
| `n_lstm_layers`   | fijo en 1            | `{1, 2}`                             |
| `l1_reg`          | no existía           | `{0.0, 1e-5, 1e-4, 1e-3}` categórico|
| `n_trials` default| 50 (era 20 en v1)    | **100**                              |
| `--runs_root`     | no existía           | CLI arg (default `resnet_lstm_optuna`) |

- Resultado en `runs/resnet_lstm_optuna_v2/`

### Cómo lanzar los v2 (24 runs × 100 trials cada arquitectura)

```bash
# 4 procesos en paralelo: 2 sitios × 2 arquitecturas
nohup bash run_sequential.sh resnet_optuna_v2 elpaso   > logs/run_resnet_v2_elpaso.out   2>&1 &
nohup bash run_sequential.sh resnet_optuna_v2 uniandes > logs/run_resnet_v2_uniandes.out 2>&1 &
nohup bash run_sequential.sh gsage_optuna_v2  elpaso   > logs/run_gsage_v2_elpaso.out    2>&1 &
nohup bash run_sequential.sh gsage_optuna_v2  uniandes > logs/run_gsage_v2_uniandes.out  2>&1 &
```

> Estimado: ~4-5 días con GPU disponible (100 trials × 20 épocas × 24 combos).
> Los runs v1 no se tocan — `already_done()` distingue por directorio.

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





  nohup bash run_sequential.sh resnet_optuna_v2 elpaso   > logs/run_resnet_v2_elpaso.out
  2>&1 &
  nohup bash run_sequential.sh resnet_optuna_v2 uniandes > logs/run_resnet_v2_uniandes.out
  2>&1 &
  nohup bash run_sequential.sh gsage_optuna_v2  elpaso   > logs/run_gsage_v2_elpaso.out
  2>&1 &
  nohup bash run_sequential.sh gsage_optuna_v2  uniandes > logs/run_gsage_v2_uniandes.out
  2>&1 &

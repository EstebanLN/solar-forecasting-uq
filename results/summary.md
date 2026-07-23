# Results Summary


## ELPASO


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 206.0 | 0.000 | — |
| SARIMA | 117.4 | 0.425 | — |
| ResNet+LSTM (Optuna v2) | 140.8 | 0.317 | 1/2 |
| Fusion ResNet+LSTM | 117.0 | 0.432 | 1/2 |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| SARIMA | 142.9 | 0.653 | — |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| SARIMA | 145.6 | 0.743 | — |

## UNIANDES


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 293.9 | 0.000 | — |
| SARIMA | 196.1 | 0.331 | — |
| ResNet+LSTM | 250.3 ±5.3 | 0.148 ±0.018 | 5/5 ✓ |
| GraphSAGE+LSTM | 248.0 ±6.0 | 0.156 ±0.020 | 5/5 ✓ |
| MLP (Optuna) | 271.2 ±13.0 | 0.077 ±0.044 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 246.7 ±4.1 | 0.161 ±0.014 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 250.8 ±5.1 | 0.147 ±0.017 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 257.6 ±17.1 | 0.124 ±0.058 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 250.5 | 0.148 | 1/2 |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 405.0 | 0.000 | — |
| SARIMA | 242.0 | 0.410 | — |
| ResNet+LSTM | 256.2 ±3.3 | 0.368 ±0.008 | 5/5 ✓ |
| GraphSAGE+LSTM | 255.1 ±3.2 | 0.370 ±0.008 | 5/5 ✓ |
| MLP (Optuna) | 275.5 ±5.6 | 0.320 ±0.014 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 255.7 ±4.8 | 0.369 ±0.012 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 252.5 ±1.5 | 0.376 ±0.004 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 259.1 ±10.9 | 0.360 ±0.027 | 4/2 ✓ |
| Fusion ResNet+LSTM | 231.2 ±2.8 | 0.429 ±0.007 | 2/2 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 470.5 | 0.000 | — |
| SARIMA | 244.5 | 0.480 | — |
| ResNet+LSTM | 284.5 ±19.5 | 0.393 ±0.048 | 5/5 ✓ |
| GraphSAGE+LSTM | 274.2 ±5.1 | 0.418 ±0.011 | 5/5 ✓ |
| MLP (Optuna) | 282.0 ±2.0 | 0.401 ±0.004 | 3/4 |
| ResNet+LSTM (Optuna) | 297.9 ±8.8 | 0.368 ±0.019 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 274.4 ±3.9 | 0.417 ±0.008 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 288.5 ±4.0 | 0.387 ±0.009 | 2/2 ✓ |

# Results Summary


## ELPASO


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 204.4 | 0.000 | — |
| SARIMA | 447.9 | -1.205 | — |
| ResNet+LSTM | 238.7 ±5.3 | -0.168 ±0.026 | 5/5 ✓ |
| GraphSAGE+LSTM | 253.7 ±25.8 | -0.241 ±0.126 | 5/5 ✓ |
| MLP (Optuna) | 279.7 ±13.2 | -0.368 ±0.064 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 238.6 ±22.4 | -0.167 ±0.110 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 244.7 ±8.1 | -0.197 ±0.040 | 4/4 ✓ |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 411.0 | 0.000 | — |
| SARIMA | 450.4 | -0.093 | — |
| ResNet+LSTM | 216.6 ±10.4 | 0.473 ±0.025 | 5/5 ✓ |
| GraphSAGE+LSTM | 244.0 ±13.5 | 0.406 ±0.033 | 5/5 ✓ |
| MLP (Optuna) | 270.1 ±3.2 | 0.343 ±0.008 | 3/4 |
| ResNet+LSTM (Optuna) | 238.5 ±42.7 | 0.420 ±0.104 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 238.9 ±15.9 | 0.419 ±0.039 | 4/4 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 565.3 | 0.000 | — |
| SARIMA | 445.8 | 0.210 | — |
| ResNet+LSTM | 220.7 ±12.0 | 0.610 ±0.021 | 5/5 ✓ |
| GraphSAGE+LSTM | 230.4 ±11.3 | 0.592 ±0.020 | 5/5 ✓ |
| ResNet+LSTM (Optuna) | 218.4 ±13.2 | 0.614 ±0.023 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 206.1 ±7.4 | 0.636 ±0.013 | 4/4 ✓ |

## UNIANDES


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 293.9 | 0.000 | — |
| SARIMA | 269.3 | 0.081 | — |
| ResNet+LSTM | 250.3 ±5.3 | 0.148 ±0.018 | 5/5 ✓ |
| GraphSAGE+LSTM | 248.0 ±6.0 | 0.156 ±0.020 | 5/5 ✓ |
| MLP (Optuna) | 271.2 ±13.0 | 0.077 ±0.044 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 246.7 ±4.1 | 0.161 ±0.014 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 250.8 ±5.1 | 0.147 ±0.017 | 4/4 ✓ |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 405.0 | 0.000 | — |
| SARIMA | 265.1 | 0.353 | — |
| ResNet+LSTM | 256.2 ±3.3 | 0.368 ±0.008 | 5/5 ✓ |
| GraphSAGE+LSTM | 255.1 ±3.2 | 0.370 ±0.008 | 5/5 ✓ |
| MLP (Optuna) | 275.5 ±5.6 | 0.320 ±0.014 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 255.7 ±4.8 | 0.369 ±0.012 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 252.5 ±1.5 | 0.376 ±0.004 | 4/4 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 470.4 | 0.000 | — |
| SARIMA | 265.6 | 0.435 | — |
| ResNet+LSTM | 284.5 ±19.5 | 0.393 ±0.048 | 5/5 ✓ |
| GraphSAGE+LSTM | 274.2 ±5.1 | 0.418 ±0.011 | 5/5 ✓ |
| MLP (Optuna) | 282.2 | 0.401 | 1/4 |
| ResNet+LSTM (Optuna) | 297.9 ±8.8 | 0.368 ±0.019 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 274.4 ±3.9 | 0.417 ±0.008 | 4/4 ✓ |

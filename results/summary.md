# Results Summary


## ELPASO


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 206.0 | 0.000 | — |
| SARIMA | 117.4 | 0.425 | — |
| MLP (Optuna) | 169.4 | 0.178 | 1/4 |
| ResNet+LSTM (Optuna v2) | 144.9 ±5.8 | 0.297 ±0.028 | 2/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 146.0 ±5.2 | 0.291 ±0.025 | 2/2 ✓ |
| Fusion ResNet+LSTM | 116.8 ±0.3 | 0.433 ±0.001 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 116.4 ±0.8 | 0.435 ±0.004 | 2/2 ✓ |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 412.4 | 0.000 | — |
| SARIMA | 142.9 | 0.653 | — |
| ResNet+LSTM (Optuna v2) | 218.0 ±9.1 | 0.471 ±0.022 | 2/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 168.7 ±3.4 | 0.591 ±0.008 | 2/2 ✓ |
| Fusion ResNet+LSTM | 136.3 ±8.0 | 0.669 ±0.019 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 134.0 ±0.7 | 0.675 ±0.002 | 2/2 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 566.4 | 0.000 | — |
| SARIMA | 145.6 | 0.743 | — |
| ResNet+LSTM (Optuna v2) | 233.5 ±17.1 | 0.588 ±0.030 | 2/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 224.1 ±4.0 | 0.604 ±0.007 | 2/2 ✓ |
| Fusion ResNet+LSTM | 147.9 ±10.8 | 0.739 ±0.019 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 138.7 ±1.8 | 0.755 ±0.003 | 2/2 ✓ |
| ResNet+LSTM (SGLD) | 287.6 | 0.492 | — |

## UNIANDES


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 293.9 | 0.000 | — |
| SARIMA | 196.1 | 0.331 | — |
| ResNet+LSTM (baseline, pre-Optuna) | 250.3 ±5.3 | 0.148 ±0.018 | 5/5 ✓ |
| GraphSAGE+LSTM (baseline, pre-Optuna) | 248.0 ±6.0 | 0.156 ±0.020 | 5/5 ✓ |
| MLP (Optuna) | 271.2 ±13.0 | 0.077 ±0.044 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 246.7 ±4.1 | 0.161 ±0.014 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 250.8 ±5.1 | 0.147 ±0.017 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 257.6 ±17.1 | 0.124 ±0.058 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 246.7 ±5.4 | 0.161 ±0.018 | 2/2 ✓ |
| Fusion ResNet+LSTM | 213.8 ±0.8 | 0.273 ±0.003 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 213.9 ±1.9 | 0.272 ±0.007 | 2/2 ✓ |
| ResNet+LSTM (SGLD) | 271.8 | 0.075 | — |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 405.0 | 0.000 | — |
| SARIMA | 242.0 | 0.410 | — |
| ResNet+LSTM (baseline, pre-Optuna) | 256.2 ±3.3 | 0.368 ±0.008 | 5/5 ✓ |
| GraphSAGE+LSTM (baseline, pre-Optuna) | 255.1 ±3.2 | 0.370 ±0.008 | 5/5 ✓ |
| MLP (Optuna) | 275.5 ±5.6 | 0.320 ±0.014 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 255.7 ±4.8 | 0.369 ±0.012 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 252.5 ±1.5 | 0.376 ±0.004 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 259.1 ±10.9 | 0.360 ±0.027 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 257.6 ±0.2 | 0.364 ±0.001 | 2/2 ✓ |
| Fusion ResNet+LSTM | 230.8 ±6.6 | 0.430 ±0.016 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 232.7 ±1.2 | 0.426 ±0.003 | 2/2 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 470.6 | 0.000 | — |
| SARIMA | 244.5 | 0.480 | — |
| ResNet+LSTM (baseline, pre-Optuna) | 284.5 ±19.5 | 0.393 ±0.048 | 5/5 ✓ |
| GraphSAGE+LSTM (baseline, pre-Optuna) | 274.2 ±5.1 | 0.418 ±0.011 | 5/5 ✓ |
| MLP (Optuna) | 282.6 ±2.1 | 0.400 ±0.004 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 297.9 ±8.8 | 0.368 ±0.019 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 274.4 ±3.9 | 0.417 ±0.008 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 288.5 ±4.0 | 0.387 ±0.009 | 2/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 267.2 ±1.9 | 0.433 ±0.004 | 2/2 ✓ |
| Fusion ResNet+LSTM | 233.1 ±3.4 | 0.505 ±0.007 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 239.3 ±10.2 | 0.492 ±0.022 | 2/2 ✓ |
| ResNet+LSTM (SGLD) | 2434.6 | -4.170 | — |

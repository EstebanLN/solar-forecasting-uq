# Results Summary


## ELPASO


### 1 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 206.0 | 0.000 | — |
| SARIMA | 117.4 | 0.425 | — |
| MLP (Optuna) | 165.6 ±5.4 | 0.196 ±0.026 | 2/4 |
| ResNet+LSTM (Optuna v2) | 142.0 ±2.8 | 0.311 ±0.014 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 146.0 ±5.2 | 0.291 ±0.025 | 2/2 ✓ |
| Surface-only LSTM (Optuna) | 118.5 ±1.3 | 0.425 ±0.007 | 4/1 ✓ |
| Fusion ResNet+LSTM | 117.6 ±2.1 | 0.429 ±0.010 | 3/2 ✓ |
| Fusion GraphSAGE+LSTM | 116.2 ±0.6 | 0.436 ±0.003 | 2/2 ✓ |

### 3 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 412.4 | 0.000 | — |
| SARIMA | 142.9 | 0.653 | — |
| MLP (Optuna) | 217.1 ±4.9 | 0.474 ±0.012 | 2/4 |
| ResNet+LSTM (Optuna v2) | 173.7 ±7.8 | 0.579 ±0.019 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 168.6 ±3.5 | 0.591 ±0.009 | 2/2 ✓ |
| ConvGraph+LSTM | 173.2 ±9.4 | 0.580 ±0.023 | 4/2 ✓ |
| Surface-only LSTM (Optuna) | 137.7 ±7.7 | 0.666 ±0.019 | 4/1 ✓ |
| Fusion ResNet+LSTM | 131.6 ±0.4 | 0.681 ±0.001 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 132.3 ±5.0 | 0.679 ±0.012 | 2/2 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 566.4 | 0.000 | — |
| SARIMA | 145.6 | 0.743 | — |
| MLP (Optuna) | 276.6 ±9.1 | 0.512 ±0.016 | 2/4 |
| ResNet+LSTM (Optuna v2) | 238.2 ±17.1 | 0.579 ±0.030 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 236.4 ±21.6 | 0.583 ±0.038 | 3/2 ✓ |
| ConvGraph+LSTM | 216.2 ±1.5 | 0.618 ±0.003 | 4/2 ✓ |
| Surface-only LSTM (Optuna) | 140.4 ±3.2 | 0.752 ±0.006 | 4/1 ✓ |
| Fusion ResNet+LSTM | 147.9 ±10.8 | 0.739 ±0.019 | 2/2 ✓ |
| Fusion GraphSAGE+LSTM | 142.3 ±5.6 | 0.749 ±0.010 | 2/2 ✓ |
| ResNet+LSTM (SGLD) | 385.8 | 0.319 | — |

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
| ResNet+LSTM (Optuna v2) | 247.5 ±3.5 | 0.158 ±0.012 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 246.7 ±5.4 | 0.161 ±0.018 | 2/2 ✓ |
| Surface-only LSTM (Optuna) | 217.0 ±1.4 | 0.262 ±0.005 | 4/1 ✓ |
| Fusion ResNet+LSTM | 213.7 ±0.6 | 0.273 ±0.002 | 4/2 ✓ |
| Fusion GraphSAGE+LSTM | 214.4 ±0.8 | 0.270 ±0.003 | 3/2 ✓ |
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
| ResNet+LSTM (Optuna v2) | 256.3 ±7.5 | 0.367 ±0.018 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 257.3 ±5.5 | 0.365 ±0.013 | 3/2 ✓ |
| ConvGraph+LSTM | 259.6 | 0.359 | 1/2 |
| Surface-only LSTM (Optuna) | 242.4 ±5.2 | 0.402 ±0.013 | 4/1 ✓ |
| Fusion ResNet+LSTM | 229.2 ±2.7 | 0.434 ±0.007 | 4/2 ✓ |
| Fusion GraphSAGE+LSTM | 229.9 ±1.5 | 0.432 ±0.004 | 2/2 ✓ |

### 6 h

| Model | RMSE_day (W/m²) | Skill_day | Seeds |
|---|---|---|---|
| Persistence | 470.7 | 0.000 | — |
| SARIMA | 244.5 | 0.480 | — |
| ResNet+LSTM (baseline, pre-Optuna) | 284.5 ±19.5 | 0.393 ±0.048 | 5/5 ✓ |
| GraphSAGE+LSTM (baseline, pre-Optuna) | 274.2 ±5.1 | 0.418 ±0.011 | 5/5 ✓ |
| MLP (Optuna) | 282.6 ±2.1 | 0.400 ±0.004 | 4/4 ✓ |
| ResNet+LSTM (Optuna) | 297.9 ±8.8 | 0.368 ±0.019 | 4/4 ✓ |
| GraphSAGE+LSTM (Optuna) | 274.4 ±3.9 | 0.417 ±0.008 | 4/4 ✓ |
| ResNet+LSTM (Optuna v2) | 283.5 ±4.9 | 0.398 ±0.010 | 4/2 ✓ |
| GraphSAGE+LSTM (Optuna v2) | 274.5 ±6.2 | 0.417 ±0.013 | 3/2 ✓ |
| ConvGraph+LSTM | 289.0 ±11.7 | 0.386 ±0.025 | 4/2 ✓ |
| Surface-only LSTM (Optuna) | 239.5 ±6.5 | 0.491 ±0.014 | 4/1 ✓ |
| Fusion ResNet+LSTM | 235.3 ±7.6 | 0.500 ±0.016 | 4/2 ✓ |
| Fusion GraphSAGE+LSTM | 232.7 ±0.9 | 0.506 ±0.002 | 2/2 ✓ |
| ResNet+LSTM (SGLD) | 854.1 | -0.814 | — |

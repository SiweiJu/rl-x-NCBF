# Offline scan-critic probe

Static Go2 (zero action, lying on the ground), 2 moving obstacles, speeds 0.5-2.5 m/s, aim offset ±1.0 m, 64 rays over 360°, range noise ±0.02 m. 177899 train / 25965 validation / 52245 test samples, split by environment. Early stopping on validation log loss.

## H = 5 steps (0.10 s), unsafe rate 0.022

| input | AUROC | AP | test log loss | train log loss |
|---|---|---|---|---|
| scan_1 | 0.941 | 0.205 | 0.0708 | 0.0587 |
| scan_2 | 0.950 | 0.252 | 0.0665 | 0.0576 |
| scan_5 | 0.964 | 0.424 | 0.0600 | 0.0377 |
| scan_10 | 0.968 | 0.447 | 0.0557 | 0.0473 |
| scan_25 | 0.966 | 0.441 | 0.0586 | 0.0421 |
| scanconv_1 | 0.973 | 0.383 | 0.0533 | 0.0537 |
| scanconv_10 | 0.988 | 0.747 | 0.0360 | 0.0290 |
| scanconv_25 | 0.989 | 0.754 | 0.0348 | 0.0273 |
| state_pos | 0.939 | 0.217 | 0.0701 | 0.0656 |
| state_full | 0.984 | 0.627 | 0.0451 | 0.0426 |
| nearest_scan_return (no training) | 0.976 | 0.403 | – | – |
| constant_velocity_oracle (no training) | 1.000 | 0.981 | – | – |

## H = 25 steps (0.50 s), unsafe rate 0.110

| input | AUROC | AP | test log loss | train log loss |
|---|---|---|---|---|
| scan_1 | 0.814 | 0.296 | 0.2849 | 0.2574 |
| scan_2 | 0.820 | 0.315 | 0.2849 | 0.2454 |
| scan_5 | 0.860 | 0.474 | 0.2543 | 0.2144 |
| scan_10 | 0.888 | 0.544 | 0.2357 | 0.1944 |
| scan_25 | 0.903 | 0.612 | 0.2254 | 0.2240 |
| scanconv_1 | 0.841 | 0.345 | 0.2643 | 0.2542 |
| scanconv_10 | 0.959 | 0.805 | 0.1482 | 0.1211 |
| scanconv_25 | 0.961 | 0.823 | 0.1440 | 0.1033 |
| state_pos | 0.822 | 0.321 | 0.2762 | 0.2480 |
| state_full | 0.988 | 0.925 | 0.0856 | 0.0575 |
| nearest_scan_return (no training) | 0.842 | 0.356 | – | – |
| constant_velocity_oracle (no training) | 0.997 | 0.978 | – | – |

## H = 50 steps (1.00 s), unsafe rate 0.211

| input | AUROC | AP | test log loss | train log loss |
|---|---|---|---|---|
| scan_1 | 0.696 | 0.348 | 0.4800 | 0.4564 |
| scan_2 | 0.709 | 0.368 | 0.4741 | 0.4470 |
| scan_5 | 0.753 | 0.471 | 0.4510 | 0.4082 |
| scan_10 | 0.792 | 0.545 | 0.4296 | 0.3650 |
| scan_25 | 0.817 | 0.611 | 0.4097 | 0.3696 |
| scanconv_1 | 0.718 | 0.370 | 0.4616 | 0.4571 |
| scanconv_10 | 0.907 | 0.767 | 0.3013 | 0.2531 |
| scanconv_25 | 0.923 | 0.827 | 0.2765 | 0.1972 |
| state_pos | 0.706 | 0.366 | 0.4715 | 0.4444 |
| state_full | 0.959 | 0.864 | 0.2123 | 0.1277 |
| nearest_scan_return (no training) | 0.713 | 0.373 | – | – |
| constant_velocity_oracle (no training) | 0.988 | 0.967 | – | – |

## Linear probe of the last hidden layer (H = 25 critics, nearest visible obstacle)

| input | R² closing speed | R² velocity (x, y) | R² position (x, y) |
|---|---|---|---|
| scan_1 | -0.11 | -0.15, -0.17 | 0.39, 0.41 |
| scan_2 | -0.06 | -0.13, -0.13 | 0.40, 0.43 |
| scan_5 | 0.15 | -0.02, -0.06 | 0.45, 0.46 |
| scan_10 | 0.32 | 0.04, 0.03 | 0.55, 0.53 |
| scan_25 | 0.49 | 0.13, 0.14 | 0.55, 0.44 |
| scanconv_1 | -0.01 | -0.05, -0.04 | 0.00, -0.02 |
| scanconv_10 | 0.68 | -0.03, -0.04 | -0.01, -0.04 |
| scanconv_25 | 0.70 | -0.05, -0.05 | -0.02, -0.05 |
| state_pos | -0.46 | -0.62, -0.51 | 0.68, 0.54 |
| state_full | 0.37 | 0.57, 0.50 | 0.52, 0.48 |
| raw scan_1 input (no network) | -0.04 | -0.04, -0.04 | – |
| raw scan_10 input (no network) | 0.21 | -0.05, -0.04 | – |

## AUROC by speed of the nearest obstacle (H = 25)

| input | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 m/s |
|---|---|---|---|---|
| scan_1 | 0.897 | 0.834 | 0.825 | 0.758 |
| scan_2 | 0.888 | 0.848 | 0.833 | 0.769 |
| scan_5 | 0.900 | 0.884 | 0.871 | 0.839 |
| scan_10 | 0.919 | 0.916 | 0.889 | 0.877 |
| scan_25 | 0.907 | 0.928 | 0.904 | 0.901 |
| scanconv_1 | 0.925 | 0.853 | 0.851 | 0.790 |
| scanconv_10 | 0.963 | 0.967 | 0.969 | 0.967 |
| scanconv_25 | 0.965 | 0.970 | 0.971 | 0.959 |
| state_pos | 0.895 | 0.834 | 0.843 | 0.767 |
| state_full | 0.990 | 0.993 | 0.992 | 0.990 |

Figures: `ap_vs_history.png`, `counterfactual.png`.

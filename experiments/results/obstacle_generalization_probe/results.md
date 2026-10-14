# Obstacle randomization: held-out generalization of the collision critic

Static Go2, 2 obstacles, speeds 0.5-2.5 m/s, aim offset ±1.0 m, 64 rays, H = 25 steps. Trained on 128 envs x 2500 steps of the TRAIN distribution with early stopping; each test set has 64 new envs x 1500 steps. Overrides are in summary.json.

| test set | unsafe rate | scan_1 | scanconv_1 | scanconv_10_ensemble | state_full | ensemble std | std AUROC vs in-distribution |
|---|---|---|---|---|---|---|---|
| in_distribution | 0.082 | 0.841 / 0.265 | 0.877 / 0.334 | 0.953 / 0.699 | 0.972 / 0.783 | 0.0155 | – |
| held_out_shape | 0.099 | 0.830 / 0.278 | 0.861 / 0.352 | 0.945 / 0.677 | 0.959 / 0.739 | 0.0167 | 0.518 |
| held_out_motion | 0.032 | 0.924 / 0.263 | 0.944 / 0.310 | 0.980 / 0.702 | 0.992 / 0.829 | 0.0055 | 0.320 |
| held_out_size | 0.110 | 0.829 / 0.313 | 0.869 / 0.371 | 0.954 / 0.745 | 0.960 / 0.748 | 0.0177 | 0.530 |
| held_out_sensor | 0.078 | 0.806 / 0.216 | 0.838 / 0.238 | 0.892 / 0.472 | 0.974 / 0.769 | 0.0206 | 0.498 |

Cells are AUROC / average precision for "unsafe". The last column is the AUROC of ensemble
disagreement for telling that test set's samples from in-distribution samples (0.5 = no signal).

Single members of the ensemble:

| test set | member 0 | member 1 | member 2 |
|---|---|---|---|
| in_distribution | 0.951 | 0.948 | 0.950 |
| held_out_shape | 0.944 | 0.940 | 0.942 |
| held_out_motion | 0.978 | 0.977 | 0.978 |
| held_out_size | 0.951 | 0.951 | 0.952 |
| held_out_sensor | 0.880 | 0.884 | 0.878 |

Figure: `generalization.png`.

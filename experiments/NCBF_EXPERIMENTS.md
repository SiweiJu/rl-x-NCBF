# NCBF safety shield: fixes, obstacle task, and the experiment plan for IROS 2027

This note covers the `multistep` branch after the safety-critic fixes. It explains what changed,
which switches reproduce the old behavior, and how to run the planned experiments on the GPU cluster.
All runs use `ncbf_ppo.flax_full_jit` (or its `_booster` alias) and the MJX environment.

## 1. What changed in the safety critic

| Issue | Before | Now (default) | Legacy switch |
|---|---|---|---|
| Ensemble (B1) | One network trained, repeated 5 times; CVaR and epistemic std were no-ops | 5 independently initialized members with per-member bootstrap masks (`ensemble_bootstrap_prob=0.8`) | `algorithm.ncbf.independent_ensemble=False` |
| Action index (B2) | Critic trained on `(s_t, a_{t-1})`, queried by the shield on `(s_t, u)` | Trained on `(s_t, a_t)` with `a_t` the executed action, matching the shield | `algorithm.ncbf.critic_action_index=previous` |
| γ_c scale (B3) | Default `gamma_c=0` on a probability scale: shield almost never active | `gamma_c=0.5` (probability threshold); cluster default also 0.5 | pass `NCBF_GAMMA_C=0` |
| γ_c and η (B7) | Residual `h_next − (1−η)h − γ_c` | Paper form `h_next − (1−η)h − η·γ_c` | `algorithm.ncbf.gamma_c_scaled_by_eta=False` |
| Composite loss (B4, H1, H2) | BCE only; label discount commented out; temporal/Lipschitz code unused | Switchable: `label_discount`, `w_temporal` (safe-to-safe pairs only), `w_lip` (state part only by default) | defaults keep them off |
| Uncertainty (B6, A6) | σ only tightened via `β·σ`; slack ignored σ | `tighten_uncertainty` and `slack_uncertainty` ∈ {none, total, epistemic, aleatoric}; paper eq. (13) slack available | defaults: `total` / `none` (old behavior) |
| Anticipation loss (B8) | stop_gradient on the shield output | unchanged by default | `anticipation_through_shield=True` follows the paper text |
| Negative replay buffer | Only windows ending in a termination were stored | Also windows where any safety target drops (ball drop without termination, non-terminal collisions) | none |
| Evaluation | Only return and length | Training eval also logs failure rate, intervention rate, mean correction, collisions; `runner.mode=test` without rendering writes `eval_summary.json` | `environment.render=True` keeps the old interactive loop |

Checkpoints written before these changes load with the legacy settings automatically
(see `PPO.load`), so old models can still be evaluated.

About B1: only member 0 was trained in every commit of this repository, back to the oldest one
(`88de75c`, 2026-04-26), in both `flax` and `flax_full_jit`. No later commit broke it. A non-zero σ in
earlier runs is the logistic-normal output std, which is the aleatoric part only. The cluster
runner uses `NCBF_OUTPUT_DISTRIBUTION=logistic_normal` by default; the Python default and the
safety-layer debug scripts use `deterministic`, where σ was exactly 0. The new diagnostics `ncbf_safety_layer/residual_std_epistemic`
and `ncbf/ensemble_disagreement` show the ensemble part separately.

Not fixed: the non-jit `ncbf_ppo.flax` path (ensemble collapse at `flax/ppo.py:281`, shield disabled
in `test()`); it now logs a warning. Use the full-jit path for all experiments.

## 2. Moving-obstacle task

Enable with `--environment.obstacles.enabled=True` (or `OBSTACLES_ENABLED=True` in the cluster scripts).

- `nr_obstacles` obstacles spawn 3–5 m from the base and move toward it in the ground plane at
  `speed_range` m/s (lateral aim noise `aim_offset`; `aim_lead=True` aims at where the robot will be
  if it keeps its velocity). They respawn when they recede beyond `despawn_distance`, or when they
  outlive their expected arrival time plus `lifetime_margin_seconds` (capped at `max_lifetime_seconds`).
- Obstacles are kinematic: no contact forces. A collision is the safety event
  "an obstacle centre comes within `radius + robot_radius` of the base".
- Observations, chosen per network with `policy_observation`, `ncbf_observation` and
  `value_observation`:
  - `state`: per obstacle, relative position and velocity in the yaw frame and the radius
    (5 values, privileged);
  - `scan`: a planar range scan from the base (`scan.nr_rays` rays over `scan.fov_degrees`, up to
    `scan.max_range`, with range noise and dropout), given as proximity 1 − range / max_range;
  - `none`; the value critic can also take `state_and_scan`; `policy` means "same as the policy".
  The history encoder sees what the safety critic sees, so with `policy_observation=none` and
  `ncbf_observation=scan` only the safety critic (through the latent) knows about obstacles.
  `predict_next_observation=True` adds the safety critic's obstacle observation to the next-step
  predictor targets.
- Safety target: a `no_collision` indicator is appended to `ncbf_target_indices`
  (head name `collision`); `terminate_on_collision=True` by default, and collisions count as
  constraint cost for PPO-Lagrangian and CPO.
- `speed_curriculum=True` grows the upper speed with the environment curriculum; evaluation uses
  the full range. Ranges are tuples on the command line: `--environment.obstacles.speed_range='(2.0, 2.0)'`.

Limitations: no physical pushes from obstacles, not rendered in the viewer, and not added to the
plain MuJoCo environment yet.

## 3. Checks done in the cloud session (CPU, no GPU)

- `tests/test_safety_critic.py` (11 tests): residual with γ_c·η, linearized constraint restored by the
  projection, epistemic std zero for identical members and positive for distinct ones, σ-scaled
  slack equals the closed form, tightening grows the correction, obstacle observation layout,
  collision → termination with the `no_collision` indicator at 0, range-scan geometry (hit
  distance, yaw, occlusion, miss, out of range, inside), per-network observation routing, the scan
  inside the environment and the obstacle lifetime, and the GRU history encoder reading only its
  selected observations.
  Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_safety_critic.py -q`.
- Tiny training runs (16 envs, 2 PPO updates) for the old code, the fixed code with every new loss
  term enabled, and the fixed code with obstacles; a checkpoint save → load → batch evaluation at a
  different obstacle speed (2.5 m/s, spawn 1.5–2.5 m), once without and once with the shield
  (`eta_cbf=0.5`, `gamma_c=0.6`, `slack_uncertainty=epistemic`). Both wrote `eval_summary.json`.
  The untrained critic makes the shield intervene on every step at the `max_delta_u` cap, so the
  numbers say nothing about performance.
- A tiny training run with the scan (`policy_observation=scan`, `value_observation=state_and_scan`,
  `predict_next_observation=True`).
- The offline scan-critic probe of Step 3b (a supervised experiment, see there).

Apart from the offline probe, these only show that the code runs and the pieces behave as
intended. All closed-loop learning experiments below need the cluster.

## 4. Experiments to run on the cluster

Run in this order; each step has a check before spending compute on the next.

### Step 1 — sanity of the fixes (G0, ~1 day of GPU)
- `train_go2_locomotion_sl_on.sh` and `_sl_off.sh` with the new defaults.
- Check in wandb: `ncbf/ensemble_disagreement` > 0 and rising off-distribution,
  `ncbf/safety_layer_active_rate` between ~1% and 30%, `eval/failure_rate` lower with the shield.
- Re-run the existing η × γ_c sweep (`test_locomotion_safety_shield.sh`) with the fixes; note that
  with `gamma_c_scaled_by_eta=True` the γ_c values mean what the paper says.

### Step 2 — main comparisons (G1, ~140 runs)
Locomotion (Go2 and one humanoid) and ball-on-plate, 5 seeds each: PPO+DR, PPO-Lagrangian, CPO,
shield with H=1, shield with H=25, plus the uncertainty rules below. Existing scripts:
`train_*_sl_on.sh`, `train_*_sl_off.sh`, `run_constrained_baselines.sh`.

### Step 3 — moving obstacles (G2)
- Training: `train_go2_obstacles_sl_on.sh`, `train_go2_obstacles_sl_off.sh`,
  `train_ppo_lagrangian_go2_obstacles.sh`, `train_cpo_go2_obstacles.sh`.
- Horizon sweep: `sbatch cluster_runner/sweep_go2_obstacles_horizon.sh`
  (H ∈ {1, 10, 25, 50} × 3 seeds, plus shield off × 3 seeds).
- Speed sweep at test time for every trained model:
  `CHECKPOINT=.../models/latest.model RUN_NAME=<name> sbatch cluster_runner/eval_go2_obstacles_speed_sweep.sh`
  (speeds 0.5–4 m/s; set `USE_SAFETY_LAYER=False` to evaluate the same policy unshielded).
- Collect: `python ncbf_tools/collect_eval_summaries.py runs/202605_ncbf/obstacles_eval`.
- Key figure: collision rate vs. obstacle speed for each H; the gap between H=1 and longer
  horizons should grow with speed. If it does not, the "look further ahead" story is not supported
  on this task.

### Step 3b — obstacles seen only through a range scan
Question: from a history of range scans, can the H-step critic infer how obstacles move? A single
scan only gives positions, so any one-step barrier built on it (e.g. nearest return − margin) is
blind to obstacle velocity. If the critic recovers velocity from the history, a one-step CBF is
short-sighted in two ways (one step of time, one frame of perception), and the critic's
uncertainty should be high right after an obstacle appears, which links to Step 4.

Configuration (`train_go2_obstacles_scan_sl_on.sh`): `OBSTACLES_POLICY_OBSERVATION=scan` (the
safety critic and the history encoder get the same), `OBSTACLES_VALUE_OBSERVATION=state_and_scan`
(privileged state only in the value critic, which is not deployed), 64 rays over 360°, range
5 m, ±2 cm noise, `NR_HISTORY_STEPS=10`, `HISTORY_ENCODER_TYPE=scan_conv`.

**Offline probe (done on CPU).** `ncbf_tools/obstacle_scan_critic_probe.py`: the robot does not
move (zero action, it falls and lies on the ground), obstacles move past it, and small critics are
trained by supervised learning on H-step collision labels from different inputs, with early
stopping and envs split into train / validation / test. Full tables and figures:
`results/obstacle_scan_probe/` (CPU, ~35 min).

Results (128 envs × 2500 steps, 2 obstacles at 0.5–2.5 m/s, aim offset ±1 m; one seed, test on
held-out envs; AUROC / average precision for "unsafe"):

| input | H = 25 (0.5 s) | H = 50 (1.0 s) |
|---|---|---|
| nearest scan return, no training (one-step distance barrier) | 0.84 / 0.36 | 0.71 / 0.37 |
| single scan, MLP / circular conv | 0.81 / 0.30, 0.84 / 0.35 | 0.70 / 0.35, 0.72 / 0.37 |
| privileged position only | 0.82 / 0.32 | 0.71 / 0.37 |
| 10 scans, MLP / circular conv | 0.89 / 0.54, 0.96 / 0.81 | 0.79 / 0.55, 0.91 / 0.77 |
| 25 scans, MLP / circular conv | 0.90 / 0.61, 0.96 / 0.82 | 0.82 / 0.61, 0.92 / 0.83 |
| privileged position + velocity | 0.99 / 0.93 | 0.96 / 0.86 |

- A single scan is as good as the privileged position and no better than the nearest return: no
  velocity information, as expected.
- With a history, the convolutional critic gets most of the way to the privileged-velocity
  critic. Its AUROC stays at 0.96–0.97 across obstacle speeds, while the single-scan critic
  drops from 0.90 (0.5–1 m/s) to 0.76 (2–2.5 m/s).
- A linear probe of the last hidden layer recovers the closing speed of the nearest obstacle with
  R² 0.68–0.70 from the convolutional history critics, about 0 from single-scan critics, and
  0.21 from the raw stacked scans.
- Counterfactual (same current scan, different past): the history critics' P(safe) falls
  monotonically with the past approach speed. At 1.2 m the 25-scan critic drops near the true
  threshold (1.3 m/s); at 1.6 m it is too optimistic (0.75 at the 2.1 m/s threshold).
- Architecture matters: flattening the history into an MLP (what the FFNN encoder does) overfits
  and barely uses the history in the counterfactual. The cluster runs therefore use the
  `scan_conv` history encoder, with FFNN as a variant.

Caveats: the robot does not move, so there is no ego-motion; the probe's convolutional critic
pools over all rays, which makes it rotation invariant (right for a static round robot, while the
`scan_conv` encoder in PPO keeps 8 angular sectors); one seed; supervised labels from a fixed data
distribution rather than closed-loop training.

**Cluster sweep.** `sbatch cluster_runner/sweep_go2_obstacles_scan.sh` (11 variants × 3 seeds; run a
subset with `VARIANT_FILTER`): main; history 1 / 5 / 25; H 1 / 50; policy blind to obstacles with
only the safety critic seeing the scan; next-scan prediction; FFNN encoder instead of `scan_conv`;
privileged state for every network (upper bound); shield off. Evaluate each checkpoint with the speed sweep,
exporting the same `OBSTACLES_*` observation variables and `NR_HISTORY_STEPS` as in training.
Re-run the probe on the cluster with more data (`--nr_envs 1024`) and, once a walking policy
exists, with ego-motion.

Figures: collision rate vs. obstacle speed for main, history 1 and privileged state; the
counterfactual of the probe; σ_epistemic over time after an obstacle appears.
Support criterion: with the shield on, history > 1 beats history 1 at high speeds and approaches
the privileged-state run.

**Next: shape and motion randomization.** The scan makes this natural, because shape only enters
through the returns. Randomize size (0.1–1.0 m), shape (oriented boxes, composites of circles),
motion (turning, stop-and-go, acceleration) and the sensor (ray count, field of view, noise,
dropout, latency, spurious returns). Hold out shapes, sizes and motions for testing and report
collision rate and σ_epistemic on them; generalization claims need these held-out sets. Keep
size × speed combinations mostly avoidable, otherwise the failures are unavoidable rather than
missed (A7). Ray–box intersection and point–box distance are closed form, so this is ~1–2 days.

### Step 4 — uncertainty rules (A6)
Same task and seeds, vary only the shield:
- `NCBF_TIGHTEN_UNCERTAINTY` ∈ {none, total, epistemic, aleatoric}
- `NCBF_SLACK_UNCERTAINTY` ∈ {none, epistemic} (paper eq. 13)
- training-time β schedule via `NCBF_STD_COEFF_START` / `NCBF_STD_COEFF_FINAL`.
Evaluate at increasing shift (perturbation strength, terrain, obstacle speed beyond the training
range) and report failure rate vs. intervention rate. First check that epistemic std actually rises
under shift (log `ncbf_safety_layer/residual_std_epistemic`); if it does not, drop this direction.

### Step 5 — ablations
`NCBF_LABEL_DISCOUNT`, `NCBF_W_TEMPORAL`, `NCBF_W_LIP`, `NCBF_ANTICIPATION_THROUGH_SHIELD`,
`NCBF_POLICY_LOSS_COEF=0` (no anticipation loss), shield only at test time (train with
`USE_SAFETY_LAYER=False`, evaluate with `USE_SAFETY_LAYER=True`), and `NCBF_INDEPENDENT_ENSEMBLE=False`.

### Not implemented yet
- Counterfactual rollout check (E1): from saved MJX states, roll out H steps with the raw action
  and with the shielded action and compare realized failure rates.
- Multi-horizon safety heads (F1) and the critic-target comparison (A2: MC vs TD vs λ targets).
- Command-level shielding for obstacles (G2) and whole-body dodging (G3).
- Obstacle shape and motion randomization (boxes, composites, turning and stop-and-go motion) with
  held-out shapes and motions for testing; see Step 3b.

## 5. Compute estimate

Go2 at 4096 envs on an RTX 3090/A5000: plan for 4–8 GPU hours per 1e9-step run. Step 2 is
~140 runs, Step 3 ~25 training runs plus evaluations, Steps 4–5 ~80 runs: roughly
1–2k GPU hours in total.

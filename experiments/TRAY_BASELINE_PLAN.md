# G1 tray-carrying baseline: training and validation (step 0)

This is step 0 of the command-level obstacle task. The tray-carrying (ball on plate) locomotion policy
trained here becomes the **frozen low-level controller**: a later high-level policy sends it velocity
commands to walk to a goal while obstacles cross its path, and the safety shield filters those
commands. Before any of that is built, this policy has to exist and be shown to carry the ball reliably
at the commands the high level will send.

Scope: train and validate only. No shield, no obstacles, no hyperparameter tuning, no code changes.
If something in the code blocks the run, stop and report it with the log.

## 1. Code

- Repository: https://github.com/SiweiJu/rl-x-NCBF
- Branch: `claude/adoring-tesla-37hecd` (based on `multistep` at `ea3b048`; adds only this plan,
  `cluster_runner/train_g1_tray_baseline.sh`, `cluster_runner/eval_g1_tray_baseline.sh` and
  `ncbf_tools/eval_tray_baseline.py`; the training code is that of `ea3b048`).
- Use the exact commit given in the hand-off message and record `git rev-parse HEAD` in the report.
  Do not train from `multistep`'s tip: another session keeps changing the algorithm there.

## 2. Before submitting

All paths below are relative to `experiments/`.

1. Cluster: the scripts target the IAS SLURM cluster (`mn`, partition `main`; the older `gpu`
   partition in the other scripts no longer exists there). On another cluster, adjust
   `--partition` and `-C` on the command line.
2. `mkdir -p cluster_runner/log`. SLURM does not create it, and the job fails at start without it.
3. Conda env `rlx-ncbf` under `/home/ju/miniconda3` (the default in `run_cluster_experiment.sh`).
   Check it imports this commit: `PYTHONPATH=.. python -c "import rl_x.algorithms.ncbf_ppo.flax_full_jit"`.
4. W&B: the runs log to entity `catherineju-rwth-aachen-university`, project `202605_ncbf`,
   group `tray_baseline`. Check `wandb login` works on the node, or set `TRACK_WANDB=False`
   (TensorBoard still logs to the run directory).
5. GPUs: the array uses 3 GPUs; the per-user cap is 10 across all jobs. Check `squeue -u $USER`.
6. Disk: each run writes TensorBoard logs and one checkpoint (small). Check `df -h ~`.

## 3. Step 0a: training (3 seeds)

```bash
cd experiments
sbatch cluster_runner/train_g1_tray_baseline.sh   # array 0-2 -> seeds 42, 43, 44
```

What it runs: `run_cluster_experiment.sh` with G1, ball on plate enabled, the policy observing the
ball and plate (`BALL_PLATE_INCLUDE_OBSERVATIONS=True`), shield off (`USE_SAFETY_LAYER=False`),
curriculum return 40, and the runner defaults: 4096 envs, 1e9 steps, 20 s episodes, flat terrain,
`stand_after_drop=True`, no termination on drop.

Outputs: `runs/202605_ncbf/tray_baseline/LNg1_tray_base_s<seed>/`, with the checkpoint
`models/latest.model` (overwritten at every save).

Wall-clock: not measured for G1 with the ball. The Go2 estimate in `NCBF_EXPERIMENTS.md` is 4–8 GPU
hours per 1e9 steps; G1 with the ball is heavier, so the script asks for 24 h. If a job hits the limit,
resubmit that seed with `--runner.load_model` pointing at its `latest.model` and report that you did.

Watch in W&B / TensorBoard during training, and report the final values:
- episode return and length;
- `env_curriculum/coefficient`: should reach 1.0, otherwise the policy never saw the full command range;
- `env_info/ball_plate_drop_latched` (mean), `env_info/ball_plate_radial_distance`;
- the termination rate.

Stop a seed early only for NaNs or a return that has not moved after 2e8 steps, and say so.

## 4. Step 0b: validation

For each seed, after training:

```bash
cd experiments
RUN=runs/202605_ncbf/tray_baseline/LNg1_tray_base_s42
CHECKPOINT=$PWD/$RUN/models/latest.model OUTPUT=$PWD/$RUN/tray_eval.json SEED=1000 \
    sbatch cluster_runner/eval_g1_tray_baseline.sh
```

Run the three evaluations one after another, not in parallel on the same checkpoint: loading unpacks
the checkpoint into `models/tmp` and deletes it afterwards. Keep `NR_ENVS=4096` (the default): the
algorithm requires `nr_envs * nr_steps` to be a multiple of the minibatch size (32768), and fewer envs
than that fail at load time.

`ncbf_tools/eval_tray_baseline.py` holds the velocity command fixed per env (commands are fractions of
the per-axis maximum, `max_command_velocity * velocity_ratio`, printed in the output) and measures over
one 20 s episode per env, 4096 envs split over 19 profiles (about 215 envs each):

| profile | what it measures |
|---|---|
| 14 `grid_*` cells (standing, forward, backward, sideways, turning, combinations) | ball/plate drop rate, fall rate, tracking RMSE of vx, vy, wz (excluding the first second) |
| `step_vx_{0.3,0.6,1.0}`: 0 for 4 s, then forward | time until the mean forward velocity reaches 63 % / 90 % of the step (`t63_seconds`, `t90_seconds`), overshoot, drop rate |
| `brake_vx_{0.6,1.0}`: forward for 4 s, then 0 | stopping time (same fields) and drop rate when stopping hard |

An env stops counting at its first fall, drop or episode end.

The per-axis maxima with the default command config are vx 1.0 m/s, vy 0.5 m/s, wz 0.2 rad/s
(`max_command_velocity = min(0.5 * 2.0, 1.0)` and `velocity_ratio = [1, 0.5, 0.2]`). So the policy is
only trained to turn slowly; the obstacle task will avoid mainly by braking and side-stepping.

Smoke test (cloud session, CPU, 2026-10-14): a 256-step training run with 16 envs saved a G1 tray
checkpoint, and this script loaded it with 24 envs over a 4 s window and wrote the JSON. The
untrained policy drops the ball within about 0.05 s, so those numbers say nothing about the policy.

### Pass criteria (proposed, to be confirmed by Siwei)

On the in-range grid cells (largest axis fraction ≤ 0.6), the worst cell must have:

- ball/plate drop rate ≤ 5 % per 20 s episode;
- fall rate ≤ 1 %;
- linear velocity tracking RMSE ≤ 0.15 m/s, yaw-rate RMSE ≤ 0.2 rad/s.

`passed` in the JSON applies exactly these. The step and brake numbers have no threshold; they are the
measurements the next step needs (the low-level lag sets how far the shield must look ahead).

Pick the seed that passes with the lowest worst-cell drop rate (tie: lower linear RMSE). If no seed
passes, do not retrain or tune: report the numbers and stop.

## 5. What to send back

- The commit used (`git rev-parse HEAD`) and `git status` showing no local changes.
- Per seed: SLURM job id and `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`, GPU type
  and node, W&B run URL, run directory, checkpoint path and its `sha256sum`, total steps reached.
- The final training values listed in section 3.
- Each `tray_eval.json` (whole file) and the console summary the script prints.
- Which seed you recommend, and anything that went wrong, with the log lines.

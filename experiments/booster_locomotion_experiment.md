# Booster Locomotion Experiment

This records the fixed-gains Booster T1 locomotion experiment started on Blackwell on 2026-05-15.

## Git State

- Repository: `git@github.com:SiweiJu/rl-x-NCBF.git`
- Branch: `codex/booster-train-t1-model`
- Tagged commit: `0d3cdc17b8fed20fb91c7a99b0c3cfb90479e47c`
- Tag: `booster_locomotion`
- Commit subject: `Fix Booster torque-PD domain randomization gains`

The tagged commit fixes the BoosterTrain motor experiment by making the Booster runners use `seen_robot.type=booster` and by making the default seen-robot domain randomization torque-PD aware. This prevents motor-actuator configs from silently training with zero PD gains.

## Host And Scripts

- Host: `blackwell`
- Working tree: `~/repo/corl/rl-x-NCBF-booster-train-t1-model`
- Conda executable: `/home/siwei/miniconda3/bin/conda`
- Conda env: `loco-mjx`
- Safety layer off script: `experiments/cluster_runner/train_booster_locomotion_sl_off.sh`
- Safety layer on script: `experiments/cluster_runner/train_booster_locomotion_sl_on.sh`

## Effective Common Parameters

These are the shared effective parameters for both 4096-env runs.

```bash
--algorithm.name=ncbf_ppo.flax_full_jit
--algorithm.total_timesteps=1000000000
--algorithm.minibatch_size=32768
--algorithm.learning_rate=4e-4
--algorithm.adaptive_lr=True
--algorithm.nr_steps=128
--algorithm.nr_epochs=4
--algorithm.ncbf.output_distribution=logistic_normal
--algorithm.ncbf.gamma_c=0
--algorithm.ncbf.eta_cbf=1.0
--algorithm.ncbf.H=25
--algorithm.ncbf_buffer.neg_buffer_size=1
--algorithm.ncbf_buffer.neg_sampling_ratio=0.2
--algorithm.ncbf.action_clipping=False
--algorithm.ncbf.lr=3e-4
--algorithm.next_step_predictor.nr_minibatches=50
--algorithm.next_step_predictor.lr=1e-5
--algorithm.next_step_predictor.aux_loss_coef=1

--environment.name=ncbf_mujoco.robot_locomotion.mjx
--environment.seed=42
--environment.nr_envs=4096
--environment.terrain.type=plane
--environment.ncbf_use_policy_observations=True
--environment.train_robot=booster_t1
--environment.episode_length_in_seconds=20
--environment.env_curriculum_level_success_episode_return=30
--environment.reward.type=defaultG1
--environment.use_booster_defaults=False
--environment.domain_randomization.mujoco_model.type=none
--environment.domain_randomization.seen_robot.type=booster
--environment.domain_randomization.unseen_robot.type=none

--runner.mode=train
--runner.track_console=False
--runner.track_tb=True
--runner.track_wandb=True
--runner.save_model=True
--runner.wandb_entity=catherineju-rwth-aachen-university
--runner.project_name=202605_ncbf
--runner.exp_name=booster_locomotion_safety_shield
```

## Runs

| Safety layer | W&B run | Run name | Blackwell PID | Log |
| --- | --- | --- | --- | --- |
| Off | https://wandb.ai/catherineju-rwth-aachen-university/202605_ncbf/runs/m4bpu6mn | `BW4096_fixed_gains_0d3cdc1_20260515_192728_booster_locomotion_old_config_boostertrain_motor_substep_sl_off` | `1873559` | `experiments/cluster_runner/log/BW4096_fixed_gains_0d3cdc1_20260515_192728_sl_off.log` |
| On | https://wandb.ai/catherineju-rwth-aachen-university/202605_ncbf/runs/96prteg8 | `BW4096_fixed_gains_0d3cdc1_20260515_192728_booster_locomotion_old_config_boostertrain_motor_substep_sl_on` | `1873560` | `experiments/cluster_runner/log/BW4096_fixed_gains_0d3cdc1_20260515_192728_sl_on.log` |

## Launch Commands

The runs were started on Blackwell with:

```bash
cd ~/repo/corl/rl-x-NCBF-booster-train-t1-model

setsid env \
  CONDA_EXE=/home/siwei/miniconda3/bin/conda \
  CONDA_ENV=loco-mjx \
  RUN_NAME_PREFIX=BW4096_fixed_gains_0d3cdc1_20260515_192728_ \
  bash experiments/cluster_runner/train_booster_locomotion_sl_off.sh \
  > experiments/cluster_runner/log/BW4096_fixed_gains_0d3cdc1_20260515_192728_sl_off.log \
  2>&1 < /dev/null &

setsid env \
  CONDA_EXE=/home/siwei/miniconda3/bin/conda \
  CONDA_ENV=loco-mjx \
  RUN_NAME_PREFIX=BW4096_fixed_gains_0d3cdc1_20260515_192728_ \
  bash experiments/cluster_runner/train_booster_locomotion_sl_on.sh \
  > experiments/cluster_runner/log/BW4096_fixed_gains_0d3cdc1_20260515_192728_sl_on.log \
  2>&1 < /dev/null &
```


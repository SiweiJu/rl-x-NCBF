#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-booster_locomotion_old_config_boostertrain_motor_substep}"
CONDA_ENV="${CONDA_ENV:-loco_mjx}"

cd "${EXPERIMENTS_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python experiment.py \
    --algorithm.name="ncbf_ppo.flax_full_jit" \
    --algorithm.total_timesteps=1000000000 \
    --algorithm.minibatch_size=4096 \
    --algorithm.learning_rate=4e-4 \
    --algorithm.nr_steps=128 \
    --algorithm.nr_epochs=4 \
    --algorithm.ncbf.use_safety_layer=False \
    --algorithm.ncbf.output_distribution="logistic_normal" \
    --algorithm.ncbf.gamma_c=0 \
    --algorithm.ncbf.eta_cbf=1.0 \
    --algorithm.ncbf.H=25 \
    --algorithm.ncbf.action_clipping=False \
    --algorithm.ncbf_buffer.neg_buffer_size=1 \
    --algorithm.ncbf_buffer.neg_sampling_ratio=0.2 \
    --algorithm.ncbf.lr=3e-4 \
    --algorithm.next_step_predictor.nr_minibatches=50 \
    --algorithm.next_step_predictor.lr=1e-5 \
    --algorithm.next_step_predictor.aux_loss_coef=1 \
    --algorithm.adaptive_lr=True \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="booster_t1" \
    --environment.use_booster_defaults=False \
    --environment.reward.type="defaultG1" \
    --environment.domain_randomization.mujoco_model.type="none" \
    --environment.domain_randomization.seen_robot.type="booster" \
    --environment.domain_randomization.unseen_robot.type="none" \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds=20 \
    --environment.env_curriculum_level_success_episode_return=30 \
    --environment.terrain.type="plane" \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="catherineju-rwth-aachen-university" \
    --runner.project_name="202605_ncbf" \
    --runner.exp_name="booster_locomotion_safety_shield" \
    --runner.run_name="${RUN_NAME}"

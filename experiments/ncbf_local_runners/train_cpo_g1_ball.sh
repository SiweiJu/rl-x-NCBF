#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
NR_STEPS="${NR_STEPS:-128}"
NR_EPOCHS="${NR_EPOCHS:-4}"
EVALUATION_AND_SAVE_FREQUENCY="${EVALUATION_AND_SAVE_FREQUENCY:--1}"
COST_LIMIT="${COST_LIMIT:-0.0}"
PROJECT_NAME="${PROJECT_NAME:-debug}"
WANDB_ENTITY="${WANDB_ENTITY:-catherineju-rwth-aachen-university}"
RUN_NAME="${RUN_NAME:-cpo_g1_ball}"

cd "${EXPERIMENTS_DIR}"

conda run --no-capture-output -n ncbf-mjx python experiment.py \
    --algorithm.name="cpo.flax_full_jit" \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS}" \
    --algorithm.minibatch_size="${MINIBATCH_SIZE}" \
    --algorithm.learning_rate="${LEARNING_RATE}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs="${NR_EPOCHS}" \
    --algorithm.evaluation_and_save_frequency="${EVALUATION_AND_SAVE_FREQUENCY}" \
    --algorithm.cost_limit="${COST_LIMIT}" \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="unitree_g1" \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds=20 \
    --environment.env_curriculum_level_success_episode_return=40 \
    --environment.terrain.type="plane" \
    --environment.ball_plate.enabled=True \
    --environment.ball_plate.include_observations=True \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="g1_ball" \
    --runner.run_name="${RUN_NAME}"

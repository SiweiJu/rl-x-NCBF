#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-g1_locomotion}"

cd "${EXPERIMENTS_DIR}"

conda run --no-capture-output -n ncbf-mjx python experiment.py \
    --algorithm.name="ncbf_ppo.flax_full_jit" \
    --algorithm.total_timesteps=1000000000 \
    --algorithm.minibatch_size=4096 \
    --algorithm.learning_rate=4e-4 \
    --algorithm.nr_steps=128 \
    --algorithm.nr_epochs=4 \
    --algorithm.ncbf.use_safety_layer=False \
    --algorithm.ncbf.H=25 \
    --algorithm.ncbf.action_clipping=False \
    --algorithm.ncbf_buffer.neg_buffer_size=1 \
    --algorithm.ncbf.coef_decay_lambda=0.95 \
    --algorithm.next_step_predictor.nr_minibatches=50 \
    --algorithm.next_step_predictor.lr=1e-5 \
    --algorithm.next_step_predictor.aux_loss_coef=1 \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="unitree_g1" \
    --environment.reward.type="defaultG1" \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds=20 \
    --environment.env_curriculum_level_success_episode_return=50 \
    --environment.terrain.type="plane" \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="catherineju-rwth-aachen-university" \
    --runner.project_name="202603_ncbf_g1" \
    --runner.exp_name="debug"

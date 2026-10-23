#!/usr/bin/env bash
# Tiny CPU training run of ncbf_ppo.flax_full_jit on Go2 (16 envs), used by the resume test.
# Usage: RUN_NAME=<name> TOTAL_TIMESTEPS=<n> scripts/cpu_tiny_run.sh [extra flags]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"
RUN_NAME="${RUN_NAME:-cpu_tiny}"
NR_ENVS="${NR_ENVS:-16}"
NR_STEPS="${NR_STEPS:-16}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2048}"   # 8 learning iterations of 16 x 16 steps
EVAL_FREQ="${EVAL_FREQ:-1024}"               # 4 learning iterations per multi iteration
export JAX_PLATFORMS=cpu
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}/experiments"
exec "${PYTHON_EXE}" experiment.py \
    --algorithm.name=ncbf_ppo.flax_full_jit \
    --algorithm.device=cpu \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs=1 \
    --algorithm.minibatch_size=64 \
    --algorithm.evaluation_and_save_frequency="${EVAL_FREQ}" \
    --algorithm.evaluation_active=False \
    --algorithm.ncbf.use_safety_layer=True \
    --algorithm.ncbf.n_ensemble=2 \
    --algorithm.ncbf.H=5 \
    --algorithm.ncbf.w_temporal=1.0 \
    --algorithm.ncbf.nr_hidden_units=32 \
    --algorithm.next_step_predictor.nr_minibatches=2 \
    --environment.name=ncbf_mujoco.robot_locomotion.mjx \
    --environment.device=cpu \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.terrain.type=plane \
    --environment.episode_length_in_seconds=2 \
    --runner.mode=train \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=False \
    --runner.project_name=pncbf_cpu_test \
    --runner.exp_name=resume \
    --runner.run_name="${RUN_NAME}" \
    "$@"

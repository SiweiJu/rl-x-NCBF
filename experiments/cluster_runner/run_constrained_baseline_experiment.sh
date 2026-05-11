#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"

: "${ALGORITHM_NAME:?ALGORITHM_NAME must be set}"
: "${ENVIRONMENT_NAME:?ENVIRONMENT_NAME must be set}"
: "${TRAIN_ROBOT:?TRAIN_ROBOT must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"
: "${CURRICULUM_RETURN:?CURRICULUM_RETURN must be set}"
: "${EXP_NAME:?EXP_NAME must be set}"

CONDA_EXE="${CONDA_EXE:-/home/ju/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"
PYTHON_EXE="${PYTHON_EXE:-}"
SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
NR_STEPS="${NR_STEPS:-128}"
NR_EPOCHS="${NR_EPOCHS:-4}"
EVALUATION_AND_SAVE_FREQUENCY="${EVALUATION_AND_SAVE_FREQUENCY:--1}"
COST_LIMIT="${COST_LIMIT:-0.0}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
BALL_PLATE_ENABLED="${BALL_PLATE_ENABLED:-False}"
BALL_PLATE_STAND_AFTER_DROP="${BALL_PLATE_STAND_AFTER_DROP:-True}"
BALL_PLATE_POST_DROP_TRUNCATION_SECONDS="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS:-3.0}"
BALL_PLATE_TERMINATE_ON_DROP="${BALL_PLATE_TERMINATE_ON_DROP:-False}"
TERRAIN_TYPE="${TERRAIN_TYPE:-plane}"
PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
WANDB_ENTITY="${WANDB_ENTITY:-catherineju-rwth-aachen-university}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-LN}"
TRACK_CONSOLE="${TRACK_CONSOLE:-False}"
TRACK_TB="${TRACK_TB:-True}"
TRACK_WANDB="${TRACK_WANDB:-True}"
SAVE_MODEL="${SAVE_MODEL:-True}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

if [[ -z "${PYTHON_EXE}" ]]; then
    CONDA_ROOT="${CONDA_EXE%/bin/conda}"
    ENV_PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
    if [[ -x "${ENV_PYTHON}" ]]; then
        PYTHON_EXE="${ENV_PYTHON}"
    else
        eval "$("${CONDA_EXE}" shell.bash hook)"
        conda activate "${CONDA_ENV}"
        PYTHON_EXE="python"
    fi
fi

cd "${EXPERIMENTS_DIR}"

extra_args=()
if [[ -n "${REWARD_TYPE:-}" ]]; then
    extra_args+=(--environment.reward.type="${REWARD_TYPE}")
fi
if [[ -n "${EVALUATION_ACTIVE:-}" ]]; then
    extra_args+=(--algorithm.evaluation_active="${EVALUATION_ACTIVE}")
fi
if [[ "${BALL_PLATE_ENABLED}" == "True" ]]; then
    : "${BALL_PLATE_INCLUDE_OBSERVATIONS:?BALL_PLATE_INCLUDE_OBSERVATIONS must be set for ball experiments}"
    extra_args+=(
        --environment.termination.terminate_on_ball_plate_drop="${BALL_PLATE_TERMINATE_ON_DROP}"
        --environment.ball_plate.enabled=True
        --environment.ball_plate.stand_after_drop="${BALL_PLATE_STAND_AFTER_DROP}"
        --environment.ball_plate.post_drop_truncation_seconds="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS}"
        --environment.ball_plate.include_observations="${BALL_PLATE_INCLUDE_OBSERVATIONS}"
    )
fi

"${PYTHON_EXE}" experiment.py \
    --algorithm.name="${ALGORITHM_NAME}" \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS}" \
    --algorithm.minibatch_size="${MINIBATCH_SIZE}" \
    --algorithm.learning_rate="${LEARNING_RATE}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs="${NR_EPOCHS}" \
    --algorithm.evaluation_and_save_frequency="${EVALUATION_AND_SAVE_FREQUENCY}" \
    --algorithm.cost_limit="${COST_LIMIT}" \
    --environment.name="${ENVIRONMENT_NAME}" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="${TRAIN_ROBOT}" \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
    --environment.env_curriculum_level_success_episode_return="${CURRICULUM_RETURN}" \
    --environment.terrain.type="${TERRAIN_TYPE}" \
    ${extra_args[@]+"${extra_args[@]}"} \
    --runner.mode="train" \
    --runner.track_console="${TRACK_CONSOLE}" \
    --runner.track_tb="${TRACK_TB}" \
    --runner.track_wandb="${TRACK_WANDB}" \
    --runner.save_model="${SAVE_MODEL}" \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME_PREFIX}${RUN_NAME}"

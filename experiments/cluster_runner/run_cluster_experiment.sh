#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"

: "${ALGORITHM_NAME:?ALGORITHM_NAME must be set}"
: "${ENVIRONMENT_NAME:?ENVIRONMENT_NAME must be set}"
: "${TRAIN_ROBOT:?TRAIN_ROBOT must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"
: "${USE_SAFETY_LAYER:?USE_SAFETY_LAYER must be set}"
: "${CURRICULUM_RETURN:?CURRICULUM_RETURN must be set}"
: "${BALL_PLATE_ENABLED:?BALL_PLATE_ENABLED must be set}"

CONDA_EXE="${CONDA_EXE:-/home/ju/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"
SEED="${SEED:-42}"
NR_ENVS="${NR_ENVS:-4096}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32768}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
NR_STEPS="${NR_STEPS:-128}"
NR_EPOCHS="${NR_EPOCHS:-4}"
NCBF_LR="${NCBF_LR:-3e-4}"
NCBF_H="${NCBF_H:-25}"
NCBF_GAMMA_C="${NCBF_GAMMA_C:-0}"
NCBF_NEG_BUFFER_SIZE="${NCBF_NEG_BUFFER_SIZE:-1}"
NCBF_NEG_SAMPLING_RATIO="${NCBF_NEG_SAMPLING_RATIO:-0.2}"
NEXT_STEP_PREDICTOR_LR="${NEXT_STEP_PREDICTOR_LR:-1e-5}"
NEXT_STEP_PREDICTOR_NR_MINIBATCHES="${NEXT_STEP_PREDICTOR_NR_MINIBATCHES:-50}"
NEXT_STEP_PREDICTOR_AUX_LOSS_COEF="${NEXT_STEP_PREDICTOR_AUX_LOSS_COEF:-1}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
TERRAIN_TYPE="${TERRAIN_TYPE:-plane}"
PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
EXP_NAME="${EXP_NAME:-LN}"
WANDB_ENTITY="${WANDB_ENTITY:-catherineju-rwth-aachen-university}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-LN}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${EXPERIMENTS_DIR}"

extra_args=()
if [[ -n "${REWARD_TYPE:-}" ]]; then
    extra_args+=(--environment.reward.type="${REWARD_TYPE}")
fi
if [[ "${BALL_PLATE_ENABLED}" == "True" ]]; then
    : "${BALL_PLATE_INCLUDE_OBSERVATIONS:?BALL_PLATE_INCLUDE_OBSERVATIONS must be set for ball experiments}"
    extra_args+=(
        --environment.ball_plate.enabled=True
        --environment.ball_plate.include_observations="${BALL_PLATE_INCLUDE_OBSERVATIONS}"
    )
fi

python experiment.py \
    --algorithm.name="${ALGORITHM_NAME}" \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS}" \
    --algorithm.minibatch_size="${MINIBATCH_SIZE}" \
    --algorithm.learning_rate="${LEARNING_RATE}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs="${NR_EPOCHS}" \
    --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
    --algorithm.ncbf.target_distribution="logistic=_normal" \
    --algorithm.ncbf.gamma_c="${NCBF_GAMMA_C}" \
    --algorithm.ncbf.H="${NCBF_H}" \
    --algorithm.ncbf_buffer.neg_buffer_size="${NCBF_NEG_BUFFER_SIZE}" \
    --algorithm.ncbf_buffer.neg_sampling_ratio="${NCBF_NEG_SAMPLING_RATIO}" \
    --algorithm.ncbf.action_clipping=False \
    --algorithm.ncbf.lr="${NCBF_LR}" \
    --algorithm.next_step_predictor.nr_minibatches="${NEXT_STEP_PREDICTOR_NR_MINIBATCHES}" \
    --algorithm.next_step_predictor.lr="${NEXT_STEP_PREDICTOR_LR}" \
    --algorithm.next_step_predictor.aux_loss_coef="${NEXT_STEP_PREDICTOR_AUX_LOSS_COEF}" \
    --environment.name="${ENVIRONMENT_NAME}" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.terrain.type="${TERRAIN_TYPE}" \
    --environment.ncbf_use_policy_observations=True \
    --environment.train_robot="${TRAIN_ROBOT}" \
    --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
    --environment.env_curriculum_level_success_episode_return="${CURRICULUM_RETURN}" \
    "${extra_args[@]}" \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME_PREFIX}${RUN_NAME}"

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
PYTHON_EXE="${PYTHON_EXE:-}"
SEED="${SEED:-42}"
NR_ENVS="${NR_ENVS:-4096}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32768}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
ADAPTIVE_LR="${ADAPTIVE_LR:-True}"
NR_STEPS="${NR_STEPS:-128}"
NR_EPOCHS="${NR_EPOCHS:-4}"
NCBF_LR="${NCBF_LR:-3e-4}"
NCBF_H="${NCBF_H:-25}"
NCBF_GAMMA_C="${NCBF_GAMMA_C:-0}"
NCBF_ETA_CBF="${NCBF_ETA_CBF:-1.0}"
NCBF_OUTPUT_DISTRIBUTION="${NCBF_OUTPUT_DISTRIBUTION:-logistic_normal}"
NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
NCBF_LAMBDA_SLACK="${NCBF_LAMBDA_SLACK:-1000.0}"
NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
NCBF_SAFETY_LAYER_PROJECTION="${NCBF_SAFETY_LAYER_PROJECTION:-soft_slack}"
NCBF_POST_CHECK_ACTUAL_RESIDUAL="${NCBF_POST_CHECK_ACTUAL_RESIDUAL:-False}"
NCBF_NEG_BUFFER_SIZE="${NCBF_NEG_BUFFER_SIZE:-1}"
NCBF_NEG_SAMPLING_RATIO="${NCBF_NEG_SAMPLING_RATIO:-0.2}"
NEXT_STEP_PREDICTOR_LR="${NEXT_STEP_PREDICTOR_LR:-1e-5}"
NEXT_STEP_PREDICTOR_NR_MINIBATCHES="${NEXT_STEP_PREDICTOR_NR_MINIBATCHES:-50}"
NEXT_STEP_PREDICTOR_AUX_LOSS_COEF="${NEXT_STEP_PREDICTOR_AUX_LOSS_COEF:-1}"
USE_DECODER_OUTPUT_FOR_POLICY="${USE_DECODER_OUTPUT_FOR_POLICY:-False}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
BALL_PLATE_STAND_AFTER_DROP="${BALL_PLATE_STAND_AFTER_DROP:-True}"
BALL_PLATE_POST_DROP_TRUNCATION_SECONDS="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS:-3.0}"
BALL_PLATE_TERMINATE_ON_DROP="${BALL_PLATE_TERMINATE_ON_DROP:-False}"
TERRAIN_TYPE="${TERRAIN_TYPE:-plane}"
PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
EXP_NAME="${EXP_NAME:-locomotion_safety_shield}"
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
if [[ -n "${USE_BOOSTER_DEFAULTS:-}" ]]; then
    extra_args+=(--environment.use_booster_defaults="${USE_BOOSTER_DEFAULTS}")
fi
if [[ -n "${MUJOCO_MODEL_DR_TYPE:-}" ]]; then
    extra_args+=(--environment.domain_randomization.mujoco_model.type="${MUJOCO_MODEL_DR_TYPE}")
fi
if [[ -n "${EVALUATION_AND_SAVE_FREQUENCY:-}" ]]; then
    extra_args+=(--algorithm.evaluation_and_save_frequency="${EVALUATION_AND_SAVE_FREQUENCY}")
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
    --algorithm.adaptive_lr="${ADAPTIVE_LR}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs="${NR_EPOCHS}" \
    --algorithm.use_decoder_output_for_policy="${USE_DECODER_OUTPUT_FOR_POLICY}" \
    --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
    --algorithm.ncbf.output_distribution="${NCBF_OUTPUT_DISTRIBUTION}" \
    --algorithm.ncbf.gamma_c="${NCBF_GAMMA_C}" \
    --algorithm.ncbf.eta_cbf="${NCBF_ETA_CBF}" \
    --algorithm.ncbf.lambda_slack="${NCBF_LAMBDA_SLACK}" \
    --algorithm.ncbf.max_delta_u="${NCBF_MAX_DELTA_U}" \
    --algorithm.ncbf.safety_layer_projection="${NCBF_SAFETY_LAYER_PROJECTION}" \
    --algorithm.ncbf.post_check_actual_residual="${NCBF_POST_CHECK_ACTUAL_RESIDUAL}" \
    --algorithm.ncbf.H="${NCBF_H}" \
    --algorithm.ncbf_buffer.neg_buffer_size="${NCBF_NEG_BUFFER_SIZE}" \
    --algorithm.ncbf_buffer.neg_sampling_ratio="${NCBF_NEG_SAMPLING_RATIO}" \
    --algorithm.ncbf.action_clipping="${NCBF_ACTION_CLIPPING}" \
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
    --runner.track_console="${TRACK_CONSOLE}" \
    --runner.track_tb="${TRACK_TB}" \
    --runner.track_wandb="${TRACK_WANDB}" \
    --runner.save_model="${SAVE_MODEL}" \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME_PREFIX}${RUN_NAME}"

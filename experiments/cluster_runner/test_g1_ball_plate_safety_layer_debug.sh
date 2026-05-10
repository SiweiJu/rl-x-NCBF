#!/bin/bash
#SBATCH --job-name=g1_ball_sl_debug
#SBATCH --output=log/out_and_err_%x_%A_%a.txt
#SBATCH --error=log/out_and_err_%x_%A_%a.txt
#SBATCH --partition=gpu
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=9:59:59
#SBATCH --array=0-5

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
    mkdir -p "${SCRIPT_DIR}/log"
    echo "Submitting ${SCRIPT_PATH} to Slurm."
    cd "${SCRIPT_DIR}"
    exec sbatch "${SCRIPT_PATH}"
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi

sanitize() {
    local value="$1"
    value="${value//-/m}"
    value="${value//./p}"
    echo "${value}"
}

task_id="${SLURM_ARRAY_TASK_ID:-0}"

case "${task_id}" in
    0)
        safety_layer="False"
        action_clipping="False"
        max_delta_u="0.5"
        gamma_c="0.1"
        eta_cbf="0.5"
        run_name="g1_ball_sl_off"
        ;;
    1)
        safety_layer="True"
        action_clipping="False"
        max_delta_u="0.5"
        gamma_c="0.1"
        eta_cbf="0.5"
        run_name="g1_ball_sl_current"
        ;;
    2)
        safety_layer="True"
        action_clipping="True"
        max_delta_u="0.5"
        gamma_c="0.1"
        eta_cbf="0.5"
        run_name="g1_ball_sl_action_clip"
        ;;
    3)
        safety_layer="True"
        action_clipping="False"
        max_delta_u="1.0"
        gamma_c="0.1"
        eta_cbf="0.5"
        run_name="g1_ball_sl_cap_$(sanitize "${max_delta_u}")"
        ;;
    4)
        safety_layer="True"
        action_clipping="False"
        max_delta_u="0.5"
        gamma_c="0.2"
        eta_cbf="0.5"
        run_name="g1_ball_sl_gamma_$(sanitize "${gamma_c}")"
        ;;
    5)
        safety_layer="True"
        action_clipping="False"
        max_delta_u="0.5"
        gamma_c="0.1"
        eta_cbf="0.25"
        run_name="g1_ball_sl_eta_$(sanitize "${eta_cbf}")"
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID=${task_id} is outside the configured sweep size 6." >&2
        exit 2
        ;;
esac

export ALGORITHM_NAME="${ALGORITHM_NAME:-ncbf_ppo.flax_full_jit}"
export ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-ncbf_mujoco.robot_locomotion.mjx}"
export TRAIN_ROBOT="${TRAIN_ROBOT:-unitree_g1}"
export REWARD_TYPE="${REWARD_TYPE:-G1ball}"
export TERRAIN_TYPE="${TERRAIN_TYPE:-plane}"

export USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-${safety_layer}}"
export NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-${action_clipping}}"
export NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-${max_delta_u}}"
export NCBF_GAMMA_C="${NCBF_GAMMA_C:-${gamma_c}}"
export NCBF_ETA_CBF="${NCBF_ETA_CBF:-${eta_cbf}}"
export NCBF_OUTPUT_DISTRIBUTION="${NCBF_OUTPUT_DISTRIBUTION:-deterministic}"
export NCBF_LAMBDA_SLACK="${NCBF_LAMBDA_SLACK:-1000.0}"
export NCBF_H="${NCBF_H:-25}"
export NCBF_NEG_BUFFER_SIZE="${NCBF_NEG_BUFFER_SIZE:-1}"
export NEXT_STEP_PREDICTOR_LR="${NEXT_STEP_PREDICTOR_LR:-1e-5}"
export NEXT_STEP_PREDICTOR_AUX_LOSS_COEF="${NEXT_STEP_PREDICTOR_AUX_LOSS_COEF:-1}"
export USE_DECODER_OUTPUT_FOR_POLICY="${USE_DECODER_OUTPUT_FOR_POLICY:-True}"
export ADAPTIVE_LR="${ADAPTIVE_LR:-True}"

export BALL_PLATE_ENABLED="${BALL_PLATE_ENABLED:-True}"
export BALL_PLATE_INCLUDE_OBSERVATIONS="${BALL_PLATE_INCLUDE_OBSERVATIONS:-False}"
export BALL_PLATE_STAND_AFTER_DROP="${BALL_PLATE_STAND_AFTER_DROP:-True}"
export BALL_PLATE_POST_DROP_TRUNCATION_SECONDS="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS:-3.0}"
export BALL_PLATE_TERMINATE_ON_DROP="${BALL_PLATE_TERMINATE_ON_DROP:-False}"

export NR_ENVS="${NR_ENVS:-1024}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000000}"
export LEARNING_RATE="${LEARNING_RATE:-4e-4}"
export NR_STEPS="${NR_STEPS:-128}"
export NR_EPOCHS="${NR_EPOCHS:-4}"
export EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
export CURRICULUM_RETURN="${CURRICULUM_RETURN:-20}"

export RUN_NAME="${RUN_NAME:-${run_name}}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-SLDBG_}"
export PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
export EXP_NAME="${EXP_NAME:-sl_check}"

echo "Running G1 ball-plate safety-layer debug: task=${task_id}, use_safety_layer=${USE_SAFETY_LAYER}, action_clipping=${NCBF_ACTION_CLIPPING}, max_delta_u=${NCBF_MAX_DELTA_U}, gamma_c=${NCBF_GAMMA_C}, eta_cbf=${NCBF_ETA_CBF}, run_name=${RUN_NAME_PREFIX}${RUN_NAME}"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

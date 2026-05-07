#!/bin/bash
#SBATCH --job-name=g1_shield_test
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
#SBATCH --array=0-21

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

read -r -a ETA_CBFS <<< "${ETA_CBFS:-0.25 0.5 1.0}"
read -r -a GAMMA_CS <<< "${GAMMA_CS:-0.01 0.2 0.4 0.5 0.6 0.8 0.99}"

eta_count="${#ETA_CBFS[@]}"
gamma_count="${#GAMMA_CS[@]}"
shield_on_count="$((eta_count * gamma_count))"
combo_count="$((shield_on_count + 1))"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

if (( task_id >= combo_count )); then
    echo "SLURM_ARRAY_TASK_ID=${task_id} is outside the configured sweep size ${combo_count}." >&2
    echo "ETA_CBFS='${ETA_CBFS[*]}' GAMMA_CS='${GAMMA_CS[*]}'" >&2
    exit 2
fi

if (( task_id < shield_on_count )); then
    safety_layer="${USE_SAFETY_LAYER:-True}"
    eta_index="$((task_id / gamma_count))"
    gamma_index="$((task_id % gamma_count))"
    eta_cbf="${NCBF_ETA_CBF:-${ETA_CBFS[eta_index]}}"
    gamma_c="${NCBF_GAMMA_C:-${GAMMA_CS[gamma_index]}}"
    eta_tag="$(sanitize "${eta_cbf}")"
    gamma_tag="$(sanitize "${gamma_c}")"
else
    safety_layer="${USE_SAFETY_LAYER:-False}"
    eta_cbf="${NCBF_ETA_CBF:-1.0}"
    gamma_c="${NCBF_GAMMA_C:-0}"
    eta_tag=""
    gamma_tag=""
fi
if [[ -n "${USE_SAFETY_SHIELD:-}" ]]; then
    safety_layer="${USE_SAFETY_SHIELD}"
fi

case "${safety_layer}" in
    True|true|1)
        safety_tag="on"
        ;;
    False|false|0)
        safety_tag="off"
        ;;
    *)
        safety_tag="$(sanitize "${safety_layer}")"
        ;;
esac

export ALGORITHM_NAME="${ALGORITHM_NAME:-ncbf_ppo.flax_full_jit}"
export ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-ncbf_mujoco.robot_locomotion.mjx}"
export TRAIN_ROBOT="${TRAIN_ROBOT:-unitree_g1}"
export REWARD_TYPE="${REWARD_TYPE:-defaultG1}"

if [[ "${safety_tag}" == "off" ]]; then
    DEFAULT_RUN_NAME="g1_locomotion_sl_off"
else
    DEFAULT_RUN_NAME="g1_locomotion_sl_${safety_tag}_eta_${eta_tag}_gamma_${gamma_tag}"
fi

export USE_SAFETY_LAYER="${safety_layer}"
export NCBF_ETA_CBF="${eta_cbf}"
export NCBF_GAMMA_C="${gamma_c}"
export RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
export PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
export EXP_NAME="${EXP_NAME:-g1_locomotion_safety_shield}"
export CURRICULUM_RETURN="${CURRICULUM_RETURN:-30}"
export BALL_PLATE_ENABLED="${BALL_PLATE_ENABLED:-False}"

export NR_ENVS="${NR_ENVS:-4096}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-32768}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
export LEARNING_RATE="${LEARNING_RATE:-4e-4}"
export NR_STEPS="${NR_STEPS:-128}"
export NR_EPOCHS="${NR_EPOCHS:-4}"
export NCBF_LR="${NCBF_LR:-3e-4}"

echo "Running G1 locomotion shield sweep: use_safety_layer=${USE_SAFETY_LAYER}, eta_cbf=${NCBF_ETA_CBF}, gamma_c=${NCBF_GAMMA_C}, run_name=${RUN_NAME}"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

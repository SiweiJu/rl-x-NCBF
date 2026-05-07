#!/bin/bash
#SBATCH --job-name=loco_shield_test
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
#SBATCH --array=0-83

set -euo pipefail

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

read -r -a ROBOTS <<< "${ROBOTS:-go2 booster}"
read -r -a USE_SAFETY_LAYERS <<< "${USE_SAFETY_LAYERS:-True False}"
read -r -a ETA_CBFS <<< "${ETA_CBFS:-0.25 0.5 1.0}"
read -r -a GAMMA_CS <<< "${GAMMA_CS:-0.01 0.2 0.4 0.5 0.6 0.8 0.99}"

robot_count="${#ROBOTS[@]}"
safety_count="${#USE_SAFETY_LAYERS[@]}"
eta_count="${#ETA_CBFS[@]}"
gamma_count="${#GAMMA_CS[@]}"
combo_count="$((safety_count * robot_count * eta_count * gamma_count))"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

if (( task_id >= combo_count )); then
    echo "SLURM_ARRAY_TASK_ID=${task_id} is outside the configured sweep size ${combo_count}." >&2
    echo "USE_SAFETY_LAYERS='${USE_SAFETY_LAYERS[*]}' ROBOTS='${ROBOTS[*]}' ETA_CBFS='${ETA_CBFS[*]}' GAMMA_CS='${GAMMA_CS[*]}'" >&2
    exit 2
fi

safety_index="$((task_id / (robot_count * eta_count * gamma_count)))"
robot_index="$(((task_id / (eta_count * gamma_count)) % robot_count))"
eta_index="$(((task_id / gamma_count) % eta_count))"
gamma_index="$((task_id % gamma_count))"

if [[ -n "${USE_SAFETY_SHIELD:-}" ]]; then
    safety_layer="${USE_SAFETY_SHIELD}"
else
    safety_layer="${USE_SAFETY_LAYER:-${USE_SAFETY_LAYERS[safety_index]}}"
fi
robot="${ROBOT:-${ROBOTS[robot_index]}}"
eta_cbf="${NCBF_ETA_CBF:-${ETA_CBFS[eta_index]}}"
gamma_c="${NCBF_GAMMA_C:-${GAMMA_CS[gamma_index]}}"
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
eta_tag="$(sanitize "${eta_cbf}")"
gamma_tag="$(sanitize "${gamma_c}")"

case "${robot}" in
    0|go2|unitree_go2)
        export ALGORITHM_NAME="${ALGORITHM_NAME:-ncbf_ppo.flax_full_jit}"
        export ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-ncbf_mujoco.robot_locomotion.mjx}"
        export TRAIN_ROBOT="${TRAIN_ROBOT:-unitree_go2}"
        DEFAULT_RUN_NAME="go2_locomotion_sl_${safety_tag}_eta_${eta_tag}_gamma_${gamma_tag}"
        ;;
    1|booster|booster_t1)
        export ALGORITHM_NAME="${ALGORITHM_NAME:-ncbf_ppo.flax_full_jit_booster}"
        export ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-ncbf_mujoco.robot_locomotion.mjx_booster}"
        export TRAIN_ROBOT="${TRAIN_ROBOT:-booster_t1}"
        DEFAULT_RUN_NAME="booster_locomotion_sl_${safety_tag}_eta_${eta_tag}_gamma_${gamma_tag}"
        ;;
    *)
        echo "Unknown robot: ${robot}" >&2
        echo "Use ROBOT=go2 or ROBOT=booster, or submit the default array job." >&2
        exit 2
        ;;
esac

export USE_SAFETY_LAYER="${safety_layer}"
export NCBF_ETA_CBF="${eta_cbf}"
export NCBF_GAMMA_C="${gamma_c}"
export RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
export PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
export EXP_NAME="${EXP_NAME:-locomotion_safety_shield}"
export CURRICULUM_RETURN="${CURRICULUM_RETURN:-30}"
export BALL_PLATE_ENABLED="${BALL_PLATE_ENABLED:-False}"

export NR_ENVS="${NR_ENVS:-4096}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-32768}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
export LEARNING_RATE="${LEARNING_RATE:-4e-4}"
export NR_STEPS="${NR_STEPS:-128}"
export NR_EPOCHS="${NR_EPOCHS:-4}"
export NCBF_LR="${NCBF_LR:-3e-4}"

echo "Running locomotion shield sweep: robot=${TRAIN_ROBOT}, use_safety_layer=${USE_SAFETY_LAYER}, eta_cbf=${NCBF_ETA_CBF}, gamma_c=${NCBF_GAMMA_C}, run_name=${RUN_NAME}"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

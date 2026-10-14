#!/bin/bash
#SBATCH --job-name=go2_obst_scan
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
#SBATCH --array=0-32

# Go2 with moving obstacles seen through a 2D range scan: 11 variants x 3 seeds (33 array tasks).
# Each variant changes one thing relative to "main" (scan for policy and safety critic, history 10, H 25).
# Submit with: sbatch cluster_runner/sweep_go2_obstacles_scan.sh
# Run a subset with e.g. VARIANT_FILTER="main hist" (substring match on the variant name); the array
# index then counts over the filtered list.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
    mkdir -p "${SCRIPT_DIR}/../log"
    cd "${SCRIPT_DIR}/.."
    exec sbatch "${SCRIPT_PATH}"
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi

# name | settings applied on top of the main configuration
ALL_VARIANTS=(
    "main|"
    "hist1|NR_HISTORY_STEPS=1"
    "hist5|NR_HISTORY_STEPS=5"
    "hist25|NR_HISTORY_STEPS=25"
    "H1|NCBF_H=1"
    "H50|NCBF_H=50"
    "policy_blind|OBSTACLES_POLICY_OBSERVATION=none OBSTACLES_NCBF_OBSERVATION=scan"
    "predict_scan|OBSTACLES_PREDICT_NEXT_OBSERVATION=True"
    "scan_conv_encoder|HISTORY_ENCODER_TYPE=scan_conv"
    "oracle_state|OBSTACLES_POLICY_OBSERVATION=state OBSTACLES_VALUE_OBSERVATION=state"
    "shield_off|USE_SAFETY_LAYER=False"
)
VARIANTS=()
for variant in "${ALL_VARIANTS[@]}"; do
    name="${variant%%|*}"
    if [[ -z "${VARIANT_FILTER:-}" ]]; then
        VARIANTS+=("${variant}")
    else
        for pattern in ${VARIANT_FILTER}; do
            if [[ "${name}" == *"${pattern}"* ]]; then
                VARIANTS+=("${variant}")
                break
            fi
        done
    fi
done
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
nr_seeds="${#SEEDS[@]}"
if (( task_id >= ${#VARIANTS[@]} * nr_seeds )); then
    echo "SLURM_ARRAY_TASK_ID=${task_id} is outside the sweep (${#VARIANTS[@]} variants x ${nr_seeds} seeds)" >&2
    exit 2
fi
variant="${VARIANTS[$(( task_id / nr_seeds ))]}"
seed="${SEEDS[$(( task_id % nr_seeds ))]}"
variant_name="${variant%%|*}"
variant_settings="${variant#*|}"

# Main configuration
export USE_SAFETY_LAYER="True"
export NCBF_H="25"
export NR_HISTORY_STEPS="10"
export OBSTACLES_POLICY_OBSERVATION="scan"
export OBSTACLES_NCBF_OBSERVATION="policy"
export OBSTACLES_VALUE_OBSERVATION="state_and_scan"
for setting in ${variant_settings}; do
    export "${setting}"
done

export SEED="${seed}"
export RUN_NAME="go2_obstacles_scan_${variant_name}_seed${seed}"
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="unitree_go2"
export EXP_NAME="${EXP_NAME:-obstacles_scan}"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"
export OBSTACLES_ENABLED="True"
export OBSTACLES_NR="${OBSTACLES_NR:-2}"
export OBSTACLES_SPEED_RANGE="${OBSTACLES_SPEED_RANGE:-(0.5, 3.0)}"
export OBSTACLES_SPEED_CURRICULUM="${OBSTACLES_SPEED_CURRICULUM:-True}"

echo "Running ${RUN_NAME}: ${variant_settings:-main configuration}"
"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

#!/bin/bash
#SBATCH --job-name=go2_obst_H
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
#SBATCH --array=0-14

# Horizon sweep on Go2 with moving obstacles: H in {1, 10, 25, 50} with the shield on, plus the shield off,
# each for 3 seeds (15 array tasks). Submit with: sbatch cluster_runner/sweep_go2_obstacles_horizon.sh
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

read -r -a HORIZONS <<< "${HORIZONS:-1 10 25 50}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
nr_seeds="${#SEEDS[@]}"
nr_shield_on="$(( ${#HORIZONS[@]} * nr_seeds ))"

if (( task_id < nr_shield_on )); then
    horizon="${HORIZONS[$(( task_id / nr_seeds ))]}"
    seed="${SEEDS[$(( task_id % nr_seeds ))]}"
    export USE_SAFETY_LAYER="True"
    export NCBF_H="${horizon}"
    export RUN_NAME="go2_obstacles_sl_on_H${horizon}_seed${seed}"
elif (( task_id < nr_shield_on + nr_seeds )); then
    seed="${SEEDS[$(( task_id - nr_shield_on ))]}"
    export USE_SAFETY_LAYER="False"
    export RUN_NAME="go2_obstacles_sl_off_seed${seed}"
else
    echo "SLURM_ARRAY_TASK_ID=${task_id} is outside the sweep" >&2
    exit 2
fi
export SEED="${seed}"
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="unitree_go2"
export EXP_NAME="${EXP_NAME:-obstacles_horizon}"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"
export OBSTACLES_ENABLED="True"
export OBSTACLES_NR="${OBSTACLES_NR:-2}"
export OBSTACLES_SPEED_RANGE="${OBSTACLES_SPEED_RANGE:-(0.5, 3.0)}"
export OBSTACLES_SPEED_CURRICULUM="${OBSTACLES_SPEED_CURRICULUM:-True}"

echo "Running ${RUN_NAME} (H=${NCBF_H:-n/a}, seed=${SEED})"
"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

#!/bin/bash
#SBATCH --job-name=booster_loco_ms_sl_off
#SBATCH --output=log/out_and_err_%x_%j.txt
#SBATCH --error=log/out_and_err_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=9:59:59

set -euo pipefail

CLUSTER_REPO_DIR="${CLUSTER_REPO_DIR:-/home/ju/repo/corl}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
BRANCH_NAME="${BRANCH_NAME:-multistep}"

mkdir -p "${CLUSTER_REPO_DIR}/experiments/log"

(
    flock 9
    git -C "${CLUSTER_REPO_DIR}" fetch "${REMOTE_NAME}"
    if git -C "${CLUSTER_REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        git -C "${CLUSTER_REPO_DIR}" switch "${BRANCH_NAME}"
    else
        git -C "${CLUSTER_REPO_DIR}" switch --track -c "${BRANCH_NAME}" "${REMOTE_NAME}/${BRANCH_NAME}"
    fi
    git -C "${CLUSTER_REPO_DIR}" merge --ff-only "${REMOTE_NAME}/${BRANCH_NAME}"
    git -C "${CLUSTER_REPO_DIR}" status --short --branch
) 9>"${CLUSTER_REPO_DIR}/.git/codex_${BRANCH_NAME}_update.lock"

export ALGORITHM_NAME="ncbf_ppo.flax_full_jit_booster"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx_booster"
export TRAIN_ROBOT="booster_t1"
export RUN_NAME="booster_locomotion_multistep_sl_off"
export USE_SAFETY_LAYER="False"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"

"${CLUSTER_REPO_DIR}/experiments/cluster_runner/run_cluster_experiment.sh"

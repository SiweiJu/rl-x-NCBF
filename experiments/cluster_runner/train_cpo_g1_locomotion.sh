#!/bin/bash
#SBATCH --job-name=cpo_g1_loco
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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_constrained_baseline_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_constrained_baseline_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi

export ALGORITHM_NAME="cpo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="unitree_g1"
export REWARD_TYPE="defaultG1"
export RUN_NAME="cpo_g1_locomotion"
export CURRICULUM_RETURN="30"
export EXP_NAME="baselines"

"${CLUSTER_RUNNER_DIR}/run_constrained_baseline_experiment.sh"

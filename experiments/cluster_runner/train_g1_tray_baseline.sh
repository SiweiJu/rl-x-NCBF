#!/bin/bash
#SBATCH --job-name=g1_tray_base
#SBATCH --output=log/out_and_err_%x_%A_%a.txt
#SBATCH --error=log/out_and_err_%x_%A_%a.txt
#SBATCH --partition=main
#SBATCH -C 'rtx3090|a5000|vram48gb'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=24:00:00
#SBATCH --array=0-2

# G1 tray-carrying (ball on plate) locomotion baseline: no shield, the policy observes the ball and plate.
# This policy becomes the frozen low-level controller of the command-level obstacle task, so it is trained
# once, with 3 seeds (42, 43, 44 via the array index), and validated with eval_g1_tray_baseline.sh.
# See experiments/TRAY_BASELINE_PLAN.md. Create log/ next to this script before submitting.
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi
export SEED="${SEED:-$((42 + ${SLURM_ARRAY_TASK_ID:-0}))}"
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="unitree_g1"
export RUN_NAME="${RUN_NAME:-g1_tray_base_s${SEED}}"
export EXP_NAME="${EXP_NAME:-tray_baseline}"
export USE_SAFETY_LAYER="False"
export CURRICULUM_RETURN="40"
export BALL_PLATE_ENABLED="True"
export BALL_PLATE_INCLUDE_OBSERVATIONS="True"
export OBSTACLES_ENABLED="False"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

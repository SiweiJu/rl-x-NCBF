#!/bin/bash
#SBATCH --job-name=go2_obst_scan_sl_on
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

# Go2 velocity tracking with moving obstacles seen only through a 2D range scan (policy and safety critic);
# the value critic also gets the privileged obstacle state. The history encoder convolves over the scan rays
# (see the offline probe in NCBF_EXPERIMENTS.md, Step 3b). The shield is on.
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="unitree_go2"
export RUN_NAME="${RUN_NAME:-go2_obstacles_scan_sl_on}"
export EXP_NAME="${EXP_NAME:-obstacles_scan}"
export USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"
export OBSTACLES_ENABLED="True"
export OBSTACLES_NR="${OBSTACLES_NR:-2}"
export OBSTACLES_SPEED_RANGE="${OBSTACLES_SPEED_RANGE:-(0.5, 3.0)}"
export OBSTACLES_SPEED_CURRICULUM="${OBSTACLES_SPEED_CURRICULUM:-True}"
export OBSTACLES_POLICY_OBSERVATION="${OBSTACLES_POLICY_OBSERVATION:-scan}"
export OBSTACLES_NCBF_OBSERVATION="${OBSTACLES_NCBF_OBSERVATION:-policy}"
export OBSTACLES_VALUE_OBSERVATION="${OBSTACLES_VALUE_OBSERVATION:-state_and_scan}"
export NR_HISTORY_STEPS="${NR_HISTORY_STEPS:-10}"
export HISTORY_ENCODER_TYPE="${HISTORY_ENCODER_TYPE:-scan_conv}"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

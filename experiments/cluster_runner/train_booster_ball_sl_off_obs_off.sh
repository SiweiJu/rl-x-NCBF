#!/bin/bash
#SBATCH --job-name=booster_ball_off_obs_off
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit_booster"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx_boosterball"
export TRAIN_ROBOT="booster_t1"
export RUN_NAME="booster_ball_sl_off_obs_off"
export USE_SAFETY_LAYER="False"
export CURRICULUM_RETURN="40"
export BALL_PLATE_ENABLED="True"
export BALL_PLATE_INCLUDE_OBSERVATIONS="False"

"${SCRIPT_DIR}/run_cluster_experiment.sh"

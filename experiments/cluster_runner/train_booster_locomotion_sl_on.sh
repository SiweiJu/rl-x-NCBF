#!/bin/bash
#SBATCH --job-name=booster_oldcfg_bt_t1_sl_on
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
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx"
export TRAIN_ROBOT="booster_t1"
export RUN_NAME="booster_locomotion_old_config_boostertrain_motor_sl_on"
export EXP_NAME="booster_locomotion_safety_shield"
export USE_SAFETY_LAYER="True"
export USE_BOOSTER_DEFAULTS="False"
export REWARD_TYPE="defaultG1"
export MUJOCO_MODEL_DR_TYPE="none"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

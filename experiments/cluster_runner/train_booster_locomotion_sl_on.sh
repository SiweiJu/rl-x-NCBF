#!/bin/bash
#SBATCH --job-name=booster_substep_lowdr_sl_on
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
export ALGORITHM_NAME="ncbf_ppo.flax_full_jit_booster"
export ENVIRONMENT_NAME="ncbf_mujoco.robot_locomotion.mjx_booster"
export TRAIN_ROBOT="booster_t1"
export RUN_NAME="booster_locomotion_substep_pd_low_dr_sl_on"
export EXP_NAME="locomotion_safety_shield"
export USE_SAFETY_LAYER="True"
export USE_BOOSTER_DEFAULTS="True"
export ACTION_DELAY_TYPE="none"
export SEEN_ROBOT_DR_TYPE="none"
export UNSEEN_ROBOT_DR_TYPE="none"
export MUJOCO_MODEL_DR_TYPE="none"
export OBSERVATION_NOISE_TYPE="none"
export PERTURBATION_TYPE="none"
export JOINT_DROPOUT_TYPE="none"
export CURRICULUM_RETURN="30"
export BALL_PLATE_ENABLED="False"

"${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh"

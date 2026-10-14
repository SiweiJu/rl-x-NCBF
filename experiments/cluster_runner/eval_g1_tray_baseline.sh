#!/bin/bash
#SBATCH --job-name=g1_tray_eval
#SBATCH --output=log/out_and_err_%x_%j.txt
#SBATCH --error=log/out_and_err_%x_%j.txt
#SBATCH --partition=main
#SBATCH -C 'rtx3090|a5000|vram48gb'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=02:00:00

# Validation of a trained G1 tray baseline (experiments/TRAY_BASELINE_PLAN.md, step 0b).
# Usage: CHECKPOINT=/abs/path/to/<run>/models/<file>.model OUTPUT=/abs/path/tray_eval_s42.json sbatch eval_g1_tray_baseline.sh
# The environment flags must match train_g1_tray_baseline.sh / run_cluster_experiment.sh; the algorithm
# config is restored from the checkpoint.
set -euo pipefail
: "${CHECKPOINT:?CHECKPOINT must be set}"
: "${OUTPUT:?OUTPUT must be set}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLUSTER_RUNNER_DIR="${SUBMIT_DIR}"
if [[ ! -f "${CLUSTER_RUNNER_DIR}/run_cluster_experiment.sh" && -f "${CLUSTER_RUNNER_DIR}/cluster_runner/run_cluster_experiment.sh" ]]; then
    CLUSTER_RUNNER_DIR="${CLUSTER_RUNNER_DIR}/cluster_runner"
fi
EXPERIMENTS_DIR="$(cd "${CLUSTER_RUNNER_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-/home/ju/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"
PYTHON_EXE="${PYTHON_EXE:-${CONDA_EXE%/bin/conda}/envs/${CONDA_ENV}/bin/python}"
NR_ENVS="${NR_ENVS:-4096}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${EXPERIMENTS_DIR}"

"${PYTHON_EXE}" ncbf_tools/eval_tray_baseline.py \
    --algorithm.name=ncbf_ppo.flax_full_jit \
    --environment.name=ncbf_mujoco.robot_locomotion.mjx \
    --environment.train_robot=unitree_g1 \
    --environment.seed="${SEED:-1000}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.terrain.type=plane \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds=20 \
    --environment.termination.terminate_on_ball_plate_drop=False \
    --environment.ball_plate.enabled=True \
    --environment.ball_plate.stand_after_drop=True \
    --environment.ball_plate.post_drop_truncation_seconds=3.0 \
    --environment.ball_plate.include_observations=True \
    --algorithm.ncbf.use_safety_layer=False \
    --runner.mode=test \
    --runner.load_model="${CHECKPOINT}" \
    --tray_eval_output="${OUTPUT}"

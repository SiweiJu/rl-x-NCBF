#!/bin/bash
#SBATCH --job-name=go2_obst_eval
#SBATCH --output=log/out_and_err_%x_%j.txt
#SBATCH --error=log/out_and_err_%x_%j.txt
#SBATCH --partition=gpu
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=3:59:59

# Evaluate one trained checkpoint at a range of obstacle speeds (the "how far ahead must the shield look" knob).
# Writes runs/<PROJECT_NAME>/<EXP_NAME>/<RUN_NAME>_speed<v>/eval_summary.json for every speed.
#   CHECKPOINT=/path/to/models/latest.model RUN_NAME=go2_obstacles_sl_on_H25_seed0 sbatch cluster_runner/eval_go2_obstacles_speed_sweep.sh
# Collect the results with: python ncbf_tools/collect_eval_summaries.py runs/<PROJECT_NAME>/<EXP_NAME>
set -euo pipefail

: "${CHECKPOINT:?CHECKPOINT must point to a latest.model file}"
: "${RUN_NAME:?RUN_NAME must be set}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EXPERIMENTS_DIR="${SUBMIT_DIR}"
if [[ -d "${EXPERIMENTS_DIR}/cluster_runner" ]]; then
    :
elif [[ -d "${EXPERIMENTS_DIR}/../cluster_runner" ]]; then
    EXPERIMENTS_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
fi
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-/home/ju/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"
PYTHON_EXE="${PYTHON_EXE:-${CONDA_EXE%/bin/conda}/envs/${CONDA_ENV}/bin/python}"
PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
EXP_NAME="${EXP_NAME:-obstacles_eval}"
NR_ENVS="${NR_ENVS:-1024}"
NR_TEST_EPISODES="${NR_TEST_EPISODES:-5}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
read -r -a SPEEDS <<< "${SPEEDS:-0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${EXPERIMENTS_DIR}"

for speed in "${SPEEDS[@]}"; do
    tag="${speed//./p}"
    echo "Evaluating ${RUN_NAME} at obstacle speed ${speed} m/s"
    "${PYTHON_EXE}" experiment.py \
        --algorithm.name="ncbf_ppo.flax_full_jit" \
        --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
        --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
        --environment.nr_envs="${NR_ENVS}" \
        --environment.train_robot="unitree_go2" \
        --environment.terrain.type="${TERRAIN_TYPE:-plane}" \
        --environment.ncbf_use_policy_observations=True \
        --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
        --environment.obstacles.enabled=True \
        --environment.obstacles.nr_obstacles="${OBSTACLES_NR:-2}" \
        --environment.obstacles.speed_curriculum=False \
        --environment.obstacles.speed_range="(${speed}, ${speed})" \
        --runner.mode=test \
        --runner.nr_test_episodes="${NR_TEST_EPISODES}" \
        --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False \
        --runner.project_name="${PROJECT_NAME}" \
        --runner.exp_name="${EXP_NAME}" \
        --runner.run_name="${RUN_NAME}_speed${tag}" \
        --runner.load_model="${CHECKPOINT}" \
        ${EXTRA_EVAL_ARGS:-}
done

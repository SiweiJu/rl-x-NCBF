#!/bin/bash
#SBATCH --job-name=go2_obst_heldout
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

# Evaluate one checkpoint on the training obstacle distribution and on each held-out preset
# (see obstacle_presets.sh). Writes runs/<PROJECT_NAME>/<EXP_NAME>/<RUN_NAME>_<preset>/eval_summary.json.
#   CHECKPOINT=/path/to/models/latest.model RUN_NAME=go2_obstacles_scan_randomized_seed0 \
#   OBSTACLES_POLICY_OBSERVATION=scan OBSTACLES_VALUE_OBSERVATION=state_and_scan NR_HISTORY_STEPS=10 \
#   sbatch cluster_runner/eval_go2_obstacles_held_out.sh
# The observation settings must match training (as for eval_go2_obstacles_speed_sweep.sh).
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
EXP_NAME="${EXP_NAME:-obstacles_held_out}"
NR_ENVS="${NR_ENVS:-1024}"
NR_TEST_EPISODES="${NR_TEST_EPISODES:-5}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
SPEED_RANGE="${OBSTACLES_SPEED_RANGE:-(0.5, 3.0)}"
read -r -a PRESETS <<< "${PRESETS:-train held_out_shape held_out_motion held_out_size held_out_sensor}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${EXPERIMENTS_DIR}"

for preset in "${PRESETS[@]}"; do
    # Build the obstacle flags in a subshell so one preset's defaults do not leak into the next.
    mapfile -t preset_args < <(
        extra_args=()
        export OBSTACLES_ENABLED="True" OBSTACLES_NR="${OBSTACLES_NR:-2}" OBSTACLES_PRESET="${preset}"
        unset OBSTACLES_SPEED_RANGE OBSTACLES_SPEED_CURRICULUM
        source "${EXPERIMENTS_DIR}/cluster_runner/obstacle_args.sh"
        if [[ -n "${NR_HISTORY_STEPS:-}" ]]; then
            extra_args+=(--environment.nr_history_steps="${NR_HISTORY_STEPS}")
        fi
        printf '%s\n' "${extra_args[@]}"
    )
    echo "Evaluating ${RUN_NAME} on obstacle preset ${preset}"
    "${PYTHON_EXE}" experiment.py \
        --algorithm.name="ncbf_ppo.flax_full_jit" \
        --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
        --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
        --environment.nr_envs="${NR_ENVS}" \
        --environment.train_robot="unitree_go2" \
        --environment.terrain.type="${TERRAIN_TYPE:-plane}" \
        --environment.ncbf_use_policy_observations=True \
        --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
        "${preset_args[@]}" \
        --environment.obstacles.speed_curriculum=False \
        --environment.obstacles.speed_range="${SPEED_RANGE}" \
        --runner.mode=test \
        --runner.nr_test_episodes="${NR_TEST_EPISODES}" \
        --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False \
        --runner.project_name="${PROJECT_NAME}" \
        --runner.exp_name="${EXP_NAME}" \
        --runner.run_name="${RUN_NAME}_${preset}" \
        --runner.load_model="${CHECKPOINT}" \
        ${EXTRA_EVAL_ARGS:-}
done

#!/bin/bash
#SBATCH --job-name=ncbf_counterfactual
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

# E1: counterfactual check of the shield for one checkpoint (see PPO.counterfactual_check). At decision points
# along a shielded rollout it branches with the raw and the shielded action and compares failures over H steps.
# Writes runs/<PROJECT_NAME>/<EXP_NAME>/<RUN_NAME>_cf_<continuation>/counterfactual_summary.json (+ records .npz).
#   CHECKPOINT=/path/to/models/latest.model RUN_NAME=go2_sl_on_seed0 sbatch cluster_runner/eval_counterfactual.sh
# The environment must match training: set TRAIN_ROBOT, BALL_PLATE_*, OBSTACLES_*, NR_HISTORY_STEPS, ... as in
# the training job. The shield's runtime parameters (NCBF_GAMMA_C, ...) are not stored in the checkpoint;
# their defaults below match run_cluster_experiment.sh. Everything else comes from the checkpoint.
# GAMMA_C_VALUES="0.3 0.5 0.7 0.9" repeats the check for each threshold (run name suffix _g<value>), which
# traces how the shield's precision and recall trade off.
# Collect: python ncbf_tools/collect_eval_summaries.py runs/<PROJECT_NAME>/<EXP_NAME> --file counterfactual_summary.json
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
ALGORITHM_NAME="${ALGORITHM_NAME:-ncbf_ppo.flax_full_jit}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-ncbf_mujoco.robot_locomotion.mjx}"
TRAIN_ROBOT="${TRAIN_ROBOT:-unitree_go2}"
TERRAIN_TYPE="${TERRAIN_TYPE:-plane}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
PROJECT_NAME="${PROJECT_NAME:-202605_ncbf}"
EXP_NAME="${EXP_NAME:-counterfactual}"
NR_ENVS="${NR_ENVS:-1024}"
USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
read -r -a CONTINUATIONS <<< "${CONTINUATIONS:-shielded raw}"
read -r -a GAMMA_C_VALUES <<< "${GAMMA_C_VALUES:-${NCBF_GAMMA_C:-0.5}}"

extra_args=(
    --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}"
    --algorithm.ncbf.eta_cbf="${NCBF_ETA_CBF:-1.0}"
    --algorithm.ncbf.lambda_slack="${NCBF_LAMBDA_SLACK:-1000.0}"
    --algorithm.ncbf.max_delta_u="${NCBF_MAX_DELTA_U:-0.5}"
    --algorithm.ncbf.action_clipping="${NCBF_ACTION_CLIPPING:-False}"
    --algorithm.ncbf.safety_layer_projection="${NCBF_SAFETY_LAYER_PROJECTION:-soft_slack}"
    --algorithm.counterfactual.enabled=True
    --algorithm.counterfactual.nr_decision_points="${CF_NR_DECISION_POINTS:-8}"
    --algorithm.counterfactual.interval="${CF_INTERVAL:-50}"
    --algorithm.counterfactual.horizon="${CF_HORIZON:-0}"
    --algorithm.counterfactual.nr_repeats="${CF_NR_REPEATS:-4}"
)
for entry in "NCBF_TIGHTEN_UNCERTAINTY:tighten_uncertainty" "NCBF_SLACK_UNCERTAINTY:slack_uncertainty" "NCBF_GAMMA_C_SCALED_BY_ETA:gamma_c_scaled_by_eta"; do
    env_name="${entry%%:*}"
    if [[ -n "${!env_name:-}" ]]; then
        extra_args+=(--algorithm.ncbf."${entry#*:}"="${!env_name}")
    fi
done
if [[ -n "${REWARD_TYPE:-}" ]]; then
    extra_args+=(--environment.reward.type="${REWARD_TYPE}")
fi
if [[ -n "${USE_BOOSTER_DEFAULTS:-}" ]]; then
    extra_args+=(--environment.use_booster_defaults="${USE_BOOSTER_DEFAULTS}")
fi
if [[ -n "${MUJOCO_MODEL_DR_TYPE:-}" ]]; then
    extra_args+=(--environment.domain_randomization.mujoco_model.type="${MUJOCO_MODEL_DR_TYPE}")
fi
if [[ -n "${NR_HISTORY_STEPS:-}" ]]; then
    extra_args+=(--environment.nr_history_steps="${NR_HISTORY_STEPS}")
fi
if [[ "${BALL_PLATE_ENABLED:-False}" == "True" ]]; then
    : "${BALL_PLATE_INCLUDE_OBSERVATIONS:?BALL_PLATE_INCLUDE_OBSERVATIONS must be set for ball experiments}"
    extra_args+=(
        --environment.termination.terminate_on_ball_plate_drop="${BALL_PLATE_TERMINATE_ON_DROP:-False}"
        --environment.ball_plate.enabled=True
        --environment.ball_plate.stand_after_drop="${BALL_PLATE_STAND_AFTER_DROP:-True}"
        --environment.ball_plate.post_drop_truncation_seconds="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS:-3.0}"
        --environment.ball_plate.include_observations="${BALL_PLATE_INCLUDE_OBSERVATIONS}"
    )
fi
source "${EXPERIMENTS_DIR}/cluster_runner/obstacle_args.sh"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${EXPERIMENTS_DIR}"

for gamma_c in "${GAMMA_C_VALUES[@]}"; do
for continuation in "${CONTINUATIONS[@]}"; do
    suffix="_cf_${continuation}"
    if (( ${#GAMMA_C_VALUES[@]} > 1 )); then
        suffix="${suffix}_g${gamma_c//./p}"
    fi
    echo "Counterfactual check of ${RUN_NAME} (gamma_c ${gamma_c}, continuation: ${continuation})"
    "${PYTHON_EXE}" experiment.py \
        --algorithm.name="${ALGORITHM_NAME}" \
        --environment.name="${ENVIRONMENT_NAME}" \
        --environment.nr_envs="${NR_ENVS}" \
        --environment.train_robot="${TRAIN_ROBOT}" \
        --environment.terrain.type="${TERRAIN_TYPE}" \
        --environment.ncbf_use_policy_observations=True \
        --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
        "${extra_args[@]}" \
        --algorithm.ncbf.gamma_c="${gamma_c}" \
        --algorithm.counterfactual.continuation="${continuation}" \
        --runner.mode=test \
        --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False \
        --runner.project_name="${PROJECT_NAME}" \
        --runner.exp_name="${EXP_NAME}" \
        --runner.run_name="${RUN_NAME}${suffix}" \
        --runner.load_model="${CHECKPOINT}" \
        ${EXTRA_EVAL_ARGS:-}
done
done

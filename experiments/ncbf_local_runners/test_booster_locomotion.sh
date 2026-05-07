#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

MODEL_PATH="/home/siwei/Downloads/latest(54).model"
EXTRA_ARGS=("$@")
if [[ "$#" -gt 0 && "$1" != --* ]]; then
    MODEL_PATH="$1"
    EXTRA_ARGS=("${@:2}")
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: MODEL_PATH=/path/to/model $0 [extra flags]" >&2
    echo "   or: $0 /path/to/model [extra flags]" >&2
    exit 2
fi

CONDA_ENV="${CONDA_ENV:-ncbf-mjx}"
SEED="${SEED:-42}"
NR_TEST_EPISODES="${NR_TEST_EPISODES:-10}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
RENDER="${RENDER:-True}"
ACTION_NOISE_SAMPLING_RATIO="${ACTION_NOISE_SAMPLING_RATIO:-0.2}"
USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-False}"
NCBF_GAMMA_C="${NCBF_GAMMA_C:-0}"
NCBF_OUTPUT_DISTRIBUTION="${NCBF_OUTPUT_DISTRIBUTION:-}"
ROLLOUT_SAVE_NAME="${ROLLOUT_SAVE_NAME:-booster_locomotion_rollout}"
PROJECT_NAME="${PROJECT_NAME:-debug}"
EXP_NAME="${EXP_NAME:-booster_locomotion_test}"
MODEL_TAG="$(basename "${MODEL_PATH}")"
MODEL_TAG="${MODEL_TAG%.*}"
RUN_NAME="${RUN_NAME:-booster_locomotion_${MODEL_TAG}}"
NCBF_OUTPUT_DISTRIBUTION_ARGS=()
if [[ -n "${NCBF_OUTPUT_DISTRIBUTION}" ]]; then
    NCBF_OUTPUT_DISTRIBUTION_ARGS=(--algorithm.ncbf.output_distribution="${NCBF_OUTPUT_DISTRIBUTION}")
fi

cd "${EXPERIMENTS_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python experiment.py \
    --algorithm.name="ncbf_ppo.flax_booster" \
    --algorithm.nr_steps=128 \
    --algorithm.minibatch_size=64 \
    --algorithm.total_timesteps=700000000 \
    --algorithm.hidden_layers="(256, 128)" \
    --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
    --algorithm.ncbf.gamma_c="${NCBF_GAMMA_C}" \
    --algorithm.ncbf.eta_cbf=0.5 \
    --algorithm.ncbf.lambda_slack=10 \
    --algorithm.ncbf.H=25 \
    --algorithm.ncbf.action_clipping=False \
    "${NCBF_OUTPUT_DISTRIBUTION_ARGS[@]}" \
    --algorithm.next_step_predictor.history_encoder_hidden_size=64 \
    --algorithm.rollout_save_name="${ROLLOUT_SAVE_NAME}" \
    --algorithm.action_noise_sampling_ratio="${ACTION_NOISE_SAMPLING_RATIO}" \
    --environment.terrain.type="plane" \
    --environment.name="ncbf_mujoco.robot_locomotion.mujoco_booster" \
    --environment.nr_envs=1 \
    --environment.seed="${SEED}" \
    --environment.render="${RENDER}" \
    --environment.command.type="booster" \
    --environment.add_goal_arrow=True \
    --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
    --environment.env_curriculum_level_success_episode_return=30 \
    --environment.ncbf_use_policy_observations=True \
    --environment.train_robot="booster_t1" \
    --runner.mode="test" \
    --runner.nr_test_episodes="${NR_TEST_EPISODES}" \
    --runner.load_model="${MODEL_PATH}" \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=False \
    --runner.save_model=False \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME}" \
    "${EXTRA_ARGS[@]}"

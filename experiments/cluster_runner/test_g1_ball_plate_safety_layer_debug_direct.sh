#!/bin/bash
#SBATCH --job-name=g1_ball_sl_direct
#SBATCH --output=log/out_and_err_%x_%A_%a.txt
#SBATCH --error=log/out_and_err_%x_%A_%a.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=15G
#SBATCH --time=71:59:59
#SBATCH --array=0-5

set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

sanitize() {
    local value="$1"
    value="${value//-/m}"
    value="${value//./p}"
    echo "${value}"
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${TASK_ID}" in
    0)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-False}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.1}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.5}"
        DEFAULT_RUN_NAME="g1_ball_sl_off"
        ;;
    1)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.1}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.5}"
        DEFAULT_RUN_NAME="g1_ball_sl_current"
        ;;
    2)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-True}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.1}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.5}"
        DEFAULT_RUN_NAME="g1_ball_sl_action_clip"
        ;;
    3)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-1.0}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.1}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.5}"
        DEFAULT_RUN_NAME="g1_ball_sl_cap_$(sanitize "${NCBF_MAX_DELTA_U}")"
        ;;
    4)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.2}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.5}"
        DEFAULT_RUN_NAME="g1_ball_sl_gamma_$(sanitize "${NCBF_GAMMA_C}")"
        ;;
    5)
        USE_SAFETY_LAYER="${USE_SAFETY_LAYER:-True}"
        NCBF_ACTION_CLIPPING="${NCBF_ACTION_CLIPPING:-False}"
        NCBF_MAX_DELTA_U="${NCBF_MAX_DELTA_U:-0.5}"
        NCBF_GAMMA_C="${NCBF_GAMMA_C:-0.1}"
        NCBF_ETA_CBF="${NCBF_ETA_CBF:-0.25}"
        DEFAULT_RUN_NAME="g1_ball_sl_eta_$(sanitize "${NCBF_ETA_CBF}")"
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside the configured sweep size 6." >&2
        exit 2
        ;;
esac

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-SLDBG_}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

echo "Task ID: ${TASK_ID}"
echo "Use safety layer: ${USE_SAFETY_LAYER}"
echo "Action clipping: ${NCBF_ACTION_CLIPPING}"
echo "Max delta u: ${NCBF_MAX_DELTA_U}"
echo "Gamma c: ${NCBF_GAMMA_C}"
echo "Eta CBF: ${NCBF_ETA_CBF}"
echo "Run name: ${RUN_NAME_PREFIX}${RUN_NAME}"
echo "---"

cd "${EXPERIMENTS_DIR}"

python experiment.py \
    --algorithm.name="ncbf_ppo.flax_full_jit" \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS:-50000000}" \
    --algorithm.minibatch_size="${MINIBATCH_SIZE:-4096}" \
    --algorithm.learning_rate="${LEARNING_RATE:-4e-4}" \
    --algorithm.adaptive_lr="${ADAPTIVE_LR:-True}" \
    --algorithm.nr_steps="${NR_STEPS:-128}" \
    --algorithm.nr_epochs="${NR_EPOCHS:-4}" \
    --algorithm.use_decoder_output_for_policy="${USE_DECODER_OUTPUT_FOR_POLICY:-True}" \
    --algorithm.ncbf.use_safety_layer="${USE_SAFETY_LAYER}" \
    --algorithm.ncbf.output_distribution="${NCBF_OUTPUT_DISTRIBUTION:-deterministic}" \
    --algorithm.ncbf.gamma_c="${NCBF_GAMMA_C}" \
    --algorithm.ncbf.eta_cbf="${NCBF_ETA_CBF}" \
    --algorithm.ncbf.lambda_slack="${NCBF_LAMBDA_SLACK:-1000.0}" \
    --algorithm.ncbf.max_delta_u="${NCBF_MAX_DELTA_U}" \
    --algorithm.ncbf.H="${NCBF_H:-25}" \
    --algorithm.ncbf_buffer.neg_buffer_size="${NCBF_NEG_BUFFER_SIZE:-1}" \
    --algorithm.ncbf_buffer.neg_sampling_ratio="${NCBF_NEG_SAMPLING_RATIO:-0.2}" \
    --algorithm.ncbf.action_clipping="${NCBF_ACTION_CLIPPING}" \
    --algorithm.ncbf.lr="${NCBF_LR:-3e-4}" \
    --algorithm.next_step_predictor.nr_minibatches="${NEXT_STEP_PREDICTOR_NR_MINIBATCHES:-50}" \
    --algorithm.next_step_predictor.lr="${NEXT_STEP_PREDICTOR_LR:-1e-5}" \
    --algorithm.next_step_predictor.aux_loss_coef="${NEXT_STEP_PREDICTOR_AUX_LOSS_COEF:-1}" \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
    --environment.seed="${SEED:-42}" \
    --environment.nr_envs="${NR_ENVS:-1024}" \
    --environment.terrain.type="${TERRAIN_TYPE:-plane}" \
    --environment.ncbf_use_policy_observations=True \
    --environment.train_robot="unitree_g1" \
    --environment.episode_length_in_seconds="${EPISODE_SECONDS:-20}" \
    --environment.env_curriculum_level_success_episode_return="${CURRICULUM_RETURN:-20}" \
    --environment.reward.type="${REWARD_TYPE:-G1ball}" \
    --environment.termination.terminate_on_ball_plate_drop="${BALL_PLATE_TERMINATE_ON_DROP:-False}" \
    --environment.ball_plate.enabled=True \
    --environment.ball_plate.stand_after_drop="${BALL_PLATE_STAND_AFTER_DROP:-True}" \
    --environment.ball_plate.post_drop_truncation_seconds="${BALL_PLATE_POST_DROP_TRUNCATION_SECONDS:-3.0}" \
    --environment.ball_plate.include_observations="${BALL_PLATE_INCLUDE_OBSERVATIONS:-False}" \
    --runner.mode="train" \
    --runner.track_console="${TRACK_CONSOLE:-False}" \
    --runner.track_tb="${TRACK_TB:-True}" \
    --runner.track_wandb="${TRACK_WANDB:-True}" \
    --runner.save_model="${SAVE_MODEL:-True}" \
    --runner.wandb_entity="${WANDB_ENTITY:-catherineju-rwth-aachen-university}" \
    --runner.project_name="${PROJECT_NAME:-202605_ncbf}" \
    --runner.exp_name="${EXP_NAME:-sl_check}" \
    --runner.run_name="${RUN_NAME_PREFIX}${RUN_NAME}"

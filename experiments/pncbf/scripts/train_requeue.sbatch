#!/bin/bash
# Requeue-safe training job for ncbf_ppo.flax_full_jit (normal QOS or --qos=scavenger).
#
# A preempted/requeued job reruns THIS script from the top with the same SLURM_JOB_ID. It is safe because:
#   - the run dir and the wandb id are fixed by RUN_NAME (no timestamps), runner.resume=True is always passed:
#     the first attempt starts fresh, later attempts restore <run dir>/resume/LATEST and continue;
#   - checkpoints are atomic (new step dir + atomic LATEST pointer), every SAVE_ITERS iterations and every
#     SAVE_MINUTES minutes of wall time;
#   - --open-mode=append keeps the log of earlier attempts; nothing is deleted on restart;
#   - SLURM_RESTART_COUNT is printed here and written to the wandb config (run_info, resume_history).
#
# Submit (from anywhere; REPO_DIR must be the checkout to run, PYTHONPATH is set to it below):
#   export REPO_DIR=/home/ju/repo/pncbf/core RUN_NAME=<unique name> WANDB_TAGS="phase/P0,..." EXTRA_ARGS="--algorithm...."
#   sbatch --qos=normal --time=01:00:00 --job-name=<name> --output=<logdir>/%x_%j.out --export=ALL \
#     experiments/pncbf/scripts/train_requeue.sbatch
# (pass values through the environment: --export=VAR=value splits at commas). Example: smoke_go2_p0.sh.
# For scavenger: --qos=scavenger (optionally --signal=B:USR1@110 is not needed: the job resumes from the last
# checkpoint; work since that checkpoint is lost).
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

: "${REPO_DIR:?set REPO_DIR to the checkout to run}"
: "${RUN_NAME:?set RUN_NAME (fixed, unique per run)}"
PROJECT_NAME="${PROJECT_NAME:-pncbf-2026}"
EXP_NAME="${EXP_NAME:-p0}"
WANDB_ENTITY="${WANDB_ENTITY:-catherineju-rwth-aachen-university}"
WANDB_TAGS="${WANDB_TAGS:-phase/P0}"
TRACK_WANDB="${TRACK_WANDB:-True}"
SAVE_ITERS="${SAVE_ITERS:-0}"
SAVE_MINUTES="${SAVE_MINUTES:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source "${CONDA_ROOT:-/home/ju/miniconda3}/bin/activate" "${CONDA_ENV:-rlx-ncbf}"
# rlx-ncbf has rl_x installed editable from another tree: force this checkout.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${REPO_DIR}/experiments"

echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname) SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} qos=${SLURM_JOB_QOS:-?}"
echo "git HEAD: $(git -C "${REPO_DIR}" rev-parse HEAD) ($(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD))"
python -c "import rl_x, jax; print('rl_x:', rl_x.__file__); print('jax devices:', jax.devices())"

# shellcheck disable=SC2086
python experiment.py \
    --algorithm.name=ncbf_ppo.flax_full_jit \
    --environment.name=ncbf_mujoco.robot_locomotion.mjx \
    --runner.mode=train \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb="${TRACK_WANDB}" \
    --runner.save_model=True \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME}" \
    --runner.wandb_tags="${WANDB_TAGS}" \
    --runner.resume=True \
    --runner.save_interval_iterations="${SAVE_ITERS}" \
    --runner.save_interval_minutes="${SAVE_MINUTES}" \
    ${EXTRA_ARGS}
echo "=== $(date -Is) finished (exit $?)"

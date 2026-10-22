#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:45:00
#SBATCH --array=0-59%2
# G0 part 1: ORIGINAL paper checkpoints (wandb latest.model; PPO, NCBF H=1, ours H=25, 5 seeds each)
# x 4 test settings, reconstructed eval. One task = one (model, setting); 60 tasks, max 2 concurrent.
#   sbatch --qos=normal --job-name=recover_g0 --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%A_%a.txt \
#          --export=ALL,EP_SECONDS=<calibrated> g0_sweep.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models
MODELS=()
for R in no_safety NCBF ours; do for f in $(ls $M/$R/*.model | sort); do MODELS+=("$R:$f"); done; done
SETTINGS=(action_noise_2 action_noise_hfield perturbation perturbation_hfield)
I=${SLURM_ARRAY_TASK_ID}
ENTRY=${MODELS[$((I / 4))]}; SETTING=${SETTINGS[$((I % 4))]}
RUN_ROOT=${ENTRY%%:*}; MODEL=${ENTRY#*:}
echo "task $I: $RUN_ROOT $MODEL $SETTING EP_SECONDS=${EP_SECONDS:-10}"
OUT=${OUT:-/home/ju/repo/pncbf/paper_eval_g0/ep${EP_SECONDS:-10}} GAMMAS=${GAMMAS:-0.405} SEEDS=${SEEDS:-1} \
  bash paper_eval_sweep.sh "$MODEL" "$RUN_ROOT" "$SETTING"

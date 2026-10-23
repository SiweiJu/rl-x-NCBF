#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:50:00
#SBATCH --array=0-14%2
# G0 part 1: ORIGINAL paper checkpoints (wandb latest.model; PPO, NCBF H=1, ours H=25, 5 seeds each)
# x 4 test settings, reconstructed eval. One task = one model, its 4 settings run as 2 parallel processes
# (the eval is CPU-bound, 2 CPUs per task); existing pkls are skipped. 15 tasks, max 2 concurrent.
#   sbatch --qos=normal --job-name=recover_g0 --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%A_%a.txt \
#          --export=ALL,EP_SECONDS=<calibrated> g0_sweep.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models
MODELS=()
for R in no_safety NCBF ours; do for f in $(ls $M/$R/*.model | sort); do MODELS+=("$R:$f"); done; done
I=${SLURM_ARRAY_TASK_ID}
ENTRY=${MODELS[$I]}; RUN_ROOT=${ENTRY%%:*}; MODEL=${ENTRY#*:}
echo "task $I: $RUN_ROOT $MODEL EP_SECONDS=${EP_SECONDS:-10}"
export OUT=${OUT:-/home/ju/repo/pncbf/paper_eval_g0/ep${EP_SECONDS:-10}} GAMMAS=${GAMMAS:-0.405} SEEDS=${SEEDS:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # two JAX processes share the GPU
( bash paper_eval_sweep.sh "$MODEL" "$RUN_ROOT" action_noise_2 && bash paper_eval_sweep.sh "$MODEL" "$RUN_ROOT" perturbation ) & P1=$!
( bash paper_eval_sweep.sh "$MODEL" "$RUN_ROOT" action_noise_hfield && bash paper_eval_sweep.sh "$MODEL" "$RUN_ROOT" perturbation_hfield ) & P2=$!
wait $P1; wait $P2

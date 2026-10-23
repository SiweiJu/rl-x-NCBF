#!/bin/bash
# sbatch wrapper for run_baseline.sh. Pass METHOD / COST_LIMIT / SEED via --export, the job name, log path
# and --time on the sbatch command line (see submit_go2_sanity.sh). Submit from experiments/pncbf/baselines.
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
set -euo pipefail
export BASELINES_DIR="${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${BASELINES_DIR}/run_baseline.sh" "$@"

#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

# G0 part 2: one array task = one checkpoint x terrain (modes B and A, all sweep points), see g0_part2.py.
# Scavenger-safe: a requeue reruns the task and every finished point (eval_summary.json) is skipped.
#   cd /home/ju/repo/pncbf/eval-g0/experiments/pncbf/eval
#   python3 g0_part2.py --list_jobs [--groups T05]          # array indices; refuses checkpoints still being written
#   sbatch --qos=scavenger --array=<indices>%<max> --job-name=g0p2 --output=/home/ju/repo/pncbf/g0_part2/log/%x_%A_%a.txt g0_part2.sbatch
# Extra driver flags in G0_ARGS (e.g. G0_ARGS="--groups T05 --force").
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f "${HERE}/g0_part2.py" ]] || HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} (${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} driver commit $(git -C "${HERE}" rev-parse --short HEAD)"
read -r -a g0_args <<< "${G0_ARGS:-}"
start=$(date +%s)
/home/ju/miniconda3/envs/rlx-ncbf/bin/python "${HERE}/g0_part2.py" --task_index "${SLURM_ARRAY_TASK_ID:?array job only}" "${g0_args[@]}"
echo "=== task wall time $(( $(date +%s) - start )) s"

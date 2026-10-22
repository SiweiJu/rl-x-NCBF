#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1:00:00

# Perturbation-sweep evaluation (T04), one array task = one checkpoint x terrain x deployment mode
# (all sampling probabilities and eval seeds of both perturbations run inside the task).
#
#   cd <repo>/experiments
#   python pncbf/eval/run_eval_sweep.py --checkpoints_file CKPTS.json --out_dir OUT --list_jobs   # -> N jobs
#   sbatch --array=0-$((N-1)) --job-name=pncbf_eval --output=log/pncbf_eval_%A_%a.txt \
#       --export=ALL,CHECKPOINTS_FILE=CKPTS.json,OUT_DIR=OUT pncbf/eval/eval_sweep.sbatch
#
# Without --array, all jobs run sequentially in one task. Extra driver flags go in SWEEP_ARGS, e.g.
#   SWEEP_ARGS="--terrains flat --sampling_probs 0.0 0.01 --modes A B".
# Time: ~TIME_PER_POINT x (#probabilities x #perturbations x #eval seeds) per task, see reports/T04.md.
set -euo pipefail

: "${CHECKPOINTS_FILE:?CHECKPOINTS_FILE must point to the checkpoint list (JSON)}"
: "${OUT_DIR:?OUT_DIR must be set}"

EXPERIMENTS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f "${EXPERIMENTS_DIR}/experiment.py" ]]; then
    EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
CONDA_EXE="${CONDA_EXE:-/home/ju/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"
PYTHON_EXE="${PYTHON_EXE:-${CONDA_EXE%/bin/conda}/envs/${CONDA_ENV}/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

task_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_args=(--task_index "${SLURM_ARRAY_TASK_ID}")
fi
read -r -a sweep_args <<< "${SWEEP_ARGS:-}"

cd "${EXPERIMENTS_DIR}"
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  commit $(git rev-parse --short HEAD)"
start=$(date +%s)
"${PYTHON_EXE}" pncbf/eval/run_eval_sweep.py \
    --checkpoints_file "${CHECKPOINTS_FILE}" \
    --out_dir "${OUT_DIR}" \
    --python "${PYTHON_EXE}" \
    "${task_args[@]}" "${sweep_args[@]}"
echo "total wall time $(( $(date +%s) - start )) s"

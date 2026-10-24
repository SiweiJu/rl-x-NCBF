#!/bin/bash
# T08 evaluation: the 9 T08 checkpoints (t08_checkpoints.json) with the T04 sweep, mode A, flat + hfield,
# protocol_default.json grids. 18 array tasks, at most 2 running at a time. Eval jobs are restartable (a requeue
# reruns the task and overwrites its eval_summary.json files), so they may run on the scavenger QOS.
#   QOS=scavenger|normal (default scavenger)  OUT_DIR (default <repo>/experiments/runs/pncbf_eval/T08)  DRY_RUN=1 prints only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # experiments/
mkdir -p log
QOS="${QOS:-scavenger}"
OUT_DIR="${OUT_DIR:-$(pwd)/runs/pncbf_eval/T08}"
CKPTS="$(pwd)/pncbf/baselines/t08_checkpoints.json"
PY="${PYTHON_EXE:-/home/ju/miniconda3/envs/rlx-ncbf/bin/python}"
n=$(PYTHONPATH="$(cd .. && pwd)" "${PY}" pncbf/eval/run_eval_sweep.py --checkpoints_file "${CKPTS}" --out_dir "${OUT_DIR}" --modes A --list_jobs | tail -1 | sed -E 's/^([0-9]+) jobs.*/\1/')
cmd=(sbatch --qos="${QOS}" --requeue --time=0:45:00 --array="0-$((n-1))%2" --job-name=t08_eval --output="log/t08_eval_%A_%a.txt"
     --export=ALL,CHECKPOINTS_FILE="${CKPTS}",OUT_DIR="${OUT_DIR}",SWEEP_ARGS="--modes A" pncbf/eval/eval_sweep.sbatch)
echo "${cmd[*]}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi

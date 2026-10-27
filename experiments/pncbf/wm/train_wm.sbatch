#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# T13 offline world-model training (P1). Requeue-safe: fixed RUN_DIR + wandb id, atomic checkpoint every SAVE_MINUTES,
# the script resumes from RUN_DIR/ckpt_latest.msgpack. Runs from the fixed tree /home/ju/repo/pncbf/wm-model-<commit7>
# (TREE_PREFIX=wm-model COMMIT=<sha> experiments/pncbf/T06/setup_tree.sh). Data is loaded into host RAM (8G/CPU),
# then copied to the GPU.
#
#   unset PYTHONPATH; COMMIT=<sha> RUN=T13_base DATA='/home/ju/repo/pncbf/p1_data/<set>/*.npz' \
#     SCHEMA=/home/ju/repo/pncbf/p1_data/<set>/schema.json ARGS="--updates 200000" \
#     sbatch --job-name=T13_base --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/train_wm.sbatch
# DATA=synthetic runs on the toy data (smoke).
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
# Sweep: SWEEP_FILE with one "RUN_NAME extra args..." line per array task (sbatch --array=0-<n-1>%3).
if [[ -n "${SWEEP_FILE:-}" ]]; then
  LINE=$(grep -v '^#' "$SWEEP_FILE" | grep -v '^\s*$' | sed -n "$(( ${SLURM_ARRAY_TASK_ID:?} + 1 ))p")
  RUN=$(echo "$LINE" | awk '{print $1}')
  ARGS="${ARGS:-} $(echo "$LINE" | sed 's/^[^ ]*//')"
fi
RUN=${RUN:?set RUN (run name, also the run dir name)}
DATA=${DATA:?set DATA (glob or synthetic)}
RUN_DIR=/home/ju/repo/pncbf/p1_wm_runs/$RUN

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} run=$RUN commit=$COMMIT"
mkdir -p "$RUN_DIR"
SCHEMA_ARGS=(); [[ -n "${SCHEMA:-}" ]] && SCHEMA_ARGS=(--schema "$SCHEMA")
read -r -a EXTRA <<< "${ARGS:-}"
python experiments/pncbf/wm/train_wm_offline.py --run_dir "$RUN_DIR" --run_name "$RUN" --data "$DATA" "${SCHEMA_ARGS[@]}" \
  --save_minutes "${SAVE_MINUTES:-15}" --wandb "${WANDB:-1}" --wandb_tags "phase/P1,exp/T13,commit/${COMMIT:0:7}${SMOKE:+,smoke}" \
  "${EXTRA[@]}"
echo "=== $(date -Is) finished; peak RSS $(sacct -j ${SLURM_JOB_ID:-0} --format=MaxRSS -n 2>/dev/null | sort | tail -1)"

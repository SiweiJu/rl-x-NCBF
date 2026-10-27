#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00
# T13 k-step evaluation of a trained world model (eval_wm.py) on the held-out shards (or SPLIT=train for dev checks).
#   unset PYTHONPATH; COMMIT=<sha> WM=<run>/wm_final.msgpack DATA=<shard root> OUT=<run>/eval_k.json \
#     sbatch --job-name=T13_eval --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/eval_wm.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT"
if [[ "${TOOL:-eval}" == diag ]]; then   # error breakdown by row category (diag_wm.py)
  python experiments/pncbf/wm/diag_wm.py --wm "${WM:?}" --data "${DATA:?}" --out "${OUT:?}" --split "${SPLIT:-heldout}" \
    --max_envs_per_shard "${MAX_ENVS:-0}" --max_windows "${MAX_WINDOWS:-50000}" --ks "${KS:-1,2,8,25}"
else
  # WM may be a comma list; MATMUL a comma list (float32,bfloat16); OUT only for a single model/dtype
  python experiments/pncbf/wm/eval_wm.py --wm "${WM:?}" --data "${DATA:?}" ${OUT:+--out $OUT} --split "${SPLIT:-heldout}" \
    --max_envs_per_shard "${MAX_ENVS:-0}" --max_windows "${MAX_WINDOWS:-50000}" ${MATMUL:+--matmul_dtype $MATMUL}
fi
echo "=== $(date -Is) finished"

#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1:00:00
# T13 speed matrix (world model only, random params, T06b-sized random policy): width / depth / ensemble,
# closed vs open loop, float32 vs bfloat16, H in {10, 16, 25}, N in {1024, 4096}. One JSON per config in OUT_DIR.
#   unset PYTHONPATH; COMMIT=<sha> OUT_DIR=/home/ju/repo/pncbf/p1_wm_runs/bench/matrix_<x> \
#     sbatch --job-name=T13_benchmat --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/bench_wm_matrix.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT"
mkdir -p "$OUT_DIR"
run() {  # name, extra args
  local name=$1; shift
  python experiments/pncbf/wm/bench_wm_speed.py --bench_no_env 1 --bench_wm_envs 1024,4096 --bench_H 10,16,25 \
    --bench_reps 20 --bench_out "$OUT_DIR/$name.json" "$@" 2>&1 | grep -E "^N=|Error|Traceback" | sed "s/^/$name /"
}
run w512d3_closed_f32
run w512d3_open_f32 --bench_open_loop 1
run w512d3_closed_bf16 --bench_dtype bfloat16
run w256d3_closed_f32 --bench_width 256
run w1024d3_closed_f32 --bench_width 1024
run w512d4_closed_f32 --bench_depth 4
run w1024d4_closed_f32 --bench_width 1024 --bench_depth 4
run w512d3_ens5_closed_f32 --bench_ensemble 5
echo "=== $(date -Is) finished"

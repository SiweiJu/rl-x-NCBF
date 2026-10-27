#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00
# T13 gate check (gate_wm.py): V of the trained WM on the shards of ONE checkpoint, top-k gate coverage.
# Pairs: CKPT_SHARDS="<ckpt path>|<shard glob> <ckpt path>|<shard glob> ..." (one runner process per pair).
#   unset PYTHONPATH; COMMIT=<sha> WM=<run>/wm_final.msgpack OUT_DIR=<dir> CKPT_SHARDS="..." \
#     sbatch --job-name=T13_gate --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/gate_wm.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
WM=${WM:?set WM}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
CKPT_SHARDS=${CKPT_SHARDS:?set CKPT_SHARDS}
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
read -r -a EXTRA <<< "${GATE_ARGS:-}"
i=0
for pair in $CKPT_SHARDS; do
  CKPT=${pair%%|*}; SHARDS=${pair#*|}; i=$((i+1))
  cp "$CKPT" "$TMPDIR/ckpt_$i.model"   # PPO.load unpacks next to the file
  (cd "$TREE/experiments" && python pncbf/wm/gate_wm.py --gate_wm "$WM" --gate_shards "$SHARDS" \
    --gate_out "$OUT_DIR/gate_$i.json" "${EXTRA[@]}" \
    --algorithm.name=ncbf_ppo.flax_full_jit --environment.name=ncbf_mujoco.robot_locomotion.mjx --runner.mode=test \
    --runner.load_model="$TMPDIR/ckpt_$i.model" --runner.track_console=False --runner.track_tb=False \
    --runner.track_wandb=False --runner.save_model=False --runner.project_name=pncbf_gate --runner.exp_name=T13 \
    --runner.run_name=gate_$i --environment.nr_envs=1024 --environment.terrain.type=plane \
    --environment.ncbf_use_policy_observations=True --environment.env_curriculum_level_success_episode_return=8 \
    --environment.command.max_velocity_per_m_factor=2.0 --environment.command.clip_max_velocity=2.0 \
    "--environment.command.velocity_ratio=(1.0,1.0,1.0)" --environment.termination.height_curriculum_scaled=True \
    --environment.termination.body_tilt_threshold=1000000.0) 2>&1 | grep -vE "GLFW|warnings.warn"
done
echo "=== $(date -Is) finished"

#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
# T14: per held-out checkpoint, t14_collect.py (WM / persistence / critic predictions + oracle / actual labels on the
# oracle decision points of its 5 shards) and gate_wm.py (|grad_u V| and gate coverage); then t14_analyze.py and
# gate_summary.py over all of them. The checkpoint of each shard set comes from its meta.json.
#   unset PYTHONPATH; COMMIT=<sha> WM=<run>/wm_final.msgpack OUT_DIR=<dir> \
#     sbatch --job-name=T14 --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/t14.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
WM=${WM:?set WM}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
DATA=${DATA:-/home/ju/repo/pncbf/p1_data/v1/test}
T_MAIN=${T_MAIN:-1}
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT wm=$WM"
mkdir -p "$OUT_DIR"
RUNNER=(--algorithm.name=ncbf_ppo.flax_full_jit --environment.name=ncbf_mujoco.robot_locomotion.mjx --runner.mode=test
  --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False
  --runner.project_name=pncbf_t14 --runner.exp_name=T14 --environment.nr_envs=1024 --environment.terrain.type=plane
  --environment.ncbf_use_policy_observations=True --environment.env_curriculum_level_success_episode_return=8
  --environment.command.max_velocity_per_m_factor=2.0 --environment.command.clip_max_velocity=2.0
  "--environment.command.velocity_ratio=(1.0,1.0,1.0)" --environment.termination.height_curriculum_scaled=True
  --environment.termination.body_tilt_threshold=1000000.0)
for CK_DIR in "$DATA"/*/; do
  LABEL=$(basename "$CK_DIR")
  [[ -n "${ONLY:-}" && "$LABEL" != *"$ONLY"* ]] && continue
  CKPT=$(python -c "import json,glob; print(json.load(open(sorted(glob.glob('$CK_DIR/*/meta.json'))[0]))['checkpoint'])")
  cp "$CKPT" "$TMPDIR/$LABEL.model"
  echo "=== $(date -Is) $LABEL ($CKPT)"
  (cd "$TREE/experiments" && python pncbf/wm/t14_collect.py --gate_wm "$WM" --gate_shards "$CK_DIR" \
     --gate_out "$OUT_DIR/t14_$LABEL.npz" --runner.load_model="$TMPDIR/$LABEL.model" --runner.run_name=t14_$LABEL "${RUNNER[@]}") \
     2>&1 | grep -vE "GLFW|warnings.warn"
  cp "$CKPT" "$TMPDIR/$LABEL.model"
  (cd "$TREE/experiments" && python pncbf/wm/gate_wm.py --gate_wm "$WM" --gate_shards "$CK_DIR" --gate_label "$LABEL" \
     --gate_out "$OUT_DIR/gate_$LABEL.json" --gate_grad_states 64 --runner.load_model="$TMPDIR/$LABEL.model" \
     --runner.run_name=gate_$LABEL "${RUNNER[@]}") 2>&1 | grep -vE "GLFW|warnings.warn" | grep -E "T_l=|Error|Traceback" | cut -c1-300
done
python experiments/pncbf/wm/t14_analyze.py "$OUT_DIR"/t14_*.npz --T_main "$T_MAIN" --B "${BOOT:-200}" --out "$OUT_DIR/t14_summary.json" | tee "$OUT_DIR/t14_summary.md"
python experiments/pncbf/wm/gate_summary.py "$OUT_DIR"/gate_*.npz --out "$OUT_DIR/gate_summary.json" | tee "$OUT_DIR/gate_summary.md"
echo "=== $(date -Is) finished"

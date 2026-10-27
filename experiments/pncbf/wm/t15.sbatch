#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0:45:00
# T15: WM shield inside eval rollouts of a fixed T06b checkpoint (t15_shield_check.py).
#   unset PYTHONPATH; COMMIT=<sha> WM=<run>/wm_final.msgpack CKPT=<model> OUT=<json> PROBS=0.4 \
#     sbatch --job-name=T15 --output=/home/ju/repo/pncbf/p2_runs/T15/logs/%x_%j.txt --export=ALL experiments/pncbf/wm/t15.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/d-shield-${COMMIT:0:7}
WM=${WM:?set WM}; CKPT=${CKPT:?set CKPT}; OUT=${OUT:?set OUT}
PROBS=${PROBS:-0.4,0.8}
MODES=${MODES:-none,gated_f32,full_f32,gated_bf16,full_bf16}
STEPS=${STEPS:-1000}
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT wm=$WM ckpt=$CKPT"
mkdir -p "$(dirname "$OUT")"
cp "$CKPT" "$TMPDIR/ckpt.model"
cd "$TREE/experiments"
python pncbf/wm/t15_shield_check.py --gate_wm "$WM" --gate_out "$OUT" --gate_modes "$MODES" --gate_probs "$PROBS" \
  --gate_steps "$STEPS" --gate_chunk 100 ${EXTRA:-} \
  --runner.load_model="$TMPDIR/ckpt.model" --runner.run_name=t15_${SLURM_JOB_ID:-x} \
  --algorithm.name=ncbf_ppo.flax_full_jit --environment.name=ncbf_mujoco.robot_locomotion.mjx --runner.mode=test \
  --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False \
  --runner.project_name=pncbf_t15 --runner.exp_name=T15 --environment.nr_envs=1024 --environment.terrain.type=plane \
  --environment.ncbf_use_policy_observations=True --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 --environment.command.clip_max_velocity=2.0 \
  "--environment.command.velocity_ratio=(1.0,1.0,1.0)" --environment.termination.height_curriculum_scaled=True \
  --environment.termination.body_tilt_threshold=1000000.0 --environment.record_wm_state=True \
  2>&1 | grep --line-buffered -vE "GLFW|warnings.warn"
echo "=== $(date -Is) finished"

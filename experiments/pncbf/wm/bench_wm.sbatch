#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1:00:00
# T13 speed: WM rollout (N x H + one VJP) vs one PPO env step of T06b (ii_h25 seed 42 checkpoint), env at NENV envs.
#   unset PYTHONPATH; COMMIT=<sha> NENV=1024 OUT=/home/ju/repo/pncbf/p1_wm_runs/bench/bench_1024.json \
#     sbatch --job-name=T13_bench --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/bench_wm.sbatch
# Extra bench flags in BENCH_ARGS (e.g. "--bench_width 1024 --bench_depth 4 --bench_schema S.json --bench_wm_file F").
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
NENV=${NENV:-1024}
OUT=${OUT:?set OUT}
CKPT=${CKPT:-/home/ju/repo/pncbf/v2-62d1616/experiments/runs/pncbf-2026/T06-fixed-ii_h25/T06_fixed_ii_h25_seed_42/models/latest.model}
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/home/ju/repo/pncbf/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$COMMIT"
mkdir -p "$(dirname "$OUT")"
# PPO.load unpacks the checkpoint zip next to it -> work on a copy in TMPDIR
cp "$CKPT" "$TMPDIR/latest.model"
read -r -a EXTRA <<< "${BENCH_ARGS:-}"
cd "$TREE/experiments"
python pncbf/wm/bench_wm_speed.py --bench_out "$OUT" --bench_wm_envs "${WM_ENVS:-1024,4096}" "${EXTRA[@]}" \
  --algorithm.name=ncbf_ppo.flax_full_jit --environment.name=ncbf_mujoco.robot_locomotion.mjx --runner.mode=test \
  --runner.load_model="$TMPDIR/latest.model" --runner.track_console=False --runner.track_tb=False \
  --runner.track_wandb=False --runner.save_model=False --runner.project_name=pncbf_bench --runner.exp_name=T13 \
  --runner.run_name=bench_$NENV --environment.nr_envs=$NENV --environment.terrain.type=plane \
  --environment.ncbf_use_policy_observations=True --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 --environment.command.clip_max_velocity=2.0 \
  "--environment.command.velocity_ratio=(1.0,1.0,1.0)" --environment.termination.height_curriculum_scaled=True \
  --environment.termination.body_tilt_threshold=1000000.0 --algorithm.ncbf.use_safety_layer=True \
  --algorithm.ncbf.gamma_c=0.6 --algorithm.ncbf.eta_cbf=0.5 --algorithm.ncbf.safety_layer_projection=soft_slack \
  --algorithm.ncbf.lambda_slack=1000 --algorithm.ncbf.max_delta_u=0 --algorithm.ncbf.gamma_c_scaled_by_eta=True
echo "=== $(date -Is) finished"

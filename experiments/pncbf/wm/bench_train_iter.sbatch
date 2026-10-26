#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
# T13 speed, the GB ratio: full training-iteration time (128-step rollout incl. shield + PPO/critic update) of the
# T06b config (1024 envs, flat) for: T06b ppo, T06b ii_h25 (critic shield), and PPO + WM shield (full VJP, gated VJP
# 25 % / 50 %, bf16 matmuls with f32 accumulation). ITERS timed iterations per config (+1 compile iteration).
#   unset PYTHONPATH; COMMIT=<sha> OUT_DIR=/home/ju/repo/pncbf/p1_wm_runs/bench/iter_<x> \
#     sbatch --job-name=T13_iter --output=/home/ju/repo/pncbf/T13-logs/%x_%j.txt --export=ALL experiments/pncbf/wm/bench_train_iter.sbatch
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}
TREE=/home/ju/repo/pncbf/wm-model-${COMMIT:0:7}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
ITERS=${ITERS:-10}
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
TOTAL=$(( (ITERS + 1) * 1024 * 128 ))
T06B=(
  --algorithm.name="ncbf_ppo.flax_full_jit"
  --algorithm.nr_steps=128
  --algorithm.nr_epochs=10
  --algorithm.minibatch_size=32768
  --algorithm.learning_rate=0.0004
  --algorithm.anneal_learning_rate=True
  --algorithm.adaptive_lr=False
  --algorithm.target_kl=0
  --algorithm.clip_range=0.1
  --algorithm.gamma=0.99
  --algorithm.gae_lambda=0.9
  --algorithm.ncbf.eta_cbf=0.5
  --algorithm.ncbf.lr=0.001
  --algorithm.ncbf.coef_decay_lambda=0.95
  --algorithm.ncbf.label_discount=True
  --algorithm.ncbf.w_lip=0
  --algorithm.ncbf.w_wd=0
  --algorithm.ncbf.nr_minibatches=64
  --algorithm.ncbf.n_ensemble=5
  --algorithm.ncbf.independent_ensemble=True
  --algorithm.ncbf.ensemble_bootstrap_prob=1.0
  --algorithm.ncbf.loss_coef=1.0
  --algorithm.ncbf.output_distribution=deterministic
  --algorithm.ncbf.tighten_uncertainty=none
  --algorithm.ncbf.slack_uncertainty=none
  --algorithm.ncbf.max_delta_u=0
  --algorithm.ncbf.safety_layer_projection=soft_slack
  --algorithm.ncbf.lambda_slack=1000
  --algorithm.ncbf.action_clipping=False
  --algorithm.ncbf.policy_loss_coef=0.01
  --algorithm.ncbf.anticipation_through_shield=False
  --algorithm.ncbf.gamma_c_scaled_by_eta=True
  --algorithm.ncbf.critic_action_index=current
  --algorithm.ncbf.terminal_critic_input=pre_reset
  --algorithm.ncbf.first_step_prev_value=gamma_c
  --algorithm.ncbf.w_temporal=1.0
  --algorithm.ncbf.temporal_mode=adjacent
  --algorithm.ncbf.temporal_pairs=safe_safe
  --algorithm.ncbf_buffer.neg_buffer_size=${NEG_BUFFER:-5}
  --algorithm.ncbf_buffer.neg_sampling_ratio=0.2
  --environment.nr_envs=1024
  --environment.terrain.type=plane
  --environment.env_curriculum_level_success_episode_return=8
  --environment.command.max_velocity_per_m_factor=2.0
  --environment.command.clip_max_velocity=2.0
  --environment.command.velocity_ratio="(1.0,1.0,1.0)"
  --environment.ncbf_use_policy_observations=True
  --algorithm.ncbf.label_source=termination_window
  --environment.termination.height_curriculum_scaled=True
  --environment.termination.body_tilt_threshold=1000000.0
  --environment.name="ncbf_mujoco.robot_locomotion.mjx"
  --environment.name=ncbf_mujoco.robot_locomotion.mjx --environment.seed=42
  --algorithm.total_timesteps=$TOTAL --algorithm.evaluation_and_save_frequency=-1
  --runner.mode=train --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False
  --runner.save_model=False --runner.project_name=pncbf_bench --runner.exp_name=T13_iter
)
PPO=(--algorithm.ncbf.use_safety_layer=False --algorithm.ncbf.H=25 --algorithm.ncbf.gamma_c=0.5)
II=(--algorithm.ncbf.use_safety_layer=True --algorithm.ncbf.H=25 --algorithm.ncbf.gamma_c=0.6)
cd "$TREE/experiments"   # runs/ is gitignored
run() {  # name, args...
  local name=$1; shift
  python pncbf/wm/bench_train_iter.py --bench_out "$OUT_DIR/$name.json" "${T06B[@]}" --runner.run_name=$name "$@" 2>&1 \
    | grep -E "^iteration |iteration_s_median|Traceback|Error" | sed "s/^/$name /"
}
run ppo_none "${PPO[@]}"
run ii_h25_none "${II[@]}"
run ppo_wm_full "${PPO[@]}" --bench_wm_mode full
run ppo_wm_gated25 "${PPO[@]}" --bench_wm_mode gated --bench_gate_frac 0.25
run ppo_wm_gated50 "${PPO[@]}" --bench_wm_mode gated --bench_gate_frac 0.5
run ppo_wm_full_bf16mm "${PPO[@]}" --bench_wm_mode full --bench_matmul_dtype bfloat16
run ppo_wm_gated25_bf16mm "${PPO[@]}" --bench_wm_mode gated --bench_gate_frac 0.25 --bench_matmul_dtype bfloat16
echo "=== $(date -Is) finished"

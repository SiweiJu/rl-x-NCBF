#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --open-mode=append
# P2 F13 check (decides whether P2/P3 runs turn on environment.forward_after_state_writes): the T06b `ppo` run
# (experiments/pncbf/T06/t06.sbatch: critic trained, shield off) on pncbf-v2 >= df7e5ed, i.e. with the new default
# policy LayerNorm, and F13 ON (VARIANT=f13on, default) or OFF (VARIANT=f13off, = T06c `b` minus value LN / GAE /
# minor switches, which T06c showed to be inert). Array index = seed index (1042 1142 1242).
# Requeue-safe (runner.resume=True, fixed run dir + wandb id). Code runs from the fixed read-only tree
# /home/ju/repo/pncbf/d-train-<commit7> (TREE_PREFIX=d-train COMMIT=<sha> experiments/pncbf/T06/setup_tree.sh);
# outputs go to /home/ju/repo/pncbf/p2_runs/runs/pncbf-2026/P2-F13-<VARIANT>/F13_<VARIANT>_seed_<seed>.
#
#   unset PYTHONPATH; COMMIT=<sha> sbatch --job-name=F13 \
#     --output=/home/ju/repo/pncbf/p2_runs/logs/%x_%A_%a.txt --export=ALL experiments/pncbf/T17/f13_check.sbatch
# Smoke: --array=0 --time=00:40:00 and SMOKE=1 in the environment.
set -euo pipefail
# DEVIATION from the paper: neg_buffer_size 5 (paper 20), as in T06b.
COMMIT=${COMMIT:?set COMMIT to the full sha}
TREE=/home/ju/repo/pncbf/${TREE_PREFIX:-d-train}-${COMMIT:0:7}
VARIANT=${VARIANT:-f13on}
case $VARIANT in f13on) F13=True ;; f13off) F13=False ;; *) echo "bad VARIANT $VARIANT"; exit 1 ;; esac
I=${SLURM_ARRAY_TASK_ID:-0}
SEEDS=(1042 1142 1242)
SEED=${SEEDS[$I]:?array index out of range}
METHOD=ppo

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
# rlx-ncbf's editable rl_x points at the live /home/ju/repo/rl-x-NCBF -> force the fixed tree; cd first because
# `python -c` puts the cwd on sys.path.
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"

TOTAL=1000000000; SUFFIX=""; SMOKE_ARGS=""
if [[ -n "${SMOKE:-}" ]]; then   # 60 PPO updates of 1024x128 in 3 eval/save blocks
  TOTAL=7864320; SUFFIX="_smoke"; SMOKE_ARGS="--algorithm.evaluation_and_save_frequency=2621440"
fi
OUT=/home/ju/repo/pncbf/p2_runs; mkdir -p "$OUT/logs"
export TMPDIR=$OUT/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
GPU_MEM_LOG=$TMPDIR/gpu_mem.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$GPU_MEM_LOG" 2>/dev/null; sleep 15; done ) &
GPU_MEM_PID=$!
trap 'kill $GPU_MEM_PID 2>/dev/null; echo "=== peak GPU memory used: $(sort -n "$GPU_MEM_LOG" | tail -1) MiB"' EXIT
RUN_NAME="F13_${VARIANT}_seed_${SEED}${SUFFIX}"
TAGS="phase/P2,exp/F13,method/${METHOD},variant/${VARIANT},robot/go2,task/locomotion,commit/${COMMIT:0:7}${SMOKE:+,smoke}"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} (${SLURM_ARRAY_JOB_ID:-}_${I}) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} run=$RUN_NAME commit=$COMMIT F13=$F13"

cd "$OUT"   # runs/ and wandb/ land in p2_runs
python "$TREE/experiments/experiment.py" \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=$TOTAL \
  --algorithm.ncbf.use_safety_layer=False \
  --algorithm.ncbf.H=25 \
  --algorithm.nr_steps=128 \
  --algorithm.nr_epochs=10 \
  --algorithm.minibatch_size=32768 \
  --algorithm.learning_rate=0.0004 \
  --algorithm.anneal_learning_rate=True \
  --algorithm.adaptive_lr=False \
  --algorithm.target_kl=0 \
  --algorithm.clip_range=0.1 \
  --algorithm.gamma=0.99 \
  --algorithm.gae_lambda=0.9 \
  --algorithm.ncbf.gamma_c=0.5 \
  --algorithm.ncbf.eta_cbf=0.5 \
  --algorithm.ncbf.lr=0.001 \
  --algorithm.ncbf.coef_decay_lambda=0.95 \
  --algorithm.ncbf.label_discount=True \
  --algorithm.ncbf.w_lip=0 \
  --algorithm.ncbf.w_wd=0 \
  --algorithm.ncbf.nr_minibatches=64 \
  --algorithm.ncbf.n_ensemble=5 \
  --algorithm.ncbf.independent_ensemble=True \
  --algorithm.ncbf.ensemble_bootstrap_prob=1.0 \
  --algorithm.ncbf.loss_coef=1.0 \
  --algorithm.ncbf.output_distribution=deterministic \
  --algorithm.ncbf.tighten_uncertainty=none \
  --algorithm.ncbf.slack_uncertainty=none \
  --algorithm.ncbf.max_delta_u=0 \
  --algorithm.ncbf.safety_layer_projection=soft_slack \
  --algorithm.ncbf.lambda_slack=1000 \
  --algorithm.ncbf.action_clipping=False \
  --algorithm.ncbf.policy_loss_coef=0.01 \
  --algorithm.ncbf.anticipation_through_shield=False \
  --algorithm.ncbf.gamma_c_scaled_by_eta=True \
  --algorithm.ncbf.critic_action_index=current \
  --algorithm.ncbf.terminal_critic_input=pre_reset \
  --algorithm.ncbf.first_step_prev_value=gamma_c \
  --algorithm.ncbf.w_temporal=1.0 \
  --algorithm.ncbf.temporal_mode=adjacent \
  --algorithm.ncbf.temporal_pairs=safe_safe \
  --algorithm.ncbf_buffer.neg_buffer_size=${NEG_BUFFER:-5} \
  --algorithm.ncbf_buffer.neg_sampling_ratio=0.2 \
  --environment.nr_envs=1024 \
  --environment.terrain.type=plane \
  --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 \
  --environment.command.clip_max_velocity=2.0 \
  --environment.command.velocity_ratio="(1.0,1.0,1.0)" \
  --environment.ncbf_use_policy_observations=True \
  --algorithm.ncbf.label_source=termination_window \
  --environment.termination.height_curriculum_scaled=True \
  --environment.termination.body_tilt_threshold=1000000.0 \
  $SMOKE_ARGS \
  --environment.forward_after_state_writes=$F13 \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed=$SEED \
  --runner.mode=train \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity=catherineju-rwth-aachen-university \
  --runner.project_name=pncbf-2026 \
  --runner.exp_name="P2-F13-${VARIANT}" \
  --runner.run_name="$RUN_NAME" \
  --runner.wandb_tags="$TAGS" \
  --runner.notes="P2 F13 check (T06b ppo + policy LN default, F13=${F13}); tree=${TREE}; commit=${COMMIT}" \
  --runner.resume=True \
  --runner.save_interval_minutes=${SAVE_MINUTES:-20} \
  --runner.jax_cache_dir=/home/ju/.cache/jax_pncbf
echo "=== $(date -Is) finished"

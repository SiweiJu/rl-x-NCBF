#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --open-mode=append
# T06c bisect: why does pncbf-v2 PPO (shield off) train less robust policies than the paper code (T05)?
# = the T06b `ppo` run of experiments/pncbf/T06/t06.sbatch (same args, critic still trained, shield off) plus the
# T06c switches (rl_x/algorithms/ncbf_ppo/flax_full_jit/paper_compat.py), one VARIANT per group of 3 array tasks:
#   array index i -> VARIANT = VARIANTS[i / 3], seed = (1042 1142 1242)[i % 3]
#   a           T06b ppo (v2 defaults)                  b           all paper switches on
#   b-no-lnpol  b, policy LayerNorm off                 b-no-lnval  b, value LayerNorm off
#   b-no-gae    b, GAE cut at truncations (v2)
#   b-no-minor  b, log-std clip on + v2 anticipation loss: optional (--array=15-17), not in the default sweep because
#               both are inert for shield-off PPO (see experiments/pncbf/reports/T06c.md)
# Requeue-safe (runner.resume=True, fixed run dir + wandb id, resume checkpoint every SAVE_MINUTES) -> scavenger.
# Code runs from the fixed read-only tree /home/ju/repo/pncbf/regress-<commit7>
# (TREE_PREFIX=regress COMMIT=<sha> experiments/pncbf/T06/setup_tree.sh).
#
#   unset PYTHONPATH; COMMIT=<sha> sbatch --qos=scavenger --job-name=T06c \
#     --output=/home/ju/repo/pncbf/T06c-logs/%x_%A_%a.txt --export=ALL experiments/pncbf/T06c/t06c.sbatch
# One variant only: --array=<3k>-<3k+2>. Smoke: --array=3 --time=00:50:00 and SMOKE=1 in the environment.
set -euo pipefail
# DEVIATION from the paper: neg_buffer_size 5 (paper 20). v2's neg buffer also stores the 10-step history stack
# (~5.5 KB/sample); 20 rollouts (2.6 M samples) OOM'd a 24 GB GPU in the smoke (job 180597: alloc 16.6 GiB).
COMMIT=${COMMIT:?set COMMIT to the full sha of the regress tree}
TREE=/home/ju/repo/pncbf/regress-${COMMIT:0:7}
I=${SLURM_ARRAY_TASK_ID:-0}
VARIANTS=(a b b-no-lnpol b-no-lnval b-no-gae b-no-minor)
SEEDS=(1042 1142 1242)
VARIANT=${VARIANTS[$((I / 3))]:?array index out of range}
SEED=${SEEDS[$((I % 3))]}
METHOD=ppo; SL=False; H=25; GC=0.5   # = T06b ppo: critic still trained (paper: H=25, gamma_c 0 logit = 0.5 prob), shield off
# Paper values of the T06c switches (paper_compat.PAPER_PPO_SWITCHES); variant b-no-X sets X back to the v2 value.
LNPOL="--algorithm.policy_layer_norm=True"
LNVAL="--algorithm.value_layer_norm=True"
GAE="--algorithm.gae_stop_at_truncation=False"
MINOR="--algorithm.policy_logstd_clip=False --algorithm.ncbf.anticipation_style=sampled_action"
case $VARIANT in
  a)          T06C_ARGS="" ;;
  b)          T06C_ARGS="$LNPOL $LNVAL $GAE $MINOR" ;;
  b-no-lnpol) T06C_ARGS="$LNVAL $GAE $MINOR" ;;
  b-no-lnval) T06C_ARGS="$LNPOL $GAE $MINOR" ;;
  b-no-gae)   T06C_ARGS="$LNPOL $LNVAL $MINOR" ;;
  b-no-minor) T06C_ARGS="$LNPOL $LNVAL $GAE" ;;
  *) echo "bad VARIANT $VARIANT"; exit 1 ;;
esac

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
# rlx-ncbf's editable rl_x points at the live /home/ju/repo/rl-x-NCBF -> force the fixed tree. Also leave the
# submit directory first: `python -c` puts the cwd on sys.path, so a checkout there (e.g. pncbf/core) would win.
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"

TOTAL=1000000000; SUFFIX=""; SMOKE_ARGS=""
if [[ -n "${SMOKE:-}" ]]; then   # 60 PPO updates of 1024x128 (fills the 50-rollout pos buffer) in 3 eval/save blocks
  TOTAL=7864320; SUFFIX="_smoke"; SMOKE_ARGS="--algorithm.evaluation_and_save_frequency=2621440"
fi
# GPU memory trace (MiB, every 15 s) for the memory check; peak printed at the end.
# own TMPDIR (node /tmp files have vanished under running jobs)
export TMPDIR=/home/ju/repo/pncbf/p1_data/tmp/T06c/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
GPU_MEM_LOG=$TMPDIR/gpu_mem.txt
# (a shell loop with >> instead of `nvidia-smi -l`, whose block-buffered file output is lost when the job is killed)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$GPU_MEM_LOG" 2>/dev/null; sleep 15; done ) &
GPU_MEM_PID=$!
trap 'kill $GPU_MEM_PID 2>/dev/null; echo "=== peak GPU memory used: $(sort -n "$GPU_MEM_LOG" | tail -1) MiB"' EXIT
RUN_NAME="T06c_${VARIANT}_seed_${SEED}${SUFFIX}"
TAGS="phase/P1,exp/T06c,method/${METHOD},variant/${VARIANT},robot/go2,task/locomotion,commit/${COMMIT:0:7}${SMOKE:+,smoke}"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} (${SLURM_ARRAY_JOB_ID:-}_${I}) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} run=$RUN_NAME commit=$COMMIT variant=$VARIANT args=[$T06C_ARGS]"

cd "$TREE/experiments"   # runs/ and wandb/ are gitignored and stay writable
python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=$TOTAL \
  --algorithm.ncbf.use_safety_layer=$SL \
  --algorithm.ncbf.H=$H \
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
  --algorithm.ncbf.gamma_c=$GC \
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
  $T06C_ARGS \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed=$SEED \
  --runner.mode=train \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity=catherineju-rwth-aachen-university \
  --runner.project_name=pncbf-2026 \
  --runner.exp_name="T06c-${VARIANT}" \
  --runner.run_name="$RUN_NAME" \
  --runner.wandb_tags="$TAGS" \
  --runner.notes="T06c ${VARIANT}; tree=${TREE}; commit=${COMMIT}; args=${T06C_ARGS}" \
  --runner.resume=True \
  --runner.save_interval_minutes=${SAVE_MINUTES:-20} \
  --runner.jax_cache_dir=/home/ju/.cache/jax_pncbf
echo "=== $(date -Is) finished"

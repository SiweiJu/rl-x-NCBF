#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --open-mode=append
# T06b: the paper's Fig. 2 / Fig. 4 runs re-trained on pncbf-v2 with the paper value for every key that has a
# switch and the fixes on (B1, B2, B3/B7, F1, F2 with w_temporal=1). v2 also differs from the paper code in ~15
# ways WITHOUT a switch: see experiments/pncbf/reports/T06_config.md. METHOD in {ii_h25, ncbf_h1, ii_h10, ii_h50, ppo};
# array index = seed index. Requeue-safe (runner.resume=True, fixed run dir + wandb id, atomic checkpoints every
# SAVE_MINUTES), so it can run on --qos=scavenger. Code runs from the fixed read-only tree
# /home/ju/repo/pncbf/v2-<commit7> (make it with setup_tree.sh).
#
#   COMMIT=<sha> METHOD=ii_h25 sbatch --qos=scavenger --job-name=T06_ii_h25 \
#     --output=/home/ju/repo/pncbf/T06-logs/%x_%A_%a.txt --export=ALL experiments/pncbf/T06/t06.sbatch
# Smoke: add  --array=0 --time=00:20:00  and  SMOKE=1  in the environment.
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the full pncbf-v2 sha}
TREE=/home/ju/repo/pncbf/v2-${COMMIT:0:7}
METHOD=${METHOD:?set METHOD=ii_h25|ncbf_h1|ii_h10|ii_h50|ppo}
I=${SLURM_ARRAY_TASK_ID:-0}

SHIELD_SEEDS=(42 43 44 145 245)
case $METHOD in
  ii_h25)  SEEDS=("${SHIELD_SEEDS[@]}"); SL=True;  H=25; GC=0.6 ;;
  ncbf_h1) SEEDS=("${SHIELD_SEEDS[@]}"); SL=True;  H=1;  GC=0.6 ;;
  ii_h10)  SEEDS=("${SHIELD_SEEDS[@]}"); SL=True;  H=10; GC=0.6 ;;
  ii_h50)  SEEDS=("${SHIELD_SEEDS[@]}"); SL=True;  H=50; GC=0.6 ;;
  ppo)     SEEDS=(1042 1142 1242 1342 1442); SL=False; H=25; GC=0.5 ;;   # critic still trained (paper: H=25, gamma_c 0 logit = 0.5 prob), shield off
  *) echo "bad METHOD $METHOD"; exit 1 ;;
esac
SEED=${SEEDS[$I]}

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
# rlx-ncbf's editable rl_x points at the live /home/ju/repo/rl-x-NCBF -> force the fixed tree
export PYTHONPATH=$TREE
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"

TOTAL=1000000000; SUFFIX=""; SMOKE_ARGS=""
if [[ -n "${SMOKE:-}" ]]; then   # 40 PPO updates of 1024x128 in 2 eval/save blocks
  TOTAL=5242880; SUFFIX="_smoke"; SMOKE_ARGS="--algorithm.evaluation_and_save_frequency=2621440"
fi
RUN_NAME="T06_fixed_${METHOD}_seed_${SEED}${SUFFIX}"
TAGS="phase/P0,exp/T06,method/${METHOD},robot/go2,task/locomotion,code/fixed,commit/${COMMIT:0:7}${SMOKE:+,smoke}"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} (${SLURM_ARRAY_JOB_ID:-}_${I}) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} run=$RUN_NAME commit=$COMMIT"

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
  --algorithm.ncbf_buffer.neg_buffer_size=20 \
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
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed=$SEED \
  --runner.mode=train \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity=catherineju-rwth-aachen-university \
  --runner.project_name=pncbf-2026 \
  --runner.exp_name="T06-fixed-${METHOD}" \
  --runner.run_name="$RUN_NAME" \
  --runner.wandb_tags="$TAGS" \
  --runner.notes="T06 fixed code; tree=${TREE}; commit=${COMMIT}" \
  --runner.resume=True \
  --runner.save_interval_minutes=${SAVE_MINUTES:-20} \
  --runner.jax_cache_dir=/home/ju/.cache/jax_pncbf
echo "=== $(date -Is) finished"

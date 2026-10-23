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
# T06a: the paper's Fig. 2 / Fig. 4 shield runs re-trained with the PAPER code (cacffe7) + only the fixes
# (branch ncbf-q-fixed: F1, F2, replay action permutation; experiments/pncbf/reports/T06_config.md). Same CLI as
# T05 (experiments/pncbf/T05/paper_fig2.sbatch) plus the fix switches; compare against T05. PPO is not affected
# by the fixes (shield off: the critic never acts on the policy, and no RNG stream changes) -> reuse T05 PPO.
# NOT requeue-safe (the paper code has no resume): normal QOS only.
#
#   COMMIT=bdd62ebbc61713c7270154ed66f84a605e677f51 METHOD=ours_h25 sbatch --qos=normal --job-name=T06a_ours_h25 \
#     --output=/home/ju/repo/pncbf/T06-logs/%x_%A_%a.txt --export=ALL experiments/pncbf/T06/t06a.sbatch
# Smoke: add  --array=0 --time=00:20:00  and  SMOKE=1  in the environment.
# Attribution runs: set FIX_ARGS to override single switches, e.g.
#   FIX_ARGS="--algorithm.ncbf.temporal_mode=legacy_raw_vs_executed" (F1 only).
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the full ncbf-q-fixed sha}
TREE=/home/ju/repo/pncbf/paper-fixed-${COMMIT:0:7}
METHOD=${METHOD:?set METHOD=ours_h25|ncbf_h1|h10|h50}
I=${SLURM_ARRAY_TASK_ID:-0}
SEEDS=(42 43 44 145 245)
case $METHOD in
  ours_h25) H=25 ;;
  ncbf_h1)  H=1 ;;
  h10)      H=10 ;;
  h50)      H=50 ;;
  *) echo "bad METHOD $METHOD"; exit 1 ;;
esac
SEED=${SEEDS[$I]}

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
# rlx-ncbf's editable rl_x points at the live /home/ju/repo/rl-x-NCBF -> force the fixed tree. Also leave the
# submit directory first: `python -c` puts the cwd on sys.path, so a checkout there (e.g. pncbf/core) would win.
export PYTHONPATH=$TREE
cd "$TREE"
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"

TOTAL=1000000000; SUFFIX=""; SMOKE_ARGS=""
if [[ -n "${SMOKE:-}" ]]; then   # 40 PPO updates of 1024x128 in 2 eval/save blocks
  TOTAL=5242880; SUFFIX="_smoke"; SMOKE_ARGS="--algorithm.evaluation_and_save_frequency=2621440"
fi
FIX_ARGS=${FIX_ARGS:-}
VARIANT=${VARIANT:-fixed}   # name the run if FIX_ARGS turns a fix off, e.g. VARIANT=f1only
export WANDB_TAGS="phase/P0,exp/T06a,method/${METHOD},robot/go2,task/locomotion,code/paper-fixed,variant/${VARIANT},commit/${COMMIT:0:7}${SMOKE:+,smoke}"
RUN_NAME="T06a_${VARIANT}_${METHOD}_seed_${SEED}${SUFFIX}"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) run=$RUN_NAME commit=$COMMIT FIX_ARGS=${FIX_ARGS}"

cd "$TREE/experiments"   # runs/ and wandb/ are gitignored
# The paper code calls os.makedirs(run_dir/models) without exist_ok, so a rerun of the same RUN_NAME would crash.
# A rerun therefore starts from scratch after moving the previous attempt aside (nothing is deleted); in wandb it
# is a new run with the same name (the paper code has no fixed run id).
RUN_DIR="runs/pncbf-2026/T06a-paper-fixed-${METHOD}/${RUN_NAME}"
if [[ -e "$RUN_DIR" ]]; then
  ATTEMPTS="runs/pncbf-2026/_failed_attempts"
  mkdir -p "$ATTEMPTS"
  mv "$RUN_DIR" "$ATTEMPTS/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
  echo "moved previous attempt of $RUN_NAME to $ATTEMPTS"
fi
# shellcheck disable=SC2086
python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=$TOTAL \
  --algorithm.ncbf.use_safety_layer=True \
  --algorithm.ncbf.gamma_c=0.405 \
  --algorithm.ncbf.H=$H \
  --algorithm.ncbf.w_wd=0 \
  --algorithm.ncbf.w_lip=0 \
  --algorithm.ncbf.w_cbf=1 \
  --algorithm.ncbf.lr=0.001 \
  --algorithm.ncbf.eta_cbf=0.5 \
  --algorithm.ncbf.nr_minibatches=64 \
  --algorithm.ncbf_buffer.pos_buffer_size=50 \
  --algorithm.ncbf_buffer.neg_buffer_size=20 \
  --algorithm.ncbf_buffer.neg_sampling_ratio=0.2 \
  --algorithm.ncbf.coef_decay_lambda=0.95 \
  --algorithm.ncbf.action_clipping=False \
  --algorithm.ncbf.critic_input=pre_step \
  --algorithm.ncbf.first_step_prev_value=gamma_c \
  --algorithm.ncbf.temporal_mode=adjacent \
  --algorithm.ncbf.temporal_space=prob \
  --algorithm.ncbf.fix_replay_action_permutation=True \
  --environment.reset_last_state=observation \
  ${FIX_ARGS} ${SMOKE_ARGS} \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed=$SEED \
  --environment.nr_envs=1024 \
  --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 \
  --environment.command.clip_max_velocity=2.0 \
  --environment.terrain.type=plane \
  --environment.ncbf_use_policy_observations=True \
  --runner.mode=train \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity=catherineju-rwth-aachen-university \
  --runner.project_name=pncbf-2026 \
  --runner.exp_name="T06a-paper-fixed-${METHOD}" \
  --runner.run_name="$RUN_NAME" \
  --runner.notes="T06a paper code + fixes; code=rl-x-NCBF@${COMMIT} (ncbf-q-fixed); slurm=${SLURM_JOB_ID:-}"
echo "=== $(date -Is) finished"

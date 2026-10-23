#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-4
# T05: retrain the paper's Fig. 2 runs with the PAPER code (commit cacffe7) and the configs recovered from
# wandb 202601_paper (experiments/pncbf/paper_configs/). METHOD (env var) in {ppo, ncbf_h1, ours_h25};
# array index = seed index. The CLI below is the exact `args` list of the paper runs (wandb metadata), only
# project/group/run name changed. NOT scavenger-safe: the paper code has no resume, a requeue restarts from 0.
#
#   METHOD=ours_h25 sbatch --qos=normal --job-name=T05_ours_h25 \
#     --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%A_%a.txt --export=ALL,METHOD=ours_h25 paper_fig2.sbatch
# Smoke: add  --array=0 --time=00:20:00 --export=ALL,METHOD=ours_h25,SMOKE=1
set -euo pipefail
COMMIT=cacffe7a2f7aeac37047a76c661298b230cb8f5d
TREE=/home/ju/repo/pncbf/paper-${COMMIT:0:7}
METHOD=${METHOD:?set METHOD=ppo|ncbf_h1|ours_h25}
I=${SLURM_ARRAY_TASK_ID:-0}

case $METHOD in
  ppo)      SEEDS=(1042 1142 1242 1342 1442); SL=False; GC=0;     H=25; PAPER_GROUP=ncbf-q-no-safe ;;
  ncbf_h1)  SEEDS=(42 43 44 145 245);         SL=True;  GC=0.405; H=1;  PAPER_GROUP=ncbf-q-safe ;;
  ours_h25) SEEDS=(42 43 44 145 245);         SL=True;  GC=0.405; H=25; PAPER_GROUP=ncbf-q-safe ;;
  *) echo "bad METHOD $METHOD"; exit 1 ;;
esac
SEED=${SEEDS[$I]}

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
# rlx-ncbf's editable rl_x points at the live /home/ju/repo/rl-x-NCBF -> force the fixed tree
export PYTHONPATH=$TREE
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__"

TOTAL=1000000000; PROJECT=pncbf-2026; SUFFIX=""
if [[ -n "${SMOKE:-}" ]]; then TOTAL=5242880; SUFFIX="_smoke"; fi   # 40 PPO updates of 1024x128
export WANDB_TAGS="phase/P0,exp/T05,method/${METHOD},robot/go2,task/locomotion,code/paper,commit/${COMMIT:0:7},paper_group/${PAPER_GROUP}${SMOKE:+,smoke}"
RUN_NAME="T05_paper_${METHOD}_seed_${SEED}${SUFFIX}"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) run=$RUN_NAME commit=$COMMIT"

cd "$TREE/experiments"   # wandb records the git commit of the cwd repo; runs/ and wandb/ are gitignored
python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=$TOTAL \
  --algorithm.ncbf.use_safety_layer=$SL \
  --algorithm.ncbf.gamma_c=$GC \
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
  --runner.project_name=$PROJECT \
  --runner.exp_name="T05-paper-${METHOD}" \
  --runner.run_name="$RUN_NAME" \
  --runner.notes="T05 paper repro; code=rl-x-NCBF@${COMMIT}; paper_group=${PAPER_GROUP}; slurm=${SLURM_JOB_ID:-}"

#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-14%3
#SBATCH --open-mode=append
# T12: offline world-model data (wm_collect.py) from the 15 T06b checkpoints. Array index = checkpoint:
#   0-4 ii_h25 seeds 42 43 44 145 245 | 5-9 ncbf_h1 (same seeds) | 10-14 ppo seeds 1042 1142 1242 1342 1442.
# Held-out split: seeds 145 / 245 (ppo: 1342 / 1442) -> test. One process runs all SETTINGS for its checkpoint;
# finished shards (summary.json) are skipped on a rerun. Code runs from the fixed tree /home/ju/repo/pncbf/wm-data-<commit7>.
#
#   COMMIT=<sha> sbatch --job-name=T12 --output=/home/ju/repo/pncbf/p1_data/logs/%x_%A_%a.txt --export=ALL t12_collect.sbatch
# Dev shard: DEV=1 sbatch --array=0 ... (256 envs, one setting, out_dir p1_data/v1_dev)
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT to the full sha}
TREE=/home/ju/repo/pncbf/wm-data-${COMMIT:0:7}
I=${SLURM_ARRAY_TASK_ID:-0}
METHODS=(ii_h25 ii_h25 ii_h25 ii_h25 ii_h25 ncbf_h1 ncbf_h1 ncbf_h1 ncbf_h1 ncbf_h1 ppo ppo ppo ppo ppo)
SEEDS=(42 43 44 145 245 42 43 44 145 245 1042 1142 1242 1342 1442)
METHOD=${METHODS[$I]}; SEED=${SEEDS[$I]}
case $SEED in 145|245|1342|1442) SPLIT=test ;; *) SPLIT=train ;; esac
LABEL="T06_fixed_${METHOD}_seed_${SEED}"
SRC="/home/ju/repo/pncbf/v2-62d1616/experiments/runs/pncbf-2026/T06-fixed-${METHOD}/${LABEL}/models/latest.model"

if [[ -n "${DEV:-}" ]]; then
  OUT=/home/ju/repo/pncbf/p1_data/v1_dev; NR_ENVS=${NR_ENVS:-256}; SETTINGS=${SETTINGS:-action_noise:0.4}
else
  OUT=/home/ju/repo/pncbf/p1_data/v1; NR_ENVS=${NR_ENVS:-2048}; SETTINGS=${SETTINGS:-action_noise:0,action_noise:0.4,action_noise:0.8,push:0.005,push:0.02}
fi
ORACLE_ENVS=${ORACLE_ENVS:-256}

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf
export PYTHONPATH=$TREE
cd "$TREE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test "$(git -C "$TREE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$TREE" status --porcelain)"
python -c "import rl_x; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x:', rl_x.__file__)"
export TMPDIR=/home/ju/repo/pncbf/p1_data/tmp/${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
# private copy of the checkpoint (PPO.load unpacks next to the file)
CK="$TMPDIR/ckpt/${LABEL}/models"; mkdir -p "$CK"; cp "$SRC" "$CK/latest.model"
echo "=== $(date -Is) job ${SLURM_JOB_ID:-none} (${SLURM_ARRAY_JOB_ID:-}_${I}) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) qos=${SLURM_JOB_QOS:-?} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} ckpt=$LABEL split=$SPLIT commit=$COMMIT out=$OUT envs=$NR_ENVS settings=$SETTINGS"

cd "$TREE/experiments"
python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --runner.mode=test \
  --runner.load_model="$CK/latest.model" \
  --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False \
  --runner.project_name=pncbf_p1 --runner.exp_name=T12 --runner.run_name="T12_${LABEL}_${SLURM_JOB_ID:-local}" \
  --runner.jax_cache_dir=/home/ju/.cache/jax_pncbf \
  --environment.train_robot=unitree_go2 \
  --environment.nr_envs=$NR_ENVS \
  --environment.terrain.type=plane \
  --environment.episode_length_in_seconds=20 \
  --environment.record_wm_state=True \
  --environment.ncbf_use_policy_observations=True \
  --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 \
  --environment.command.clip_max_velocity=2.0 \
  --environment.command.velocity_ratio="(1.0,1.0,1.0)" \
  --environment.termination.height_curriculum_scaled=True \
  --environment.termination.body_tilt_threshold=1000000.0 \
  --algorithm.ncbf.use_safety_layer=False \
  --algorithm.wm_collect.enabled=True \
  --algorithm.wm_collect.out_dir="$OUT" \
  --algorithm.wm_collect.checkpoint_label="$LABEL" \
  --algorithm.wm_collect.checkpoint_source="$SRC" \
  --algorithm.wm_collect.method="$METHOD" \
  --algorithm.wm_collect.train_seed=$SEED \
  --algorithm.wm_collect.split=$SPLIT \
  --algorithm.wm_collect.settings="$SETTINGS" \
  --algorithm.wm_collect.nr_steps=${NR_STEPS:-1000} \
  --algorithm.wm_collect.oracle_nr_envs=$ORACLE_ENVS
echo "=== $(date -Is) finished"

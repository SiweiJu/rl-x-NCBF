#!/bin/bash
#SBATCH --job-name=ncbf_base
#SBATCH --output=log/out_and_err_%A_%a.txt
#SBATCH --error=log/out_and_err_%A_%a.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=71:59:59
#SBATCH --array=0-4

set -euo pipefail

WD_VALUES=(0)
LIP_VALUES=(0)
NR_MINIBATCHES=(64)
LR_VALUES=(0.001)
POS_BUFFER_SIZES=(50)
NEG_BUFFER_SIZES=(20)
GAMMA_C_VALUES=(0)
H_VALUES=(25)
SEED_VALUES=(1042 1142 1242 1342 1442)

total_jobs=$(( ${#WD_VALUES[@]} * ${#LIP_VALUES[@]} * ${#NR_MINIBATCHES[@]} * ${#LR_VALUES[@]} * ${#POS_BUFFER_SIZES[@]} * ${#GAMMA_C_VALUES[@]} * ${#H_VALUES[@]} * ${#SEED_VALUES[@]} ))

if [[ ${SLURM_ARRAY_TASK_ID} -ge ${total_jobs} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} is out of range (0-$((total_jobs-1)))"
  exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID}

wd_idx=$(( task_id % ${#WD_VALUES[@]} ))
task_id=$(( task_id / ${#WD_VALUES[@]} ))

lip_idx=$(( task_id % ${#LIP_VALUES[@]} ))
task_id=$(( task_id / ${#LIP_VALUES[@]} ))

nr_minibatches_idx=$(( task_id % ${#NR_MINIBATCHES[@]} ))
task_id=$(( task_id / ${#NR_MINIBATCHES[@]} ))

lr_idx=$(( task_id % ${#LR_VALUES[@]} ))
task_id=$(( task_id / ${#LR_VALUES[@]} ))

buffer_idx=$(( task_id % ${#POS_BUFFER_SIZES[@]} ))
task_id=$(( task_id / ${#POS_BUFFER_SIZES[@]} ))

gamma_c_idx=$(( task_id % ${#GAMMA_C_VALUES[@]} ))
task_id=$(( task_id / ${#GAMMA_C_VALUES[@]} ))

h_idx=$(( task_id % ${#H_VALUES[@]} ))
task_id=$(( task_id / ${#H_VALUES[@]} ))

seed_idx=$(( task_id % ${#SEED_VALUES[@]} ))

W_WD=${WD_VALUES[$wd_idx]}
W_LIP=${LIP_VALUES[$lip_idx]}
NR_MINIBATCH=${NR_MINIBATCHES[$nr_minibatches_idx]}
LR=${LR_VALUES[$lr_idx]}
POS_BUFFER_SIZE=${POS_BUFFER_SIZES[$buffer_idx]}
NEG_BUFFER_SIZE=${NEG_BUFFER_SIZES[$buffer_idx]}
GAMMA_C=${GAMMA_C_VALUES[$gamma_c_idx]}
H=${H_VALUES[$h_idx]}
SEED=${SEED_VALUES[$seed_idx]}

RUN_SUFFIX="seed_${SEED}"

eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf

python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=1000000000 \
  --algorithm.ncbf.use_safety_layer=False \
  --algorithm.ncbf.gamma_c="${GAMMA_C}" \
  --algorithm.ncbf.H="${H}" \
  --algorithm.ncbf.w_wd="${W_WD}" \
  --algorithm.ncbf.w_lip="${W_LIP}" \
  --algorithm.ncbf.w_cbf=1 \
  --algorithm.ncbf.lr="${LR}" \
  --algorithm.ncbf.eta_cbf=0.5 \
  --algorithm.ncbf.nr_minibatches="${NR_MINIBATCH}" \
  --algorithm.ncbf_buffer.pos_buffer_size="${POS_BUFFER_SIZE}" \
  --algorithm.ncbf_buffer.neg_buffer_size="${NEG_BUFFER_SIZE}" \
  --algorithm.ncbf_buffer.neg_sampling_ratio=0.2 \
  --algorithm.ncbf.coef_decay_lambda=0.95 \
  --algorithm.ncbf.action_clipping=False \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed="${SEED}" \
  --environment.nr_envs=1024 \
  --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=2.0 \
  --environment.command.clip_max_velocity=2.0 \
  --environment.terrain.type="plane" \
  --environment.ncbf_use_policy_observations=True \
  --runner.mode="train" \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity="catherineju-rwth-aachen-university" \
  --runner.project_name="202601_paper" \
  --runner.exp_name="ncbf-q-no-safe" \
  --runner.run_name="no_safe_${RUN_SUFFIX}"

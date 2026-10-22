#!/bin/bash

#SBATCH --job-name=ncbf_base
#SBATCH --output=log/out_and_err_%A_%a.txt
#SBATCH --error=log/out_and_err_%A_%a.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C 'vram48gb'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=25000M
#SBATCH --time=10:59:59
#SBATCH --array=0


eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate rlx-ncbf

python experiment.py \
  --algorithm.name="ncbf_ppo.flax_full_jit" \
  --algorithm.total_timesteps=50000000 \
  --algorithm.ncbf.use_safety_layer=False \
  --algorithm.ncbf.H=25 \
  --algorithm.ncbf.w_wd=0 \
  --algorithm.ncbf.lr=0.001 \
  --algorithm.ncbf.nr_minibatches=512 \
  --algorithm.ncbf_buffer.pos_buffer_size=100 \
  --algorithm.ncbf_buffer.neg_buffer_size=20 \
  --algorithm.ncbf_buffer.neg_sampling_ratio=0.1 \
  --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
  --environment.seed=42 \
  --environment.nr_envs=2048 \
  --environment.env_curriculum_level_success_episode_return=8 \
  --environment.command.max_velocity_per_m_factor=5.0 \
  --environment.command.clip_max_velocity=2.0 \
  --runner.mode="train" \
  --runner.track_console=False \
  --runner.track_tb=True \
  --runner.track_wandb=True \
  --runner.save_model=True \
  --runner.wandb_entity="catherineju-rwth-aachen-university" \
  --runner.project_name="202511_ncbf" \
  --runner.exp_name="dual_buffer_neg0d1_maxv2" \
  --runner.run_name="test"

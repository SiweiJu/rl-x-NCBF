#!/bin/bash

#SBATCH --job-name=ncbf_base
#SBATCH --output=log/out_and_err_%A_%a.txt
#SBATCH --error=log/out_and_err_%A_%a.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -C 'rtx3090|a5000|vram48gb'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=10:59:59
#SBATCH --array=0


eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"
conda activate loco_mjx

python experiment.py \
    --algorithm.name="ppo.flax_full_jit" \
    --algorithm.total_timesteps=2000011264 \
    --environment.name="custom_mujoco.robot_locomotion.mjx" \
    --environment.nr_envs=512 \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="catherineju-rwth-aachen-university" \
    --runner.project_name="202511_ncbf" \
    --runner.exp_name="base_policy" \
    --runner.run_name="base_policy" \
    --runner.notes="placeholder" \

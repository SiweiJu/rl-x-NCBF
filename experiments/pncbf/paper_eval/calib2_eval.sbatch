#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
# Calibrate the unknown eval settings (episode length, command limits) against the paper's Fig. 2 values
# for the ours-seed-42 and PPO-seed-1042 checkpoints, action noise on flat terrain.
#   sbatch --qos=normal --job-name=recover_calib2 --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%j.txt calib2_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models
# round 2 (after round 1 showed return ~0.04/step, i.e. paper's 7.85 ~ 200 steps): short horizons, 30 episodes
BASE=/home/ju/repo/pncbf/paper_eval_calib2
for EP in 3 4 5; do   # episode_length_in_seconds is an int flag
  OUT=$BASE/ep${EP} EP_SECONDS=$EP AN_PROBS="0.0 0.6" GAMMAS=0.405 SEEDS=1 NR_EPISODES=30 \
    bash paper_eval_sweep.sh $M/ours/q-safe-H-gamma_0.405_seed_42_H_25.model ours action_noise_2
  OUT=$BASE/ep${EP} EP_SECONDS=$EP AN_PROBS="0.0 0.6" GAMMAS=0.405 SEEDS=1 NR_EPISODES=30 \
    bash paper_eval_sweep.sh $M/no_safety/no_safe_seed_1042.model no_safety action_noise_2
done
python summarize_pkls.py $BASE --threshold 1

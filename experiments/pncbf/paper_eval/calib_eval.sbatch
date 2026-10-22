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
#   sbatch --qos=normal --job-name=recover_calib --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%j.txt calib_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models
BASE=/home/ju/repo/pncbf/paper_eval_calib
for EP in 5 10 20; do
  for CMD in train default; do
    if [[ $CMD == train ]]; then CLIP=2.0; else CLIP=1.0; fi
    OUT=$BASE/ep${EP}_cmd${CMD} EP_SECONDS=$EP CMD_CLIP=$CLIP AN_PROBS="0.0 0.6" GAMMAS=0.405 SEEDS=1 \
      bash paper_eval_sweep.sh $M/ours/q-safe-H-gamma_0.405_seed_42_H_25.model ours action_noise_2
    if [[ $CMD == train ]]; then
      OUT=$BASE/ep${EP}_cmd${CMD} EP_SECONDS=$EP CMD_CLIP=$CLIP AN_PROBS="0.0" GAMMAS=0.405 SEEDS=1 \
        bash paper_eval_sweep.sh $M/no_safety/no_safe_seed_1042.model no_safety action_noise_2
    fi
  done
done
python summarize_pkls.py $BASE

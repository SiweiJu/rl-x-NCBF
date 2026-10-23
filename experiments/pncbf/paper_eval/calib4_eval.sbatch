#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:45:00
# Calibration round 4: which checkpoints are the paper's "PPO-RL" (run_root no_safety)? In G0 part 1 the
# ncbf-q-no-safe models are far more robust than the paper's PPO curve, while NCBF(H=1) matches. Candidates:
# no_safety_H_sweep (branch ncbf @78d9f1c, H=25 / H=1, seed 42) and ncbf-q/no_safe_seed_42 (614c6b9).
#   sbatch --qos=normal --job-name=recover_calib4 --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%j.txt calib4_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models; BASE=/home/ju/repo/pncbf/paper_eval_calib4
export GAMMAS=0.405 SEEDS=1 NR_EPISODES=20 EP_SECONDS=5 AN_PROBS="0.0 0.2"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
( for H in 25 1; do
    TREE=/home/ju/repo/pncbf/paper-78d9f1c NCBF_POLICY_OBS="" CMD_CLIP=1.0 OUT=$BASE \
      bash paper_eval_sweep.sh "$M/no_safety_H_sweep/no_safety_H_sweep_wd_0_lip_0_mb_64_lr_0.001_buf_50_20_gc_0.5_h_${H}_seed_42.model" no_safety_H_sweep action_noise_2
  done ) & P1=$!
( OUT=$BASE bash paper_eval_sweep.sh $M/ncbf_q/no_safe_seed_42.model no_safety_ncbfq action_noise_2 ) & P2=$!
wait $P1; wait $P2
python summarize_pkls.py $BASE

#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
# Calibration round 3: disturbance strength. G0 interim showed PPO seed 1042 is far MORE robust than in the
# paper (noise 0.4 hfield: 0.5 vs 0.01; perturbation 0.02 flat: 0.9 vs 0.37), i.e. the paper's disturbances
# were stronger. Test the paper text's "noise std = 50% of action range" (paper-repro flag) and a 4x
# perturbation, flat terrain, 20 episodes of 10 s (success = len >= 250 is horizon-independent).
#   sbatch --qos=normal --job-name=recover_calib3 --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%j.txt calib3_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
M=/home/ju/repo/pncbf/paper_models; BASE=/home/ju/repo/pncbf/paper_eval_calib3
export TREE=/home/ju/repo/pncbf/paper-03ba21a GAMMAS=0.405 SEEDS=1 NR_EPISODES=20 EP_SECONDS=10
for V in "1.0 absolute" "0.5 range_fraction" "0.25 range_fraction"; do
  set -- $V
  for MOD in "no_safety no_safe_seed_1042" "ours q-safe-H-gamma_0.405_seed_42_H_25"; do
    set -- $V $MOD
    OUT=$BASE/noise_$1_$2 NOISE_STD=$1 NOISE_MODE=$2 AN_PROBS="0.2 0.6" bash paper_eval_sweep.sh $M/$3/$4.model $3 action_noise_2
  done
done
for S in 2 4; do
  OUT=$BASE/pert_x$S PERT_SCALE=$S PERT_PROBS="0.02" bash paper_eval_sweep.sh $M/no_safety/no_safe_seed_1042.model no_safety perturbation
done
python summarize_pkls.py $BASE

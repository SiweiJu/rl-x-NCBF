#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:30:00
# Smoke test of the reconstructed paper eval: ONE checkpoint (ours, seed 42, H=25), ONE sweep point
# (action_noise_2 = action noise on flat terrain, sampling prob 0, gamma_c 0.405, eval seed 1, 10 episodes).
# Submit from experiments/pncbf/paper_eval (IAS copy):
#   sbatch --job-name=recover_eval_smoke --output=/home/ju/repo/pncbf/recover-scratch/log/%x_%j.txt smoke_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
MODEL=/home/ju/repo/pncbf/paper_models/ours/q-safe-H-gamma_0.405_seed_42_H_25.model
AN_PROBS="0.0" GAMMAS="0.405" SEEDS="1" OUT=/home/ju/repo/pncbf/paper_eval_smoke \
  bash paper_eval_sweep.sh "$MODEL" ours action_noise_2
python - <<'PY'
import glob, pickle, numpy as np
for f in glob.glob("/home/ju/repo/pncbf/paper_eval_smoke/experiments/runs/ours/action_noise_2/*/rollout/*.pkl"):
    eps = pickle.load(open(f, "rb"))
    L = np.array([len(e["dones"]) for e in eps]); R = np.array([float(np.ravel(e["ret"])[0]) for e in eps])
    print(f, "\n lengths", L.tolist(), "\n returns", np.round(R, 3).tolist())
    print(f" finished(len>=250)={np.mean(L >= 250):.2f} finished(len>=max)={np.mean(L >= L.max()):.2f} avg_return={R.mean():.3f}")
    print(" constraint_active rate", float(np.mean(np.concatenate([np.ravel(np.asarray(e['constraint_active'])) for e in eps]))))
PY

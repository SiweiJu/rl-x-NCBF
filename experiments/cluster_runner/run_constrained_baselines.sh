#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/cluster_runner" ]]; then
    EXPERIMENTS_DIR="${SLURM_SUBMIT_DIR}"
    SCRIPT_DIR="${EXPERIMENTS_DIR}/cluster_runner"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${EXPERIMENTS_DIR}"

mkdir -p log

scripts=(
    train_cpo_go2_locomotion.sh
    train_cpo_g1_locomotion.sh
    train_cpo_g1_ball.sh
    train_ppo_lagrangian_go2_locomotion.sh
    train_ppo_lagrangian_g1_locomotion.sh
    train_ppo_lagrangian_g1_ball.sh
)

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "cluster_runner/${script}"
done

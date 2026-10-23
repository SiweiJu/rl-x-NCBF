#!/bin/bash
# T08: Go2 locomotion sanity runs of the constrained baselines at paper scale (1024 envs, 1e9 steps).
#   METHODS      default "ppo_lag_gradient ppo_lag_pid cpo"
#   COST_LIMITS  default "0 0.01 0.05"   (expected failures per episode)
#   SEEDS        default "0"
#   TIME         default 12:00:00 (smoke: PPO 160k, PPO-Lag 135k, CPO 70k steps/s on an A5000 -> 1e9 steps in 2-4 h + eval)
#   DRY_RUN=1    print the sbatch commands only (default)
# Run from anywhere; it cds into experiments/pncbf/baselines so that SLURM_SUBMIT_DIR is this folder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p log
METHODS="${METHODS:-ppo_lag_gradient ppo_lag_pid cpo}"
COST_LIMITS="${COST_LIMITS:-0 0.01 0.05}"
SEEDS="${SEEDS:-0}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-1}"
for method in ${METHODS}; do
    for cost_limit in ${COST_LIMITS}; do
        for seed in ${SEEDS}; do
            name="t08_${method}_d${cost_limit}_s${seed}"
            cmd=(sbatch --qos=normal --time="${TIME}" --job-name="${name}" --output="log/${name}_%j.txt"
                 --export=ALL,METHOD="${method}",COST_LIMIT="${cost_limit}",SEED="${seed}" train_go2.sbatch)
            echo "${cmd[*]}"
            if [[ "${DRY_RUN}" == "0" ]]; then "${cmd[@]}"; fi
        done
    done
done

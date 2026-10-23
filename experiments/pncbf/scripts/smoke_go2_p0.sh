#!/bin/bash
# P0 GPU smoke test: Go2 locomotion, ncbf_ppo.flax_full_jit, shield on, F1/F2 fixes on (defaults) with the
# temporal loss active, 1024 envs, 24 learning iterations, resume checkpoint every 4 iterations.
# Run on the IAS login node from the checkout:  experiments/pncbf/scripts/smoke_go2_p0.sh [RUN_NAME]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_NAME="${1:-p0_smoke_go2_$(git -C "${REPO_DIR}" rev-parse --short HEAD)}"
LOG_DIR="${REPO_DIR}/experiments/pncbf/logs"
mkdir -p "${LOG_DIR}"
NR_ENVS=1024
NR_STEPS=128
BATCH=$((NR_ENVS * NR_STEPS))
EXTRA_ARGS="--algorithm.total_timesteps=$((BATCH * 24)) --algorithm.evaluation_and_save_frequency=$((BATCH * 8)) \
--algorithm.nr_steps=${NR_STEPS} --algorithm.nr_epochs=4 --algorithm.minibatch_size=32768 \
--algorithm.learning_rate=4e-4 --algorithm.adaptive_lr=True \
--algorithm.ncbf.use_safety_layer=True --algorithm.ncbf.output_distribution=logistic_normal \
--algorithm.ncbf.H=25 --algorithm.ncbf.gamma_c=0.5 --algorithm.ncbf.eta_cbf=1.0 --algorithm.ncbf.lr=3e-4 \
--algorithm.ncbf.w_temporal=1.0 --algorithm.ncbf.max_delta_u=0.5 \
--algorithm.next_step_predictor.lr=1e-5 \
--environment.nr_envs=${NR_ENVS} --environment.seed=42 --environment.terrain.type=plane \
--environment.ncbf_use_policy_observations=True --environment.train_robot=unitree_go2 \
--environment.episode_length_in_seconds=20"
# Pass settings through the environment (--export=ALL): --export=VAR=... would split values at commas.
export REPO_DIR RUN_NAME EXTRA_ARGS SAVE_ITERS=4 SAVE_MINUTES=20
export WANDB_TAGS="phase/P0,method/pncbf_ii_fixed,robot/go2,task/locomotion,smoke"
sbatch --qos=normal --time=00:30:00 --job-name="${RUN_NAME}" --output="${LOG_DIR}/%x_%j.out" --export=ALL \
    "${REPO_DIR}/experiments/pncbf/scripts/train_requeue.sbatch"

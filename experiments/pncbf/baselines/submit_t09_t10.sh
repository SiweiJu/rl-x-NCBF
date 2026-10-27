#!/bin/bash
# T09 (P3O) / T10 (PPO + CaT) launches on IAS, normal QOS, through train_go2.sbatch -> run_baseline.sh.
#   MODE=smoke   20 updates (2.6M steps at 1024 envs, eval + save at the end), <= 20 min:
#                p3o d=0.01, ppo_lag_gradient d=0.01 (same-commit speed reference), ppo_cat on Go2 obstacles
#   MODE=sanity  1e9 steps, seed 0: p3o d in {0.01, 0.05}; ppo_cat on Go2 obstacles
#   ENV_PRESET   sanity default t06b (T06b env, the P2 setting; see run_baseline.sh), t08 = the T08 env; smoke: t08
#   WHAT         sanity only: p3o | cat | both (default both)
#   MEM_PER_CPU (default 5G; overrides train_go2.sbatch). Obstacle-task jobs: 8G approved by the user (2026-10-27).
#               (smoke MaxRSS: PPO+CaT obstacles 9.5 GB of the 10 GB of 2 x 5G)
#   KAPPA (default 20), SEEDS (default 0), TIME (default smoke 0:20:00, sanity 12:00:00), DRY_RUN=1 prints only (default)
# Run it from a fixed tree per commit (/home/ju/repo/pncbf/p3o-<shortsha>), never from a tree you keep editing.
# Obstacle setting = cluster_runner/train_go2_obstacles_sl_off.sh (2 obstacles, speed (0.5, 3.0) with curriculum)
# + OBSTACLES_PRESET=train; CaT runs with terminate_on_collision=False (collisions are CaT terminations).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p log
unset PYTHONPATH
MODE="${MODE:-sanity}"
SEEDS="${SEEDS:-0}"
KAPPA="${KAPPA:-20.0}"
DRY_RUN="${DRY_RUN:-1}"
MEM_PER_CPU="${MEM_PER_CPU:-5G}"
WHAT="${WHAT:-both}"
# Variables go through the environment (--export=ALL): values such as "(0.5, 3.0)" contain commas.
OBSTACLE_VARS=(OBSTACLES_ENABLED=True OBSTACLES_NR=2 "OBSTACLES_SPEED_RANGE=(0.5, 3.0)" OBSTACLES_SPEED_CURRICULUM=True OBSTACLES_PRESET=train)

submit() {  # submit <job name> VAR=VALUE ...
    local name="$1"; shift
    local cmd=(env "$@" sbatch --qos=normal --time="${TIME}" --job-name="${name}" --output="log/${name}_%j.txt"
               --mem-per-cpu="${MEM_PER_CPU}" --export=ALL train_go2.sbatch)
    printf '%q ' "${cmd[@]}"; echo
    if [[ "${DRY_RUN}" == "0" ]]; then "${cmd[@]}"; fi
}

case "${MODE}" in
    smoke)
        TIME="${TIME:-0:20:00}"
        COMMON=(PHASE=P2 EXP_NAME=P2_T09_T10_smoke TOTAL_TIMESTEPS=2621440 EVALUATION_AND_SAVE_FREQUENCY=2621440 TAG_EXTRA=smoke SEED=0)
        submit t09_smoke_p3o_d0.01 "${COMMON[@]}" TASK=T09 METHOD=p3o COST_LIMIT=0.01 KAPPA="${KAPPA}"
        submit t09_smoke_ppolag_d0.01 "${COMMON[@]}" TASK=T09 METHOD=ppo_lag_gradient COST_LIMIT=0.01
        submit t10_smoke_ppo_cat_obst "${COMMON[@]}" TASK=T10 METHOD=ppo_cat RUN_NAME=ppo_cat_go2_obstacles_seed0 "${OBSTACLE_VARS[@]}"
        ;;
    sanity)
        TIME="${TIME:-12:00:00}"
        ENV_PRESET="${ENV_PRESET:-t06b}"
        for seed in ${SEEDS}; do
            if [[ "${WHAT}" == "p3o" || "${WHAT}" == "both" ]]; then
                for d in 0.01 0.05; do
                    submit "t09_p3o_k${KAPPA}_d${d}_${ENV_PRESET}_s${seed}" PHASE=P2 TASK=T09 EXP_NAME=P2_T09_sanity METHOD=p3o \
                        COST_LIMIT="${d}" KAPPA="${KAPPA}" SEED="${seed}" ENV_PRESET="${ENV_PRESET}" RUN_SUFFIX="_k${KAPPA}_${ENV_PRESET}"
                done
            fi
            if [[ "${WHAT}" == "cat" || "${WHAT}" == "both" ]]; then
                submit "t10_ppo_cat_obst_${ENV_PRESET}_s${seed}" PHASE=P2 TASK=T10 EXP_NAME=P2_T10_sanity METHOD=ppo_cat SEED="${seed}" \
                    ENV_PRESET="${ENV_PRESET}" RUN_NAME="ppo_cat_go2_obstacles_${ENV_PRESET}_seed${seed}" "${OBSTACLE_VARS[@]}"
            fi
        done
        ;;
    *) echo "Unknown MODE ${MODE}" >&2; exit 2 ;;
esac

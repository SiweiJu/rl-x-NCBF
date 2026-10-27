#!/bin/bash
# Launches one constrained-baseline training run (T07/T08). Configure with environment variables:
#   METHOD      ppo_lag_gradient | ppo_lag_pid | cpo | ppo (plain PPO reference) | p3o (T09) | ppo_cat (T10, PPO + CaT)
#   KAPPA       P3O penalty factor (default 20, paper)        CAT_SOURCE / CAT_P_MAX  CaT constraint (collision, 1.0)
#   OBSTACLES_ENABLED=True + OBSTACLES_* / OBSTACLES_PRESET  moving obstacles (../../cluster_runner/obstacle_args.sh);
#               ppo_cat sets OBSTACLES_TERMINATE_ON_COLLISION=False unless given (CaT replaces the hard termination)
#   ENV_PRESET  t08 (default: T08 env, curriculum return 30, v2 default termination/command) |
#               t06b (experiments/pncbf/T06/t06.sbatch: paper height termination, tilt off, curriculum return 8,
#               command factors 2.0 / clip 2.0 / velocity_ratio (1,1,1)) -- the P2 setting, comparable with D
#   PHASE, TASK wandb tags phase/<PHASE>, task/<TASK>, exp/<TASK> (default P0, T08)
#   COST_LIMIT  expected failures per episode (cost = 1 on the failure step), e.g. 0, 0.01, 0.05
#   SEED, NR_ENVS, TOTAL_TIMESTEPS, MINIBATCH_SIZE, NR_EPOCHS, NR_STEPS, LEARNING_RATE, TAG_EXTRA, RUN_SUFFIX
# Defaults = paper training scale for Go2 locomotion (wandb 202601_paper: 1024 envs, 128 steps,
# minibatch 32768, 10 epochs, lr 4e-4, 1e9 steps, 20 s episodes, curriculum return 30).
# Extra experiment.py flags can be appended as script arguments.
set -euo pipefail

SCRIPT_DIR="${BASELINES_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXPERIMENTS_DIR="${REPO_DIR}/experiments"

: "${METHOD:?METHOD must be ppo_lag_gradient, ppo_lag_pid, cpo, ppo, p3o or ppo_cat}"
COST_LIMIT="${COST_LIMIT:-0.0}"
SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32768}"
NR_EPOCHS="${NR_EPOCHS:-10}"
NR_STEPS="${NR_STEPS:-128}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
EVALUATION_AND_SAVE_FREQUENCY="${EVALUATION_AND_SAVE_FREQUENCY:-17301504}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
ENV_PRESET="${ENV_PRESET:-t08}"
env_preset_args=()
case "${ENV_PRESET}" in
    t08) CURRICULUM_RETURN="${CURRICULUM_RETURN:-30}" ;;
    t06b)
        CURRICULUM_RETURN="${CURRICULUM_RETURN:-8}"
        env_preset_args+=(--environment.command.max_velocity_per_m_factor=2.0 --environment.command.clip_max_velocity=2.0
                          "--environment.command.velocity_ratio=(1.0,1.0,1.0)"
                          --environment.termination.height_curriculum_scaled=True
                          --environment.termination.body_tilt_threshold=1000000.0)
        ;;
    *) echo "Unknown ENV_PRESET ${ENV_PRESET}" >&2; exit 1 ;;
esac
TRAIN_ROBOT="${TRAIN_ROBOT:-unitree_go2}"
PROJECT_NAME="${PROJECT_NAME:-pncbf-2026}"
WANDB_ENTITY="${WANDB_ENTITY:-catherineju-rwth-aachen-university}"
EXP_NAME="${EXP_NAME:-P0_T08_baselines}"
TRACK_WANDB="${TRACK_WANDB:-True}"
TRACK_TB="${TRACK_TB:-True}"
TRACK_CONSOLE="${TRACK_CONSOLE:-False}"
SAVE_MODEL="${SAVE_MODEL:-True}"
CONDA_ENV="${CONDA_ENV:-rlx-ncbf}"

method_args=()
case "${METHOD}" in
    ppo_lag_gradient)
        ALGORITHM_NAME="ppo_lagrangian.flax_full_jit"
        method_args+=(--algorithm.lagrangian.update_rule=gradient --algorithm.lagrangian.cost_measure=episode)
        ;;
    ppo_lag_pid)
        ALGORITHM_NAME="ppo_lagrangian.flax_full_jit"
        method_args+=(--algorithm.lagrangian.update_rule=pid --algorithm.lagrangian.cost_measure=episode)
        ;;
    cpo)
        ALGORITHM_NAME="cpo.flax_full_jit"
        method_args+=(--algorithm.cost_measure=episode)
        ;;
    ppo)
        ALGORITHM_NAME="ppo.flax_full_jit"
        ;;
    p3o)
        ALGORITHM_NAME="p3o.flax_full_jit"
        method_args+=(--algorithm.cost_measure=episode --algorithm.p3o.kappa="${KAPPA:-20.0}")
        ;;
    ppo_cat)
        ALGORITHM_NAME="p3o.flax_full_jit"
        method_args+=(--algorithm.p3o.enabled=False --algorithm.cat.enabled=True
                      --algorithm.cat.constraint_source="${CAT_SOURCE:-collision}" --algorithm.cat.p_max="${CAT_P_MAX:-1.0}")
        if [[ "${OBSTACLES_ENABLED:-False}" == "True" ]]; then
            export OBSTACLES_TERMINATE_ON_COLLISION="${OBSTACLES_TERMINATE_ON_COLLISION:-False}"
        fi
        ;;
    *)
        echo "Unknown METHOD ${METHOD}" >&2; exit 1 ;;
esac
if [[ "${METHOD}" != "ppo" && "${METHOD}" != "ppo_cat" ]]; then
    method_args+=(--algorithm.cost_limit="${COST_LIMIT}")
fi

extra_args=()
TASK_KIND="locomotion"
if [[ "${OBSTACLES_ENABLED:-False}" == "True" ]]; then
    source "${REPO_DIR}/experiments/cluster_runner/obstacle_args.sh"
    TASK_KIND="obstacles"
fi
PHASE="${PHASE:-P0}"
TASK="${TASK:-T08}"

GIT_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
RUN_NAME="${RUN_NAME:-${METHOD}_go2_d${COST_LIMIT}_seed${SEED}${RUN_SUFFIX:-}}"
export WANDB_TAGS="phase/${PHASE},task/${TASK},exp/${TASK},method/${METHOD},robot/go2,task/${TASK_KIND},env/${ENV_PRESET},cost_limit/${COST_LIMIT}${TAG_EXTRA:+,${TAG_EXTRA}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${REPO_DIR}"  # set, not prepend (P2 brief)

if [[ -z "${PYTHON_EXE:-}" ]]; then
    PYTHON_EXE="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
fi

echo "commit=${GIT_COMMIT} env_preset=${ENV_PRESET} method=${METHOD} algorithm=${ALGORITHM_NAME} cost_limit=${COST_LIMIT} seed=${SEED} run=${RUN_NAME}"
echo "tags=${WANDB_TAGS}"
# The rlx-ncbf conda env has rl_x installed editable from /home/ju/repo/rl-x-NCBF; PYTHONPATH above makes this
# checkout win. Fail loudly if it does not.
RLX_FILE="$(cd "${EXPERIMENTS_DIR}" && "${PYTHON_EXE}" -c 'import rl_x; print(rl_x.__file__)' 2>/dev/null | tail -1)"
echo "rl_x.__file__=${RLX_FILE} git_rev=$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [[ "${RLX_FILE}" != "${REPO_DIR}/rl_x/"* ]]; then
    echo "rl_x is imported from ${RLX_FILE}, not from ${REPO_DIR}" >&2; exit 1
fi
cd "${EXPERIMENTS_DIR}"
"${PYTHON_EXE}" experiment.py \
    --algorithm.name="${ALGORITHM_NAME}" \
    --algorithm.total_timesteps="${TOTAL_TIMESTEPS}" \
    --algorithm.minibatch_size="${MINIBATCH_SIZE}" \
    --algorithm.learning_rate="${LEARNING_RATE}" \
    --algorithm.nr_steps="${NR_STEPS}" \
    --algorithm.nr_epochs="${NR_EPOCHS}" \
    --algorithm.evaluation_and_save_frequency="${EVALUATION_AND_SAVE_FREQUENCY}" \
    ${method_args[@]+"${method_args[@]}"} \
    ${env_preset_args[@]+"${env_preset_args[@]}"} \
    ${extra_args[@]+"${extra_args[@]}"} \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="${TRAIN_ROBOT}" \
    --environment.ncbf_use_policy_observations=True \
    --environment.episode_length_in_seconds="${EPISODE_SECONDS}" \
    --environment.env_curriculum_level_success_episode_return="${CURRICULUM_RETURN}" \
    --environment.terrain.type=plane \
    --runner.mode=train \
    --runner.track_console="${TRACK_CONSOLE}" \
    --runner.track_tb="${TRACK_TB}" \
    --runner.track_wandb="${TRACK_WANDB}" \
    --runner.save_model="${SAVE_MODEL}" \
    --runner.wandb_entity="${WANDB_ENTITY}" \
    --runner.project_name="${PROJECT_NAME}" \
    --runner.exp_name="${EXP_NAME}" \
    --runner.run_name="${RUN_NAME}" \
    --runner.notes="git_commit=${GIT_COMMIT} method=${METHOD} cost_limit=${COST_LIMIT} env_preset=${ENV_PRESET} kappa=${KAPPA:-}" \
    "$@"

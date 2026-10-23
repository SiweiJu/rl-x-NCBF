#!/bin/bash
# Reconstructed outer evaluation sweep of the RSS-2026 paper (F10).
#
# The paper evaluated every checkpoint with the NON-jit `ncbf_ppo.flax` test() on the CPU MuJoCo env
# `ncbf_mujoco.robot_locomotion.mujoco` (nr_envs=1, runner.mode=test) and wrote one pkl per sweep point:
#   experiments/runs/<run_root>/<setting>/<entry>/rollout/eval_safety_<T|F>_gamma_<gc>_sampling_prob_<sp>_seed_<s>.pkl
# which misc/rss/write_evaluation_to_csv.py + plot_line_plot_one_figure.py read. The loop that set the
# parameters was never committed; this script re-creates it using only CLI flags of the paper code
# at origin/ncbf-q = 01180c6, which is identical to the training commit cacffe7 under rl_x/algorithms and
# rl_x/runner. (The CPU env at cacffe7 crashes: NameError jnp in get_observation_space, removed in e0084af;
# e0084af's qpos/qvel change is a no-op for Go2.) So no code change is needed.
#
# Usage (one checkpoint per call; runs the points sequentially):
#   TREE=/home/ju/repo/pncbf/paper-01180c6 OUT=/home/ju/repo/pncbf/paper_eval \
#   bash paper_eval_sweep.sh <model.model> <run_root> [settings...]
# run_root: no_safety | NCBF | ours | H10 | H50 | safe_fall | ncbf   (legend keys of the plot scripts)
# settings: action_noise_2 action_noise_hfield perturbation perturbation_hfield  (default: all four)
# Env overrides: GAMMAS (default: gamma_c stored in the checkpoint), SEEDS (default "1"),
#   NR_EPISODES (default 10), EP_SECONDS (default 10), AN_PROBS, PERT_PROBS, PERT_SCALE (default 2),
#   NOISE_STD/NOISE_MODE (paper-repro tree only; e.g. 0.5 range_fraction = paper text '50% of action range'),
#   CMD_CLIP/CMD_FACTOR (command limits, default 2.0/2.0 as in training),
#   SAFETY (default: use_safety_layer stored in the checkpoint), DRY_RUN=1 prints the commands only.
#
# --algorithm.minibatch_size=128: PPO.__init__ asserts (nr_steps*nr_envs) % minibatch_size == 0; with the
#   checkpoint's 128 steps x 1 env and 32768 minibatch that fails, so the paper eval must have overridden it
#   too. It does not affect test().
# Recovered facts (see experiments/pncbf/reports/T03_T05.md):
#   - x values of the paper figures: action noise {0,0.2,0.4,0.6,0.8,1}, perturbation {0.002,...,0.02}
#   - 10 episodes per pkl and 5 entries (models) per method for ours/H10/H50/SafeFall/state critic
#   - PPO-RL points pool 8 (action noise) / 7 (perturbation) pkls per model -> a gamma list was looped
#     even for the no-safety models; the exact gamma values and eval seeds are NOT recoverable.
#   - action noise: N(0,1) added to the processed action with prob. sp per step (test() in ppo.py);
#     perturbation: env DR perturbation, sampling_probability=sp, magnitude x2 (paper App. "we double the
#     magnitude"); curriculum coeff is forced to 1 in the CPU env (eval_mode=True).
#   - Episode length used at eval is not recorded; EP_SECONDS=10 follows the P0 plan (§6.3).
set -euo pipefail

MODEL=$(readlink -f "$1"); RUN_ROOT=$2; shift 2
SETTINGS=${*:-"action_noise_2 action_noise_hfield perturbation perturbation_hfield"}
TREE=${TREE:-/home/ju/repo/pncbf/paper-01180c6}
OUT=${OUT:-/home/ju/repo/pncbf/paper_eval}
SEEDS=${SEEDS:-1}
NR_EPISODES=${NR_EPISODES:-10}
EP_SECONDS=${EP_SECONDS:-10}   # int seconds (absl int flag)
CMD_CLIP=${CMD_CLIP:-2.0}        # training value; CPU-env default is 1.0
CMD_FACTOR=${CMD_FACTOR:-2.0}
AN_PROBS=${AN_PROBS:-"0.0 0.2 0.4 0.6 0.8 1.0"}
PERT_PROBS=${PERT_PROBS:-"0.002 0.005 0.01 0.015 0.02"}
PERT_SCALE=${PERT_SCALE:-2}
ENTRY=$(basename "$MODEL" .model)

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "rlx-ncbf" ]]; then
  eval "$(/home/ju/miniconda3/bin/conda shell.bash hook)"; conda activate rlx-ncbf
fi
# rlx-ncbf has an editable rl_x install pointing at the LIVE tree /home/ju/repo/rl-x-NCBF; force the fixed tree.
export PYTHONPATH=$TREE
python -c "import rl_x,sys; assert rl_x.__file__.startswith('$TREE'), rl_x.__file__; print('rl_x from', rl_x.__file__)"
echo "tree commit: $(git -C "$TREE" rev-parse HEAD)  dirty: $(git -C "$TREE" status --porcelain | wc -l)"

# per-call private copy: PPO.load() unpacks into <model dir>/tmp and deletes it afterwards
WORK=$(mktemp -d "${TMPDIR:-/tmp}/paper_eval_XXXXXX"); trap 'rm -rf "$WORK"' EXIT
cp "$MODEL" "$WORK/$ENTRY.model"
read -r CKPT_SL CKPT_GC < <(python - "$WORK/$ENTRY.model" <<'PY'
import json, sys, zipfile
c = json.load(zipfile.ZipFile(sys.argv[1]).open("config_algorithm.json"))
print(c["ncbf"]["use_safety_layer"], c["ncbf"]["gamma_c"])
PY
)
SAFETY=${SAFETY:-$CKPT_SL}
GAMMAS=${GAMMAS:-$CKPT_GC}
# branch ncbf (version (i), 78d9f1c) has no ncbf_use_policy_observations flag: run with NCBF_POLICY_OBS=""
NCBF_POLICY_OBS=${NCBF_POLICY_OBS-True}
NCBF_POLICY_OBS_ARG=(); [[ -n $NCBF_POLICY_OBS ]] && NCBF_POLICY_OBS_ARG=(--environment.ncbf_use_policy_observations=$NCBF_POLICY_OBS)
tok() { python -c "import sys; v=float(sys.argv[1]); print(str(abs(v)).replace('.','p') if v>=0 else 'm'+str(abs(v)).replace('.','p'))" "$1"; }

mkdir -p "$OUT/experiments"; cd "$OUT/experiments"
for SETTING in $SETTINGS; do
  case $SETTING in
    action_noise_2)       TERRAIN=plane;          KIND=noise ;;
    action_noise_hfield)  TERRAIN=hfield_diverse; KIND=noise ;;
    perturbation)         TERRAIN=plane;          KIND=pert ;;
    perturbation_hfield)  TERRAIN=hfield_diverse; KIND=pert ;;
    *) echo "unknown setting $SETTING"; exit 1 ;;
  esac
  if [[ $KIND == noise ]]; then PROBS=$AN_PROBS; else PROBS=$PERT_PROBS; fi
  for SP in $PROBS; do for GC in $GAMMAS; do for SEED in $SEEDS; do
    NAME="eval_safety_${SAFETY}_gamma_$(tok "$GC")_sampling_prob_$(tok "$SP")_seed_${SEED}"
    PKL="runs/$RUN_ROOT/$SETTING/$ENTRY/rollout/$NAME.pkl"
    if [[ -f $PKL ]]; then echo "[skip] $PKL"; continue; fi
    if [[ $KIND == noise ]]; then
      X_ARGS=(--algorithm.action_noise_sampling_ratio="$SP")
      # needs a tree with branch paper-repro (03ba21a): eval-only noise-std flags
      [[ -n "${NOISE_STD:-}" ]] && X_ARGS+=(--algorithm.action_noise_std="$NOISE_STD" --algorithm.action_noise_std_mode="${NOISE_MODE:-absolute}")
    else
      X_ARGS=(--algorithm.action_noise_sampling_ratio=0.0
              --environment.domain_randomization.perturbation.sampling_probability="$SP"
              --environment.domain_randomization.perturbation.trunk_velocity_clip_mass_factor=$(python -c "print(0.1*$PERT_SCALE)")
              --environment.domain_randomization.perturbation.trunk_velocity_clip_limit=$(python -c "print(1.0*$PERT_SCALE)")
              --environment.domain_randomization.perturbation.max_joint_velocity=$(python -c "print(0.5*$PERT_SCALE)")
              --environment.domain_randomization.perturbation.max_joint_position=$(python -c "print(0.01*$PERT_SCALE)"))
    fi
    CMD=(python "$TREE/experiments/experiment.py"
      --algorithm.name="ncbf_ppo.flax"
      --environment.name="ncbf_mujoco.robot_locomotion.mujoco"
      --runner.mode=test
      --runner.load_model="$WORK/$ENTRY.model"
      --runner.nr_test_episodes="$NR_EPISODES"
      --runner.track_console=False --runner.track_tb=False --runner.track_wandb=False --runner.save_model=False
      --runner.project_name="$RUN_ROOT" --runner.exp_name="$SETTING" --runner.run_name="$ENTRY"
      --algorithm.rollout_save_name="$NAME"
      --algorithm.minibatch_size=128
      --algorithm.ncbf.gamma_c="$GC"
      --algorithm.ncbf.use_safety_layer="$SAFETY"
      "${X_ARGS[@]}"
      --environment.seed="$SEED"
      --environment.nr_envs=1
      --environment.terrain.type="$TERRAIN"
      --environment.episode_length_in_seconds="$EP_SECONDS"
      ${NCBF_POLICY_OBS_ARG[@]+"${NCBF_POLICY_OBS_ARG[@]}"}
      --environment.env_curriculum_level_success_episode_return=8
      --environment.command.max_velocity_per_m_factor="$CMD_FACTOR"
      --environment.command.clip_max_velocity="$CMD_CLIP")
    echo "[run] $PKL"
    if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "${CMD[@]}"; echo; continue; fi
    t0=$(date +%s); "${CMD[@]}"; echo "[done] $PKL in $(( $(date +%s) - t0 )) s"
  done; done; done
done

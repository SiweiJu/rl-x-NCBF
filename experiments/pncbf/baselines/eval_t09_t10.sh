#!/bin/bash
# T09/T10 evaluation of the sanity checkpoints (submit_t09_t10.sh MODE=sanity), normal QOS (evaluations are single-shot).
#   TREE      tree the sanity runs were started from (required), e.g. /home/ju/repo/pncbf/p3o-<sha>
#   WHAT      p3o | cat | both (default both)
#   KAPPA     kappa used in the sanity run names (default 20.0)
#   ENV_PRESET t06b (default; T06b env args + G0 part 2 protocol, the D comparison) | t08 (T08 env + protocol_default)
#   OUT_DIR   default /home/ju/repo/pncbf/p2_runs/T09_T10/eval
#   DRY_RUN=1 writes the checkpoint lists and prints the sbatch commands only
# P3O: T04 sweep, mode A, flat + hfield: 2 checkpoints x 2 terrains. t06b: G0 part 2 protocol (as T06b / D); t08: protocol_default (as T08).
# CaT: Go2 obstacles, flat, no perturbation (action_noise p = 0), one entry per obstacle speed 0.5..4.0 m/s
#      (training preset `train`, terminate_on_collision=True at eval so success = no fall and no collision;
#      eval_summary.json also has obstacle_collision).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # experiments/
mkdir -p log pncbf/baselines/eval_lists
unset PYTHONPATH
: "${TREE:?TREE must be the sanity-run tree}"
WHAT="${WHAT:-both}"
KAPPA="${KAPPA:-20.0}"
ENV_PRESET="${ENV_PRESET:-t06b}"
case "${ENV_PRESET}" in
    t06b) PROTOCOL="$(pwd)/pncbf/eval/g0_part2_protocol.json" ;;
    t08) PROTOCOL="$(pwd)/pncbf/eval/protocol_default.json" ;;
    *) echo "Unknown ENV_PRESET ${ENV_PRESET}" >&2; exit 2 ;;
esac
export TMPDIR_EVAL="${TMPDIR_EVAL:-/home/ju/repo/pncbf/p2_runs/tmp/t09_t10_eval}"
OUT_DIR="${OUT_DIR:-/home/ju/repo/pncbf/p2_runs/T09_T10/eval}"
PY="${PYTHON_EXE:-/home/ju/miniconda3/envs/rlx-ncbf/bin/python}"
RUNS="${TREE}/experiments/runs/pncbf-2026"
LISTS="$(pwd)/pncbf/baselines/eval_lists"

# Obstacle env flags of the training preset (speed set per entry below).
extra_args=()
unset OBSTACLES_SPEED_RANGE OBSTACLES_SPEED_CURRICULUM OBSTACLES_TERMINATE_ON_COLLISION
OBSTACLES_ENABLED=True OBSTACLES_NR=2 OBSTACLES_PRESET=train
export OBSTACLES_ENABLED OBSTACLES_NR OBSTACLES_PRESET
source "$(pwd)/cluster_runner/obstacle_args.sh"
OBSTACLE_ARGS_JSON="$(printf '%s\n' "${extra_args[@]}" | "${PY}" -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')"

ENV_PRESET="${ENV_PRESET}" RUNS="${RUNS}" KAPPA="${KAPPA}" LISTS="${LISTS}" OBSTACLE_ARGS_JSON="${OBSTACLE_ARGS_JSON}" "${PY}" - <<'EOF'
import json, os
runs, kappa, lists, preset = os.environ["RUNS"], os.environ["KAPPA"], os.environ["LISTS"], os.environ["ENV_PRESET"]
if preset == "t06b":  # = g0_part2_groups.json T06b env_args (training env of experiments/pncbf/T06/t06.sbatch)
    base_env = ["--environment.ncbf_use_policy_observations=True", "--environment.env_curriculum_level_success_episode_return=8",
                "--environment.command.max_velocity_per_m_factor=2.0", "--environment.command.clip_max_velocity=2.0",
                "--environment.command.velocity_ratio=(1.0,1.0,1.0)", "--environment.termination.height_curriculum_scaled=True",
                "--environment.termination.body_tilt_threshold=1000000.0"]
    suffix = "_t06b"
else:
    base_env = ["--environment.ncbf_use_policy_observations=True", "--environment.env_curriculum_level_success_episode_return=30"]
    suffix = ""
p3o = [{
    "label": f"t09_p3o_k{kappa}_d{d}{suffix}_s0", "method": f"p3o_k{kappa}_d{d}", "algorithm": "p3o.flax_full_jit",
    "path": f"{runs}/P2_T09_sanity/p3o_go2_d{d}_seed0_k{kappa}{suffix}/models/latest.model",
    "train_seed": 0, "shield": False, "env_args": base_env, "algo_args": [],
} for d in ("0.01", "0.05")]
obstacle_args = json.loads(os.environ["OBSTACLE_ARGS_JSON"])
cat = [{
    "label": f"t10_ppo_cat{suffix}_s0_v{v}", "method": "ppo_cat", "algorithm": "p3o.flax_full_jit",
    "path": f"{runs}/P2_T10_sanity/ppo_cat_go2_obstacles{suffix}_seed0/models/latest.model",
    "train_seed": 0, "shield": False, "algo_args": [],
    "env_args": base_env + obstacle_args + ["--environment.obstacles.terminate_on_collision=True",
                                            "--environment.obstacles.speed_curriculum=False",
                                            f"--environment.obstacles.speed_range=({v}, {v})"],
} for v in ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0")]
for name, entries in (("t09_p3o", p3o), ("t10_cat", cat)):
    missing = [e["path"] for e in entries if not os.path.exists(e["path"])]
    if missing:
        print(f"WARNING {name}: missing checkpoints {sorted(set(missing))}")
    json.dump(entries, open(f"{lists}/{name}.json", "w"), indent=1)
    print(f"{lists}/{name}.json: {len(entries)} entries")
EOF

submit() {  # submit <name> <checkpoints json> <sweep args>
    local name="$1" ckpts="$2" sweep_args="$3"
    local n
    n=$(PYTHONPATH="$(cd .. && pwd)" "${PY}" pncbf/eval/run_eval_sweep.py --checkpoints_file "${ckpts}" --out_dir "${OUT_DIR}" ${sweep_args} --list_jobs | tail -1 | sed -E 's/^([0-9]+) jobs.*/\1/')
    local cmd=(sbatch --qos=normal --time=1:00:00 --array="0-$((n-1))%3" --job-name="${name}" --output="log/${name}_%A_%a.txt"
               --export=ALL,TMPDIR="${TMPDIR_EVAL}",CHECKPOINTS_FILE="${ckpts}",OUT_DIR="${OUT_DIR}",SWEEP_ARGS="${sweep_args}" pncbf/eval/eval_sweep.sbatch)
    echo "${cmd[*]}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
}
if [[ "${WHAT}" == "p3o" || "${WHAT}" == "both" ]]; then
    [[ "${DRY_RUN:-0}" == "1" ]] || mkdir -p "${TMPDIR_EVAL}"
    submit t09_eval "${LISTS}/t09_p3o.json" "--protocol ${PROTOCOL} --modes A"
fi
if [[ "${WHAT}" == "cat" || "${WHAT}" == "both" ]]; then
    [[ "${DRY_RUN:-0}" == "1" ]] || mkdir -p "${TMPDIR_EVAL}"
    submit t10_eval "${LISTS}/t10_cat.json" "--protocol ${PROTOCOL} --modes A --terrains flat --perturbations action_noise --sampling_probs 0.0"
fi

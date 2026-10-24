#!/usr/bin/env python3
"""G0 part 2 driver: evaluate the final checkpoints of the paper runs, T05, T06a and T06b with one protocol.

One job = one checkpoint x one terrain (both deployment modes and all sweep points inside one process). A job calls
the run_eval_sweep.py of the code tree the checkpoint belongs to (paper code for paper/T05, ncbf-q-fixed-eval for
T06a, pncbf-v2-eval-g0 for T06b) with that tree's conda env. Output:
    <out_dir>/<group>/<group>_<method>_s<seed>/<perturbation>_<terrain>/mode_<A|B>/sp_<p>/seed_0/eval_summary.json

    python3 g0_part2.py --list_jobs [--groups T05 T06a]          # job table (array indices) and missing checkpoints
    python3 g0_part2.py --task_index 7                           # run one job (used by g0_part2.sbatch)
    python3 g0_part2.py --task_index 7 --dry_run                 # print the command
Jobs whose checkpoint is missing or was modified less than --min_age_minutes ago (still training) are refused
unless --force. Points whose eval_summary.json exists are skipped (requeue-safe).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDA_ROOT = os.environ.get("CONDA_ROOT", "/home/ju/miniconda3")


def load_groups(path):
    config = json.loads(Path(path).read_text())
    protocol_path = Path(config["protocol"])
    if not protocol_path.is_absolute():
        protocol_path = Path(path).resolve().parent / protocol_path
    return config, json.loads(protocol_path.read_text()), protocol_path


def checkpoints(config, groups=None):
    out = []
    for group, spec in config["groups"].items():
        if groups and group not in groups:
            continue
        for method, mspec in spec["methods"].items():
            for seed in mspec["seeds"]:
                out.append({"group": group, "method": method, "seed": seed, "shield": mspec["shield"],
                            "path": mspec["path"].format(seed=seed), "label": f"{group}_{method}_s{seed}"})
    return out


def jobs(config, protocol, groups=None):
    return [dict(ckpt, terrain=terrain) for ckpt in checkpoints(config, groups) for terrain in protocol["terrains"]]


def checkpoint_status(path, min_age_minutes):
    if not os.path.exists(path):
        return "missing"
    age = (time.time() - os.path.getmtime(path)) / 60
    return f"recent ({age:.0f} min)" if age < min_age_minutes else "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups_file", default=str(HERE / "g0_part2_groups.json"))
    ap.add_argument("--groups", nargs="*", default=None, help="subset of groups (paper T05 T06a T06b)")
    ap.add_argument("--out_dir", default=None, help="override the groups file out_dir")
    ap.add_argument("--task_index", type=int, default=None)
    ap.add_argument("--list_jobs", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if the checkpoint is recent (still training)")
    ap.add_argument("--min_age_minutes", type=float, default=30.0)
    ap.add_argument("--sweep_args", default="", help="extra run_eval_sweep.py arguments (smoke: '--sampling_probs 0.0 0.6')")
    args = ap.parse_args()

    config, protocol, protocol_path = load_groups(args.groups_file)
    job_list = jobs(config, protocol, args.groups)
    out_dir = args.out_dir or config["out_dir"]

    if args.list_jobs:
        for i, job in enumerate(job_list):
            print(f"{i}\t{job['label']}\t{job['terrain']}\t{checkpoint_status(job['path'], args.min_age_minutes)}")
        ready = sum(checkpoint_status(j["path"], args.min_age_minutes) == "ok" for j in job_list)
        print(f"{len(job_list)} jobs ({ready} with a finished-looking checkpoint) -> sbatch --array=0-{len(job_list) - 1}")
        return

    if args.task_index is None:
        raise SystemExit("pass --task_index (or --list_jobs)")
    job = job_list[args.task_index]
    group = config["groups"][job["group"]]
    status = checkpoint_status(job["path"], args.min_age_minutes)
    if status != "ok" and not args.force:
        raise SystemExit(f"checkpoint {status}: {job['path']} (use --force to evaluate anyway)")
    python = f"{CONDA_ROOT}/envs/{group['conda_env']}/bin/python"
    tree = Path(group["tree"])
    entry = {"label": job["label"], "path": job["path"], "method": job["method"], "train_seed": job["seed"],
             "shield": job["shield"], "env_args": group.get("env_args", []), "algo_args": group.get("algo_args", [])}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([entry], f)
        ckpt_file = f.name
    cmd = [python, str(tree / "experiments/pncbf/eval/run_eval_sweep.py"), "--checkpoints_file", ckpt_file,
           "--protocol", str(protocol_path), "--out_dir", os.path.join(out_dir, job["group"]),
           "--terrains", job["terrain"], "--task_index", "0", "--python", python, *args.sweep_args.split()]
    print(f"[g0_part2 job {args.task_index}] {job['label']} {job['terrain']} tree={tree} env={group['conda_env']}\n  "
          + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    result = subprocess.run(cmd, cwd=tree / "experiments")
    os.unlink(ckpt_file)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

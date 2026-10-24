#!/usr/bin/env python3
"""Perturbation-sweep evaluation driver (T04): checkpoint x test setting x sampling probability x eval seed.

One job = one checkpoint on one terrain in one deployment mode. A job is one `experiments/experiment.py
--runner.mode=test --algorithm.eval_sweep.enabled=True` process that loads the checkpoint once and evaluates both
perturbations (action noise, push) at every sampling probability and eval seed in-process
(rl_x/algorithms/ncbf_ppo/flax_full_jit/eval_sweep.py). Output, one directory per sweep point:

    <out_dir>/<label>/<perturbation>_<terrain>/mode_<A|B>/sp_<p>/seed_<s>/{eval_summary.json, eval_episodes.npz}

Checkpoints come from a JSON list (--checkpoints_file) or --checkpoint label=path (repeatable). JSON entries:
    {"label": "ours_h25_s1", "path": ".../models/latest.model", "method": "ours_h25", "train_seed": 1,
     "shield": true, "env_args": [...], "algo_args": ["--algorithm.ncbf.gamma_c=5", ...]}
optional "algorithm" (default ncbf_ppo.flax_full_jit; ppo_lagrangian.flax_full_jit / cpo.flax_full_jit are evaluated
policy-only, mode A, shield defaults to false).
`env_args` must reproduce the training env (not stored in the checkpoint); `algo_args` must give the shield's
runtime parameters (gamma_c, eta_cbf, ... are not restored from the checkpoint). Mode B is skipped for entries
with "shield": false.

Examples:
    python pncbf/eval/run_eval_sweep.py --checkpoints_file ckpts.json --out_dir runs/pncbf_eval/T05 --list_jobs
    python pncbf/eval/run_eval_sweep.py --checkpoints_file ckpts.json --out_dir ... --task_index $SLURM_ARRAY_TASK_ID
    python pncbf/eval/run_eval_sweep.py --checkpoint test=/path/latest.model --terrains flat \
        --sampling_probs 0.0 0.01 --perturbations action_noise --out_dir /tmp/sweep --dry_run
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = EXPERIMENTS_DIR.parent
NCBF_ALGORITHM = "ncbf_ppo.flax_full_jit"
DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "protocol_default.json"


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_checkpoints(args):
    entries = []
    if args.checkpoints_file:
        entries.extend(json.loads(Path(args.checkpoints_file).read_text()))
    for spec in args.checkpoint or []:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint expects label=path, got {spec}")
        entries.append({"label": label, "path": path})
    if not entries:
        raise SystemExit("No checkpoints: pass --checkpoints_file or --checkpoint label=path")
    for entry in entries:
        entry.setdefault("algorithm", NCBF_ALGORITHM)
        if entry["algorithm"] != NCBF_ALGORITHM:
            entry.setdefault("shield", False)  # ppo_lagrangian / cpo: policy only, mode A
        entry.setdefault("method", "")
        entry.setdefault("train_seed", -1)
        entry.setdefault("shield", True)
        entry.setdefault("env_args", [])
        entry.setdefault("algo_args", [])
        entry["path"] = str(Path(entry["path"]).expanduser().resolve())
    return entries


# Shield parameters that PPO.load (pncbf-v2) does not restore from the checkpoint ("runtime" parameters).
RUNTIME_NCBF_KEYS = ("gamma_c", "eta_cbf", "lambda_slack", "action_clipping", "max_delta_u", "safety_layer_projection",
                     "safety_layer_min_grad_norm", "safety_layer_std_coeff_start", "safety_layer_std_coeff_final",
                     "gamma_c_scaled_by_eta", "tighten_uncertainty", "slack_uncertainty", "first_step_prev_value")


def shield_args_from_checkpoint(path):
    """--algorithm.ncbf.<key>=<value> for the shield parameters stored in the checkpoint's config (gamma_c etc. as
    trained). Paper-code checkpoints get them restored anyway; for pncbf-v2 checkpoints PPO.load skips them."""
    import zipfile
    try:
        config = json.load(zipfile.ZipFile(path).open("config_algorithm.json"))
    except Exception as e:
        print(f"[warn] cannot read config_algorithm.json from {path}: {e}")
        return []
    ncbf = config.get("ncbf", {})
    out = []
    for key in RUNTIME_NCBF_KEYS:
        if key in ncbf:
            value = ncbf[key]
            out.append(f"--algorithm.ncbf.{key}={value}")
    return out


def load_protocol(args):
    protocol = json.loads(Path(args.protocol).read_text())
    overrides = {
        "nr_envs": args.nr_envs, "episode_length_in_seconds": args.episode_length_in_seconds,
        "modes": args.modes, "perturbations": args.perturbations, "eval_seeds": args.eval_seeds,
        "noise_std": args.noise_std, "noise_std_mode": args.noise_std_mode, "noise_point": args.noise_point,
        "push_scale": args.push_scale, "push_impl": args.push_impl, "background_push_prob": args.background_push_prob,
        "initial_command_zero_prob": args.initial_command_zero_prob,
        "paper_shield": args.paper_shield,
        "skip_existing": args.skip_existing,
    }
    protocol.update({k: v for k, v in overrides.items() if v is not None})
    if args.sampling_probs is not None:
        protocol["sampling_probs"] = {name: args.sampling_probs for name in ("action_noise", "push")}
    if args.terrains is not None:
        protocol["terrains"] = {name: protocol["terrains"][name] for name in args.terrains}
    return protocol


def build_jobs(checkpoints, protocol):
    jobs = []
    for entry in checkpoints:
        for terrain_label, terrain_type in protocol["terrains"].items():
            modes = [m for m in protocol["modes"] if m != "B" or entry["shield"]]
            if protocol.get("modes_in_one_job", False):
                if modes:  # one process per checkpoint x terrain: load once, run the modes one after the other
                    jobs.append({"checkpoint": entry, "terrain_label": terrain_label, "terrain_type": terrain_type, "mode": ",".join(modes)})
                continue
            for mode in modes:
                jobs.append({"checkpoint": entry, "terrain_label": terrain_label, "terrain_type": terrain_type, "mode": mode})
    return jobs


def _csv(values):
    return ",".join(f"{float(v):g}" for v in values)


def job_command(job, protocol, out_dir, python):
    entry = job["checkpoint"]
    sp = protocol["sampling_probs"]
    cmd = [
        python, "experiment.py",
        f"--algorithm.name={entry['algorithm']}",
        "--environment.name=ncbf_mujoco.robot_locomotion.mjx",
        "--runner.mode=test",
        f"--runner.load_model={entry['path']}",
        f"--algorithm.eval_sweep.checkpoint_source={entry.get('source_path', entry['path'])}",
        "--runner.track_console=False", "--runner.track_tb=False", "--runner.track_wandb=False", "--runner.save_model=False",
        "--runner.project_name=pncbf_eval", "--runner.exp_name=eval_sweep",
        f"--runner.run_name={entry['label']}_{job['terrain_label']}_mode{job['mode'].replace(',', '')}",
        f"--environment.nr_envs={protocol['nr_envs']}",
        f"--environment.terrain.type={job['terrain_type']}",
        f"--environment.episode_length_in_seconds={protocol['episode_length_in_seconds']}",
        *protocol.get("env_args", []), *entry["env_args"],
        *(shield_args_from_checkpoint(entry.get("source_path", entry["path"])) if entry.get("shield_args_from_checkpoint", True) and entry["algorithm"] == NCBF_ALGORITHM else []),
        *protocol.get("algo_args", []), *entry["algo_args"],
        *([f"--algorithm.ncbf.use_safety_layer={'B' in job['mode']}"] if entry["algorithm"] == NCBF_ALGORITHM else []),
        "--algorithm.eval_sweep.enabled=True",
        f"--algorithm.eval_sweep.out_dir={out_dir}",
        f"--algorithm.eval_sweep.checkpoint_label={entry['label']}",
        f"--algorithm.eval_sweep.method={entry['method']}",
        f"--algorithm.eval_sweep.train_seed={int(entry['train_seed'])}",
        f"--algorithm.eval_sweep.terrain_label={job['terrain_label']}",
        f"--algorithm.eval_sweep.perturbations={','.join(protocol['perturbations'])}",
        f"--algorithm.eval_sweep.action_noise_sampling_probs={_csv(sp['action_noise'])}",
        f"--algorithm.eval_sweep.push_sampling_probs={_csv(sp['push'])}",
        f"--algorithm.eval_sweep.eval_seeds={','.join(str(int(s)) for s in protocol['eval_seeds'])}",
        f"--algorithm.eval_sweep.deployment_mode={job['mode']}",
        f"--algorithm.eval_sweep.noise_std={protocol['noise_std']}",
        f"--algorithm.eval_sweep.noise_std_mode={protocol['noise_std_mode']}",
        f"--algorithm.eval_sweep.noise_point={protocol['noise_point']}",
        f"--algorithm.eval_sweep.push_scale={protocol['push_scale']}",
        f"--algorithm.eval_sweep.push_impl={protocol['push_impl']}",
        f"--algorithm.eval_sweep.background_push_prob={protocol['background_push_prob']}",
        f"--algorithm.eval_sweep.success_threshold_steps={int(protocol['success_threshold_steps'])}",
        f"--algorithm.eval_sweep.paper_shield={protocol.get('paper_shield', 'train')}",
        f"--algorithm.eval_sweep.initial_command_zero_prob={float(protocol.get('initial_command_zero_prob', 1.0))}",
        f"--algorithm.eval_sweep.skip_existing={bool(protocol.get('skip_existing', False))}",
        f"--algorithm.eval_sweep.push_scale_joints={bool(protocol.get('push_scale_joints', False))}",
    ]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints_file", default=None)
    ap.add_argument("--checkpoint", action="append", help="label=path/to/latest.model (repeatable)")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--terrains", nargs="*", default=None, help="subset of protocol terrains, e.g. flat hfield")
    ap.add_argument("--modes", nargs="*", default=None, choices=["A", "B"])
    ap.add_argument("--perturbations", nargs="*", default=None, choices=["action_noise", "push"])
    ap.add_argument("--sampling_probs", nargs="*", type=float, default=None, help="same list for both perturbations")
    ap.add_argument("--eval_seeds", nargs="*", type=int, default=None)
    ap.add_argument("--nr_envs", type=int, default=None)
    ap.add_argument("--episode_length_in_seconds", type=int, default=None)
    ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--noise_std_mode", default=None, choices=["absolute", "action_range_fraction"])
    ap.add_argument("--noise_point", default=None, choices=["pre_shield", "post_shield"])
    ap.add_argument("--push_scale", type=float, default=None)
    ap.add_argument("--push_impl", default=None, choices=["env", "env_traced", "eval_loop"])
    ap.add_argument("--background_push_prob", type=float, default=None)
    ap.add_argument("--initial_command_zero_prob", type=float, default=None)
    ap.add_argument("--skip_existing", action="store_true", default=None, help="skip points whose summary exists (requeue)")
    ap.add_argument("--paper_shield", default=None, choices=["train", "cpu_test"], help="paper-code checkpoints only")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--task_index", type=int, default=None, help="run only this job (slurm array index)")
    ap.add_argument("--list_jobs", action="store_true", help="print the job list (for --array=0-<n-1>) and exit")
    ap.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    checkpoints = load_checkpoints(args)
    protocol = load_protocol(args)
    jobs = build_jobs(checkpoints, protocol)
    out_dir = str(Path(args.out_dir).expanduser().resolve())

    if args.list_jobs:
        for i, job in enumerate(jobs):
            print(f"{i}\t{job['checkpoint']['label']}\t{job['terrain_label']}\tmode {job['mode']}")
        print(f"{len(jobs)} jobs -> sbatch --array=0-{len(jobs) - 1}")
        return

    selected = list(enumerate(jobs)) if args.task_index is None else [(args.task_index, jobs[args.task_index])]
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"git_commit": _git_commit(), "protocol": protocol, "checkpoints": checkpoints, "nr_jobs": len(jobs),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"repo {REPO_DIR} commit {manifest['git_commit']} (jobs run with PYTHONPATH={REPO_DIR})", flush=True)
    if not args.dry_run:
        Path(out_dir, "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    failed = []
    for index, job in selected:
        private = None
        if not args.dry_run:
            # PPO.load unpacks into <checkpoint dir>/tmp and deletes it: give every job a private copy so that array
            # tasks sharing a checkpoint directory (e.g. paper_models/ours/*.model) do not collide.
            private = tempfile.mkdtemp(prefix=f"pncbf_eval_{job['checkpoint']['label']}_")
            os.makedirs(os.path.join(private, "models"))
            shutil.copy(job["checkpoint"]["path"], os.path.join(private, "models", "latest.model"))
            job = dict(job, checkpoint=dict(job["checkpoint"], source_path=job["checkpoint"]["path"], path=os.path.join(private, "models", "latest.model")))
        cmd = job_command(job, protocol, out_dir, args.python)
        print(f"[job {index}/{len(jobs)}] {job['checkpoint']['label']} {job['terrain_label']} mode {job['mode']}\n  cd {EXPERIMENTS_DIR} && "
              + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if args.dry_run:
            continue
        start = time.time()
        # rl_x must come from THIS checkout (on IAS, conda env rlx-ncbf has an editable install of another tree)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_DIR)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        result = subprocess.run(cmd, cwd=EXPERIMENTS_DIR, env=env)
        print(f"[job {index}] exit {result.returncode} after {time.time() - start:.0f} s", flush=True)
        shutil.rmtree(private, ignore_errors=True)
        if result.returncode != 0:
            failed.append(index)
    if failed:
        raise SystemExit(f"Failed jobs: {failed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump config/summary/metadata/tags/files of every run in a wandb project (T03).

Run on the IAS login node (CPU only):
  source ~/miniconda3/bin/activate rlx-ncbf
  python dump_wandb.py --out paper_configs
Writes <out>/<group>/<run_id>.json and <out>/runs.tsv.
"""
import argparse, json, os, math

import wandb


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="catherineju-rwth-aachen-university")
    ap.add_argument("--project", default="202601_paper")
    ap.add_argument("--out", default="paper_configs")
    a = ap.parse_args()
    api = wandb.Api(timeout=60)
    rows = []
    for r in api.runs(f"{a.entity}/{a.project}"):
        meta = {}
        try:
            meta = r.metadata or {}
        except Exception as e:  # noqa
            meta = {"_error": str(e)}
        files = []
        try:
            files = [{"name": f.name, "size": f.size, "updated_at": str(getattr(f, "updated_at", ""))}
                     for f in r.files() if not f.name.startswith("media/")]
        except Exception as e:  # noqa
            files = [{"_error": str(e)}]
        summary = {k: v for k, v in dict(r.summary._json_dict).items() if not k.startswith("_wandb")}
        d = dict(id=r.id, name=r.name, group=r.group, job_type=r.job_type, state=r.state, tags=list(r.tags),
                 created_at=str(r.created_at), heartbeat_at=str(getattr(r, "heartbeat_at", "")), url=r.url,
                 commit=getattr(r, "commit", None) or meta.get("git", {}).get("commit"),
                 git=meta.get("git"), program=meta.get("program"), codePath=meta.get("codePath"),
                 args=meta.get("args"), host=meta.get("host"), gpu=meta.get("gpu"),
                 slurm={k: v for k, v in (meta.get("slurm") or {}).items()} if isinstance(meta.get("slurm"), dict) else meta.get("slurm"),
                 runtime_s=summary.get("_runtime"), config=r.config, summary=summary, files=files)
        g = r.group or "_nogroup"
        os.makedirs(os.path.join(a.out, g), exist_ok=True)
        with open(os.path.join(a.out, g, f"{r.id}.json"), "w") as f:
            json.dump(clean(d), f, indent=1, sort_keys=True)
        rows.append((g, r.id, r.name, r.state, ",".join(r.tags), str(d["commit"])[:10], str(r.created_at),
                     f"{(summary.get('_runtime') or 0)/3600:.1f}"))
        print(rows[-1])
    rows.sort()
    with open(os.path.join(a.out, "runs.tsv"), "w") as f:
        f.write("group\tid\tname\tstate\ttags\tcommit\tcreated\truntime_h\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()

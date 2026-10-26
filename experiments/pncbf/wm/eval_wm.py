"""T13 evaluation: k-step error of a trained world model per state group, overall and on near-failure states
(last 50 steps before a termination), on the held-out checkpoints' data. Writes JSON (error vs k = plot data) and a
markdown table.

  python experiments/pncbf/wm/eval_wm.py --wm RUN/wm_final.msgpack --data '<glob>' --heldout_pattern seed_245 \
      --out RUN/eval_k.json
Recorded actions are replayed (u~ from the data); at shocks / DR resampling the recorded x+ is substituted (method A),
so the error measures the dynamics, not the unpredictable shocks. "clean" = windows without any shock inside.
"""
import argparse
import json
import os

import numpy as np

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE


def table(res, split, ks):
    groups = [g for g in next(v for v in res[split].values() if v) if not g.startswith("_")]
    lines = [f"| group | " + " | ".join(f"k={k} model / persist" for k in ks) + " |",
             "|---|" + "---|" * len(ks)]
    for g in groups:
        cells = []
        for k in ks:
            v = res[split].get(k) or res[split].get(str(k))
            cells.append("–" if v is None else f"{v[g]['model']:.4g} / {v[g]['persist']:.3g}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    n = [(res[split].get(k) or {}).get("_n", 0) for k in ks]
    lines.append(f"| windows | " + " | ".join(str(x) for x in n) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--heldout_pattern", default="seed_145/,seed_245/,seed_1342/,seed_1442/")
    p.add_argument("--ks", default="1,2,4,8,16,25,32,50")
    p.add_argument("--table_ks", default="1,8,25,50")
    p.add_argument("--near_failure_window", type=int, default=50)
    p.add_argument("--max_windows", type=int, default=50000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    wm, bundle, meta = WE.load_bundle(args.wm)
    if args.data == "synthetic":
        a = meta["args"]
        held, _ = WD.synthetic_dataset(n_envs=max(a["synthetic_envs"] // 4, 8), T=a["synthetic_T"], seed=a["seed"] + 1000)
    else:
        pats = [q for q in (meta["schema"].get("heldout_patterns") or args.heldout_pattern.split(",")) if q]
        paths = [q for q in WD.find_shards(args.data) if any(t in q for t in pats)]
        held, _ = WD.load_shards(paths, meta["schema"].get("field_map"))
    ks = [int(k) for k in args.ks.split(",")]
    res = WE.evaluate(wm, bundle, held, ks, args.near_failure_window, args.max_windows)
    res["meta"] = {"wm": args.wm, "heldout_pattern": args.heldout_pattern, "rows": int(len(held["done"])),
                   "terminations": int(held["terminated"].sum()), "train_args": meta.get("args")}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    tks = [int(k) for k in args.table_ks.split(",")]
    md = (f"### overall (RMSE, physical units)\n{table(res, 'overall', tks)}\n\n"
          f"### near failure (last {args.near_failure_window} steps before a termination)\n{table(res, 'near_failure', tks)}\n")
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()

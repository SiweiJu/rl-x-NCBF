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


def key_table(res, ks, groups=("imu_height", "roll_pitch")):
    """imu_height (m) and roll_pitch (rad) at k = 8 / 16 / 25, overall and near failure: model / persistence."""
    lines = ["### key: model / persistence RMSE (imu_height in m, roll_pitch in rad)",
             "| group | split | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for g in groups:
        for split in ("overall", "near_failure"):
            cells = []
            for k in ks:
                v = res[split].get(k) or res[split].get(str(k))
                cells.append("–" if v is None or g not in v else f"{v[g]['model']:.4f} / {v[g]['persist']:.4f}")
            lines.append(f"| {g} | {split} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True, help="wm_final.msgpack, or a comma list (data is loaded once)")
    p.add_argument("--data", required=True)
    p.add_argument("--heldout_pattern", default="seed_145/,seed_245/,seed_1342/,seed_1442/")
    p.add_argument("--ks", default="1,2,4,8,16,25,32,50")
    p.add_argument("--table_ks", default="1,8,16,25,50")
    p.add_argument("--near_failure_window", type=int, default=50)
    p.add_argument("--max_windows", type=int, default=50000)
    p.add_argument("--max_envs_per_shard", type=int, default=0)
    p.add_argument("--split", default="heldout", choices=["heldout", "train"], help="train = in-sample (dev checks)")
    p.add_argument("--matmul_dtype", default="", help="forward matmul dtype(s), comma list, e.g. float32,bfloat16 "
                                                      "('' = as trained)")
    p.add_argument("--out", default="", help="output .json for a single --wm / dtype; else <run dir>/<out_name>_<dtype>.json")
    p.add_argument("--out_name", default="eval_heldout")
    args = p.parse_args()
    import dataclasses
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W
    wms = [w for w in args.wm.split(",") if w]
    dtypes = args.matmul_dtype.split(",") if args.matmul_dtype else [""]
    loaded = [WE.load_bundle(w) for w in wms]
    held_full = None
    if args.data == "synthetic":
        a = loaded[0][2]["args"]
        held_full, _ = WD.synthetic_dataset(n_envs=max(a["synthetic_envs"] // 4, 8), T=a["synthetic_T"], seed=a["seed"] + 1000)
    elif WD.find_shard_dirs(args.data):
        held_full, _ = WD.load_wmdata(args.data, args.heldout_pattern.split(","), args.max_envs_per_shard,
                                      which=(args.split,))[args.split]
    else:
        meta0 = loaded[0][2]
        pats = [q for q in (meta0["schema"].get("heldout_patterns") or args.heldout_pattern.split(",")) if q]
        paths = [q for q in WD.find_shards(args.data) if any(t in q for t in pats)]
        held_full, _ = WD.load_shards(paths, meta0["schema"].get("field_map"))
    ks = [int(k) for k in args.ks.split(",")]
    dev_cache = {}
    for wpath, (wm0, bundle, meta) in zip(wms, loaded):
        act = meta["schema"].get("xi_active")
        sig = tuple(act) if act is not None else None
        held = WD.select_xi_dims(held_full, np.asarray(act)) if act is not None else held_full
        if sig not in dev_cache:
            dev_cache.clear()
            dev_cache[sig] = WD.to_device(held)
        for dt in dtypes:
            wm = W.WorldModel(wm0.layout, dataclasses.replace(wm0.cfg, matmul_dtype=dt)) if dt else wm0
            res = WE.evaluate(wm, bundle, held, ks, args.near_failure_window, args.max_windows, dev=dev_cache[sig])
            res["meta"] = {"wm": wpath, "split": args.split, "matmul_dtype": wm.cfg.matmul_dtype,
                           "heldout_pattern": args.heldout_pattern, "rows": int(WD.n_rows(held)),
                           "terminations": int(held["terminated"].sum()), "train_args": meta.get("args")}
            out = args.out if (args.out and len(wms) == 1 and len(dtypes) == 1) else os.path.join(
                os.path.dirname(os.path.abspath(wpath)), f"{args.out_name}_{wm.cfg.matmul_dtype}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=1)
            tks = [int(k) for k in args.table_ks.split(",")]
            key = key_table(res, [k for k in (8, 16, 25) if k in ks])
            md = (key + "\n" + f"### overall (RMSE, physical units)\n{table(res, 'overall', tks)}\n\n"
                  f"### near failure (last {args.near_failure_window} steps before a termination)\n"
                  f"{table(res, 'near_failure', tks)}\n")
            with open(os.path.splitext(out)[0] + ".md", "w") as f:
                f.write(md)
            print(f"===== {wpath} matmul={wm.cfg.matmul_dtype} -> {out}\n{md}", flush=True)


if __name__ == "__main__":
    main()

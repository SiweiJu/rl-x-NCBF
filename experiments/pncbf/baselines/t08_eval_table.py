"""Markdown tables (success +- SE per sweep point) of the T08 eval sweep. Usage: python t08_eval_table.py <OUT_DIR>"""
import json, sys, glob, os
from collections import defaultdict
root = sys.argv[1]
data = defaultdict(dict)
for f in glob.glob(os.path.join(root, "**", "eval_summary.json"), recursive=True):
    s = json.load(open(f))
    data[(s["perturbation"], s["terrain"])][(s["checkpoint_label"], s["sampling_prob"])] = (s["success_rate"], s["success_rate_se"], s["mean_episode_return"])
order = [f"t08_{m}_d{d}_s0" for m in ["ppo_lag_gradient", "ppo_lag_pid", "cpo"] for d in ["0", "0.01", "0.05"]]
for pert in ["action_noise", "push"]:
    for terr in ["flat", "hfield"]:
        tab = data.get((pert, terr), {})
        ps = sorted({p for (_, p) in tab})
        print(f"\n**{pert}, {terr}** (success ± SE, 1024 episodes per point)\n")
        print("| checkpoint | " + " | ".join(f"p={p:g}" for p in ps) + " |")
        print("|---|" + "---|" * len(ps))
        for lab in order:
            cells = [f"{tab[(lab,p)][0]:.3f} ± {tab[(lab,p)][1]:.3f}" if (lab, p) in tab else "–" for p in ps]
            print(f"| {lab.replace('t08_','').replace('_s0','')} | " + " | ".join(cells) + " |")

#!/usr/bin/env python3
"""Download the paper checkpoints (wandb file `latest.model`) into the layout the eval sweep expects:
    <out>/<run_root>/<run_name>.model
run_root is the folder name the paper plot scripts map to legend labels
(misc/rss/plot_line_plot_one_figure.py LEGEND_NAME_MAP): no_safety=PPO-RL, NCBF=H=1, ours, H10, H50,
safe_fall, ncbf (state critic, version (i)).
Run on the IAS login node (CPU only):  python fetch_paper_models.py --out /home/ju/repo/pncbf/paper_models [--only ours]
"""
import argparse
import os

import wandb

ENTITY_PROJECT = "catherineju-rwth-aachen-university/202601_paper"
# run_root -> wandb run ids (see experiments/pncbf/paper_configs/INDEX.md)
RUNS = {
    "no_safety": ["ym84duai", "g3vqg0f8", "p37kpgcs", "2k30xlgc", "wcofn9cv"],  # ncbf-q-no-safe, seeds 1042..1442
    "ours": ["5wpajwko", "zdqj50w0", "wr9r03p2", "lpukb0yz", "27cfbtby"],       # q-safe-H gamma 0.405 H25, seeds 42,43,44,145,245
    "NCBF": ["ksaqa0ay", "e1oolmdg", "5xsqhdr5", "z84dphij", "f3s2gzet"],       # same, H1
    "H10": ["m9j107v4", "7596c1fl", "xxn6cuqz", "rvc4q1sa", "ixnfwras"],
    "H50": ["pa7gv3f0", "cazxxpqv", "fzhmn9ai", "m37s9i3z", "ptkx5nwh"],
    "safe_fall": ["cr77zxxi", "1kgldart", "u8vt9x0t", "b0xzdofb", "6x0xfgyy"],
    "ncbf": ["flclqegy", "6r5bglx3", "hfg5gt43", "b76fjubv", "tlufhugo"],       # version (i), branch ncbf
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    api = wandb.Api(timeout=60)
    for root, ids in RUNS.items():
        if a.only and root not in a.only:
            continue
        for rid in ids:
            r = api.run(f"{ENTITY_PROJECT}/{rid}")
            dst_dir = os.path.join(a.out, root)
            dst = os.path.join(dst_dir, f"{r.name}.model")
            if os.path.exists(dst):
                print("[skip]", dst)
                continue
            tmp = os.path.join(dst_dir, f".tmp_{rid}")
            os.makedirs(tmp, exist_ok=True)
            r.file("latest.model").download(root=tmp, replace=True)
            os.replace(os.path.join(tmp, "latest.model"), dst)
            os.rmdir(tmp)
            print("[ok]", rid, dst)


if __name__ == "__main__":
    main()

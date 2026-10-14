"""Offline check: does a scan-history collision critic trained with obstacle randomization generalize?

Training data comes from the randomized distribution TRAIN (circles, boxes and chains of circles;
constant, turning, stop-and-go and accelerating motion; sensor noise, dropout, latency, jitter and
spurious returns). The critics are then tested on

    in_distribution   new environments from TRAIN
    held_out_shape    L-shaped obstacles only (no L shape in training)
    held_out_motion   random-walk motion only (no random walk in training)
    held_out_size     circles and boxes larger than any in training
    held_out_sensor   more noise, dropout, latency and spurious returns than in training

with the inputs single scan (MLP and convolution), 10-scan convolution (an ensemble of members
trained from different initializations and minibatch orders) and the privileged state, which
describes each obstacle by centre, velocity and bounding radius only. The ensemble's disagreement
on each test set shows whether epistemic uncertainty flags the held-out cases.

Like obstacle_scan_critic_probe.py, the robot does not move and labels come from a fixed data
distribution; this tests perception and generalization of the critic, not the closed loop.

Run from the repository root (CPU, ~35 min):
    JAX_PLATFORMS=cpu PYTHONPATH=. python experiments/ncbf_tools/obstacle_generalization_probe.py --out_dir experiments/runs/generalization_probe
"""
import argparse
import copy
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_scan_critic_probe import (  # noqa: E402
    Critic, ScanConvCritic, auroc, build_samples, classification_metrics, collect, input_fn, labels_for, make_env,
    predict, train_critic)

TRAIN = {
    "shapes": {"weights": {"circle": 1.0, "box": 1.0, "circles": 1.0, "l_shape": 0.0}},
    "motions": {"weights": {"constant": 1.0, "turning": 1.0, "stop_and_go": 1.0, "accelerating": 1.0, "random_walk": 0.0}},
    "scan": {"noise_range": (0.0, 0.04), "dropout_prob_range": (0.0, 0.05), "latency_steps_range": (0, 1),
             "angle_jitter": 0.01, "spurious_prob": 0.002},
}
TEST_SETS = {
    "in_distribution": {},
    "held_out_shape": {"shapes": {"weights": {"circle": 0.0, "box": 0.0, "circles": 0.0, "l_shape": 1.0}}},
    "held_out_motion": {"motions": {"weights": {"constant": 0.0, "turning": 0.0, "stop_and_go": 0.0, "accelerating": 0.0, "random_walk": 1.0}}},
    "held_out_size": {"radius_range": (0.3, 0.45), "shapes": {"box_half_length_range": (0.5, 0.8), "box_half_width_range": (0.3, 0.45)}},
    "held_out_sensor": {"scan": {"noise_range": (0.06, 0.08), "dropout_prob_range": (0.15, 0.2), "latency_steps_range": (2, 3), "spurious_prob": 0.01}},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="experiments/runs/generalization_probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nr_envs", type=int, default=128)
    parser.add_argument("--nr_steps", type=int, default=2500)
    parser.add_argument("--test_nr_envs", type=int, default=64)
    parser.add_argument("--test_nr_steps", type=int, default=1500)
    parser.add_argument("--episode_seconds", type=int, default=20)
    parser.add_argument("--nr_obstacles", type=int, default=2)
    parser.add_argument("--aim_offset", type=float, default=1.0)
    parser.add_argument("--speed_range", type=float, nargs=2, default=(0.5, 2.5))
    parser.add_argument("--nr_rays", type=int, default=64)
    parser.add_argument("--max_range", type=float, default=5.0)
    parser.add_argument("--scan_noise", type=float, default=0.02, help="replaced by the noise range of TRAIN / the test set")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--test_sets", nargs="+", default=list(TEST_SETS), choices=list(TEST_SETS), help="in_distribution must be included")
    parser.add_argument("--report_only", action="store_true", help="redraw results.md and the figure from out_dir/summary.json")
    return parser.parse_args()


def merged(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merged(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_dataset(args, overrides, nr_envs, nr_steps, seed, test_fraction):
    tic = time.time()
    env = make_env(args, overrides, nr_envs=nr_envs)
    data = collect(env, args, nr_steps=nr_steps, seed=seed)
    t_idx, e_idx, split = build_samples(data, args.history, args.horizon, test_fraction, args.val_fraction)
    dataset = {
        "scans": jnp.asarray(data["scan"]),
        "state": jnp.asarray(data["obstacle_state"]),
        "data": data,
        "split": split,
        "t": t_idx,
        "e": e_idx,
        "dt": float(env.dt),
    }
    print(f"  {nr_envs} envs x {nr_steps} steps in {time.time() - tic:.0f}s, collision rate per step {data['collision'].mean():.3f}", flush=True)
    return dataset


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.report_only:
        with open(os.path.join(args.out_dir, "summary.json")) as f:
            results = json.load(f)
        stored = argparse.Namespace(**{**results["args"], "out_dir": args.out_dir})
        write_report(results, stored, f"scanconv_{stored.history}_ensemble")
        return
    started = time.time()

    print("Training distribution", flush=True)
    train_set = load_dataset(args, TRAIN, args.nr_envs, args.nr_steps, args.seed, test_fraction=0.0)
    split = train_set["split"]
    train = tuple(x[split == 0] for x in (train_set["t"], train_set["e"]))
    val = tuple(x[split == 1] for x in (train_set["t"], train_set["e"]))
    train = train + (labels_for(train_set["data"], *train, args.horizon),)
    val = val + (labels_for(train_set["data"], *val, args.horizon),)
    print(f"  {len(train[0])} train / {len(val[0])} validation samples", flush=True)

    test_sets = {}
    for offset, (name, overrides) in enumerate(TEST_SETS.items(), start=1):
        if name not in args.test_sets:
            continue
        print(f"Test set {name}", flush=True)
        dataset = load_dataset(args, merged(TRAIN, overrides), args.test_nr_envs, args.test_nr_steps, args.seed + 100 * offset, test_fraction=1.0)
        dataset["y"] = labels_for(dataset["data"], dataset["t"], dataset["e"], args.horizon)
        test_sets[name] = dataset

    models = [("scan", 1, "scan_1"), ("scanconv", 1, "scanconv_1")]
    models += [("scanconv", args.history, f"scanconv_{args.history}_member{i}") for i in range(args.ensemble)]
    models += [("state_full", 0, "state_full")]
    predictions = {name: {} for name in test_sets}
    training_info = {}
    key = jax.random.PRNGKey(args.seed + 1)
    for kind, history, name in models:
        tic = time.time()
        key, train_key = jax.random.split(key)
        model = ScanConvCritic(args.hidden, history, args.nr_rays) if kind == "scanconv" else Critic(args.hidden)
        make_train_input = input_fn(kind, history, train_set["scans"], train_set["state"], args.nr_obstacles)
        params, info = train_critic(model, make_train_input, train, val, args, train_key)
        training_info[name] = info
        for test_name, dataset in test_sets.items():
            make_input = input_fn(kind, history, dataset["scans"], dataset["state"], args.nr_obstacles)
            predictions[test_name][name], _ = predict(model, params, make_input, dataset["t"], dataset["e"])
        print(f"{name:22s} val log loss {info['val_log_loss']:.4f} ({time.time() - tic:.0f}s)", flush=True)

    ensemble_name = f"scanconv_{args.history}_ensemble"
    members = [f"scanconv_{args.history}_member{i}" for i in range(args.ensemble)]
    results = {"args": vars(args), "train": TRAIN, "test_sets": TEST_SETS, "training": training_info, "metrics": {}}
    reference_std = None
    for test_name, dataset in test_sets.items():
        y = dataset["y"]
        stacked = np.stack([predictions[test_name][m] for m in members])
        predictions[test_name][ensemble_name] = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        entry = {"nr_samples": int(len(y)), "unsafe_rate": float(1.0 - y.mean()), "models": {},
                 "ensemble_std_mean": float(std.mean()), "ensemble_std_mean_unsafe": float(std[y < 0.5].mean())}
        for name, p in predictions[test_name].items():
            entry["models"][name] = classification_metrics(p, y)
        if test_name == "in_distribution":
            reference_std = std
        else:
            # How well does disagreement separate this test set from in-distribution samples?
            scores = np.concatenate([reference_std, std])
            is_held_out = np.concatenate([np.zeros_like(reference_std), np.ones_like(std)])
            entry["ensemble_std_auroc_vs_in_distribution"] = auroc(scores, is_held_out)
        results["metrics"][test_name] = entry
        print(f"{test_name:16s} unsafe {entry['unsafe_rate']:.3f}  " + "  ".join(
            f"{n} {entry['models'][n]['auroc']:.3f}" for n in ("scan_1", "scanconv_1", ensemble_name, "state_full"))
            + f"  std {entry['ensemble_std_mean']:.4f}", flush=True)

    results["runtime_seconds"] = time.time() - started
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_report(results, args, ensemble_name)
    print(f"Wrote {args.out_dir}/summary.json, results.md and generalization.png in {time.time() - started:.0f}s", flush=True)


def write_report(results, args, ensemble_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette (light): categorical slots 1-4, surface and text inks.
    surface, ink, ink_secondary, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    series = [("scan_1", "single scan, MLP", "#2a78d6"), ("scanconv_1", "single scan, conv", "#eb6834"),
              (ensemble_name, f"{args.history} scans, conv (ensemble mean)", "#1baf7a"), ("state_full", "privileged state (bounding circle)", "#eda100")]
    test_names = list(results["metrics"])
    labels = [name.replace("held_out_", "held-out\n").replace("in_distribution", "in\ndistribution") for name in test_names]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink_secondary, "axes.labelcolor": ink_secondary,
                         "xtick.color": ink_secondary, "ytick.color": ink_secondary, "text.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]}, facecolor=surface)
    # Dots rather than bars: AUROC is read against chance (0.5), not against zero.
    ax = axes[0]
    offset = 0.6 / len(series)
    x = np.arange(len(test_names))
    for i, (name, label, color) in enumerate(series):
        values = [results["metrics"][t]["models"][name]["auroc"] for t in test_names]
        ax.plot(x + (i - (len(series) - 1) / 2) * offset, values, linestyle="none", marker="o", markersize=7,
                markeredgecolor=surface, markeredgewidth=1.5, color=color, label=label, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(test_names) - 0.5)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("AUROC, collision within 0.5 s")
    ax.set_title("Critic accuracy on each test set", color=ink, loc="left")
    handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    stds = [results["metrics"][t]["ensemble_std_mean"] for t in test_names]
    ax.bar(x, stds, 0.6, color="#2a78d6", zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean std of P(safe) across members")
    ax.set_title(f"Ensemble disagreement ({args.history} scans, conv)", color=ink, loc="left")
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(series), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(args.out_dir, "generalization.png"), dpi=150, facecolor=surface)
    plt.close(fig)

    names = [name for name, _, _ in series]
    lines = ["# Obstacle randomization: held-out generalization of the collision critic", "",
             f"Static Go2, {args.nr_obstacles} obstacles, speeds {args.speed_range[0]}-{args.speed_range[1]} m/s, "
             f"aim offset ±{args.aim_offset} m, {args.nr_rays} rays, H = {args.horizon} steps. Trained on "
             f"{args.nr_envs} envs x {args.nr_steps} steps of the TRAIN distribution with early stopping; each test set "
             f"has {args.test_nr_envs} new envs x {args.test_nr_steps} steps. Overrides are in summary.json.", "",
             "| test set | unsafe rate | " + " | ".join(names) + " | ensemble std | std AUROC vs in-distribution |",
             "|---|---|" + "---|" * len(names) + "---|---|"]
    for test_name, entry in results["metrics"].items():
        cells = [f"{entry['models'][n]['auroc']:.3f} / {entry['models'][n]['average_precision']:.3f}" for n in names]
        separation = entry.get("ensemble_std_auroc_vs_in_distribution")
        lines.append(f"| {test_name} | {entry['unsafe_rate']:.3f} | " + " | ".join(cells)
                     + f" | {entry['ensemble_std_mean']:.4f} | {'–' if separation is None else f'{separation:.3f}'} |")
    lines += ["", "Cells are AUROC / average precision for \"unsafe\". The last column is the AUROC of ensemble",
              "disagreement for telling that test set's samples from in-distribution samples (0.5 = no signal).", "",
              "Single members of the ensemble:", "", "| test set | " + " | ".join(f"member {i}" for i in range(args.ensemble)) + " |",
              "|---|" + "---|" * args.ensemble]
    for test_name, entry in results["metrics"].items():
        lines.append(f"| {test_name} | " + " | ".join(
            f"{entry['models'][f'scanconv_{args.history}_member{i}']['auroc']:.3f}" for i in range(args.ensemble)) + " |")
    lines += ["", "Figure: `generalization.png`."]
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

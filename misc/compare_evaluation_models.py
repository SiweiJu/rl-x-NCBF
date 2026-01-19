#!/usr/bin/env python3
"""
Compare rollout evaluation PKLs across multiple models (inputs are PKL file paths).

Legend names are overridden via MODEL_LABELS below.

Usage:
  python compare_eval_pkls_no_safety.py file1.pkl file2.pkl ...
  python compare_eval_pkls_no_safety.py --glob "/path/**/rollout/eval_safety_False_ratio_*.pkl"
"""

import argparse
import glob
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


FILE_RE = re.compile(r"eval_safety_False_ratio_(?P<ratio>[0-9mp]+)\.pkl$")

# ---- Custom legend labels (folder name -> legend label) ----
MODEL_LABELS = {
    "long_run_no_safety": "Without safety layer",
    "plane_terrain_sweep_policy_loss_wd_0_lip_1e-5_mb_64_lr_0.001_buf_50_20_eta_.5_gamma_0_plc_0.0001": "With safety layer",
}


def tok_to_float(tok: str) -> float:
    sign = -1.0 if tok.startswith("m") else 1.0
    core = tok[1:] if tok.startswith("m") else tok
    core = core.replace("p", ".")
    return sign * float(core)


def parse_ratio_from_filename(p: Path) -> Optional[float]:
    m = FILE_RE.search(p.name)
    if not m:
        return None
    return tok_to_float(m.group("ratio"))


def infer_model_name_from_path(pkl_path: Path) -> str:
    parts = pkl_path.parts
    if "rollout" in parts:
        idx = parts.index("rollout")
        if idx > 0:
            return parts[idx - 1]
    return pkl_path.parent.name


def display_model_name(model_folder_name: str) -> str:
    return MODEL_LABELS.get(model_folder_name, model_folder_name)


def load_rollouts(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} did not contain a list. Got: {type(obj)}")
    return obj


def episode_length(ep: Dict[str, Any]) -> int:
    for k in ("dones", "rewards", "actions", "raw_action", "safe_action", "predictions"):
        if k in ep:
            try:
                return int(np.asarray(ep[k]).shape[0])
            except Exception:
                pass
    for v in ep.values():
        try:
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                return int(arr.shape[0])
        except Exception:
            continue
    return 0


def episode_return(ep: Dict[str, Any]) -> Optional[float]:
    if "ret" in ep:
        try:
            arr = np.asarray(ep["ret"]).reshape(-1)
            return float(arr[0])
        except Exception:
            pass
    if "episode_return" in ep:
        try:
            return float(ep["episode_return"])
        except Exception:
            pass
    if "rewards" in ep:
        try:
            return float(np.sum(np.asarray(ep["rewards"], dtype=float)))
        except Exception:
            pass
    return None


def summarize_file(path: Path, threshold: int) -> Dict[str, Any]:
    rollouts = load_rollouts(path)
    lengths = np.array([episode_length(ep) for ep in rollouts], dtype=int)

    finished = int(np.sum(lengths >= threshold))
    early = int(np.sum(lengths < threshold))

    rets = []
    missing_ret = 0
    for ep in rollouts:
        r = episode_return(ep)
        if r is None:
            missing_ret += 1
        else:
            rets.append(r)
    rets_arr = np.array(rets, dtype=float) if rets else np.array([], dtype=float)

    model_folder = infer_model_name_from_path(path)

    return {
        "file": str(path),
        "model_folder": model_folder,
        "model": display_model_name(model_folder),
        "ratio": parse_ratio_from_filename(path),
        "n_episodes": len(rollouts),
        "finished_episodes(>=thr)": finished,
        "early_terminated(<thr)": early,
        "threshold": threshold,
        "avg_return": float(np.mean(rets_arr)) if rets_arr.size else float("nan"),
        "std_return": float(np.std(rets_arr)) if rets_arr.size else float("nan"),
        "missing_ret": missing_ret,
        "avg_ep_len": float(np.mean(lengths)) if lengths.size else float("nan"),
        "min_ep_len": int(np.min(lengths)) if lengths.size else 0,
        "max_ep_len": int(np.max(lengths)) if lengths.size else 0,
    }


def format_table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "model", "ratio",
        "n_episodes",
        "finished_episodes(>=thr)",
        "early_terminated(<thr)",
        "avg_return", "std_return",
        "missing_ret",
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            return f"{v:.4f}"
        return str(v)

    widths = {c: max(len(c), max((len(fmt(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)

    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def plot_summary(rows: List[Dict[str, Any]], plot_out: Optional[str], show: bool) -> None:
    rows_use = [
        r for r in rows
        if r.get("ratio") is not None
        and r.get("model") is not None
        and r.get("finished_episodes(>=thr)") is not None
        and not np.isnan(r.get("avg_return", np.nan))
    ]
    if not rows_use:
        print("[WARN] No rows available for plotting.")
        return

    models = sorted(set(r["model"] for r in rows_use))

    fig, ax_fin = plt.subplots(figsize=(10, 5))
    ax_ret = ax_fin.twinx()

    for model in models:
        grp = [r for r in rows_use if r["model"] == model]
        grp = sorted(grp, key=lambda r: float(r["ratio"]))

        x = np.array([r["ratio"] for r in grp], dtype=float)
        y_fin = np.array([r["finished_episodes(>=thr)"] for r in grp], dtype=float)
        y_ret = np.array([r["avg_return"] for r in grp], dtype=float)

        line_fin, = ax_fin.plot(x, y_fin, marker="o", linewidth=2, label=f"{model} (finished)")
        ax_ret.plot(
            x, y_ret,
            marker="x", linestyle="--", linewidth=2,
            color=line_fin.get_color(),
            label=f"{model} (ret)"
        )

    ax_fin.set_xlabel("Action noise ratio")
    ax_fin.set_ylabel("Number of finished episodes (len >= threshold)")
    ax_ret.set_ylabel("Average return")
    ax_fin.grid(True, linestyle="--", alpha=0.3)

    h1, l1 = ax_fin.get_legend_handles_labels()
    h2, l2 = ax_ret.get_legend_handles_labels()
    ax_fin.legend(h1 + h2, l1 + l2, loc="best", fontsize=9, ncol=2)

    fig.tight_layout()

    if plot_out:
        outp = Path(plot_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=200)
        print(f"Wrote plot: {outp}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl_files", nargs="*", help="Paths to eval_safety_False_ratio_*.pkl files.")
    ap.add_argument("--glob", dest="glob_pats", action="append", default=[],
                    help='Glob for .pkl files; may be repeated. Example: "/path/**/rollout/eval_safety_False_ratio_*.pkl"')
    ap.add_argument("--threshold", type=int, default=250,
                    help="Finished episode threshold: length >= threshold counts as finished.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional path to write a CSV summary.")
    ap.add_argument("--plot_out", type=str, default=None,
                    help="Optional path to save the plot (e.g., summary.png).")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not display the plot window (useful on headless nodes).")
    args = ap.parse_args()

    files: List[str] = []
    for pat in args.glob_pats:
        files.extend(sorted(glob.glob(pat, recursive=True)))
    files.extend(args.pkl_files)

    # de-dup while preserving order
    seen = set()
    uniq_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq_files.append(f)

    if not uniq_files:
        raise SystemExit("No input files provided. Use positional args and/or --glob.")

    rows: List[Dict[str, Any]] = []
    for f in uniq_files:
        p = Path(f)
        if not p.is_file():
            print(f"[WARN] Missing: {p}")
            continue
        ratio = parse_ratio_from_filename(p)
        if ratio is None:
            print(f"[WARN] Filename did not match expected pattern: {p.name}")
            continue
        try:
            rows.append(summarize_file(p, args.threshold))
        except Exception as e:
            print(f"[ERROR] Failed on {p}: {e}")
            continue

    # sort for printing: model label then ratio
    rows = sorted(rows, key=lambda r: (r["model"], float(r["ratio"])))

    print(format_table(rows))

    if args.out_csv and rows:
        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote CSV: {out_path}")

    plot_summary(rows, plot_out=args.plot_out, show=(not args.no_show))


if __name__ == "__main__":
    main()

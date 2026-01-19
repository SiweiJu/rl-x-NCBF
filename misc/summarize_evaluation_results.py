#!/usr/bin/env python3
"""
Summarize rollout evaluation PKLs and plot results.

For each input .pkl file (one experiment setting), this script reports:
  - n_episodes
  - finished_episodes: number of episodes with length >= threshold
  - early_terminated: number of episodes with length < threshold
  - avg_return (episode["ret"])
  - std_return

Then it plots:
  - x-axis: action noise ratio
  - left y-axis: number of finished episodes
  - right y-axis: avg return
  - safety_layer=False shown in grey
  - safety_layer=True colored from red->green depending on gamma_c

Usage:
  python summarize_eval_pkls.py /path/a.pkl /path/b.pkl
  python summarize_eval_pkls.py --glob "/path/to/rollout/eval_safety_*.pkl"
  python summarize_eval_pkls.py --glob "..." --out_csv results.csv --plot_out summary.png
"""

import argparse
import glob
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

SETTING_RE = re.compile(
    r"eval_safety_(?P<safety>True|False)_ratio_(?P<ratio>[0-9mp]+)"
    r"(?:_gamma_(?P<gamma>[0-9mp]+))?"
    r"\.pkl$"
)


def parse_setting_from_filename(p: Path) -> Dict[str, Any]:
    m = SETTING_RE.search(p.name)
    if not m:
        return {"safety_layer": None, "ratio": None, "gamma_c": None, "name": p.stem}

    def tok_to_float(tok: Optional[str]) -> Optional[float]:
        if tok is None:
            return None
        sign = -1.0 if tok.startswith("m") else 1.0
        core = tok[1:] if tok.startswith("m") else tok
        core = core.replace("p", ".")
        return sign * float(core)

    return {
        "safety_layer": (m.group("safety") == "True"),
        "ratio": tok_to_float(m.group("ratio")),
        "gamma_c": tok_to_float(m.group("gamma")),
        "name": p.stem,
    }


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
            return ep["ret"][0]
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
    setting = parse_setting_from_filename(path)
    rollouts = load_rollouts(path)

    lengths = np.array([episode_length(ep) for ep in rollouts], dtype=int)
    early = int(np.sum(lengths < threshold))
    finished = int(np.sum(lengths >= threshold))

    rets = []
    missing_ret = 0
    for ep in rollouts:
        r = episode_return(ep)
        if r is None:
            missing_ret += 1
        else:
            rets.append(r)
    rets_arr = np.array(rets, dtype=float) if rets else np.array([], dtype=float)

    return {
        "file": str(path),
        "name": setting["name"],
        "safety_layer": setting["safety_layer"],
        "ratio": setting["ratio"],
        "gamma_c": setting["gamma_c"],
        "n_episodes": len(rollouts),
        "finished_episodes(>=thr)": finished,
        "early_terminated(<thr)": early,
        "threshold": threshold,
        "avg_return": float(np.mean(rets_arr)) if rets_arr.size else float("nan"),
        "std_return": float(np.std(rets_arr)) if rets_arr.size else float("nan"),
        "missing_ret": missing_ret,
        # keep these for debugging/optional post-processing
        "avg_ep_len": float(np.mean(lengths)) if lengths.size else float("nan"),
        "min_ep_len": int(np.min(lengths)) if lengths.size else 0,
        "max_ep_len": int(np.max(lengths)) if lengths.size else 0,
    }


def format_table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "safety_layer", "ratio", "gamma_c",
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


def _sorted_unique(xs: List[float]) -> List[float]:
    return sorted(set(float(x) for x in xs))


def plot_summary(rows: List[Dict[str, Any]], plot_out: Optional[str] = None, show: bool = True) -> None:
    rows_use = [
        r for r in rows
        if r.get("ratio") is not None
        and r.get("finished_episodes(>=thr)") is not None
        and not np.isnan(r.get("avg_return", np.nan))
        and r.get("safety_layer") is not None
    ]
    if not rows_use:
        print("[WARN] No rows with (ratio, finished_episodes, avg_return, safety_layer) available for plotting.")
        return

    fig, ax_fin = plt.subplots(figsize=(10, 5))
    ax_ret = ax_fin.twinx()

    # Safety layer OFF: single grey series across ratios
    off = [r for r in rows_use if r["safety_layer"] is False]
    if off:
        off = sorted(off, key=lambda r: float(r["ratio"]))
        x = np.array([r["ratio"] for r in off], dtype=float)
        y_fin = np.array([r["finished_episodes(>=thr)"] for r in off], dtype=float)
        y_ret = np.array([r["avg_return"] for r in off], dtype=float)
        ax_fin.plot(x, y_fin, color="0.5", marker="o", linewidth=2, label="No safety layer (finished)")
        ax_ret.plot(x, y_ret, color="0.5", marker="x", linestyle="--", linewidth=2, label="No safety layer (ret)")

    # Safety layer ON: multiple series by gamma, colored red->green
    on = [r for r in rows_use if r["safety_layer"] is True]
    gammas = _sorted_unique([r["gamma_c"] for r in on if r.get("gamma_c") is not None])
    if on and gammas:
        norm = mcolors.Normalize(vmin=min(gammas), vmax=max(gammas))
        cmap = cm.get_cmap("RdYlGn")  # red -> yellow -> green

        for g in gammas:
            grp = [r for r in on if r.get("gamma_c") == g]
            grp = sorted(grp, key=lambda r: float(r["ratio"]))
            x = np.array([r["ratio"] for r in grp], dtype=float)
            y_fin = np.array([r["finished_episodes(>=thr)"] for r in grp], dtype=float)
            y_ret = np.array([r["avg_return"] for r in grp], dtype=float)

            col = cmap(norm(g))
            ax_fin.plot(x, y_fin, color=col, marker="o", linewidth=2, label=f"Safety layer γ={g:g} (finished)")
            ax_ret.plot(x, y_ret, color=col, marker="x", linestyle="--", linewidth=2, label=f"Safety layer γ={g:g} (ret)")

    ax_fin.set_xlabel("Action noise ratio")
    ax_fin.set_ylabel("Number of finished episodes (len >= threshold)")
    ax_ret.set_ylabel("Average return")
    ax_fin.grid(True, linestyle="--", alpha=0.3)

    # Combined legend
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
    ap.add_argument("pkl_files", nargs="*", help="Paths to .pkl files.")
    ap.add_argument("--glob", dest="glob_pat", default=None,
                    help='Glob for .pkl files, e.g. "/path/to/rollout/eval_safety_*.pkl"')
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
    if args.glob_pat:
        files.extend(sorted(glob.glob(args.glob_pat)))
    files.extend(args.pkl_files)

    # de-dup while preserving order
    seen = set()
    uniq_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq_files.append(f)

    if not uniq_files:
        raise SystemExit("No input files provided. Use positional args or --glob.")

    rows: List[Dict[str, Any]] = []
    for f in uniq_files:
        p = Path(f)
        if not p.is_file():
            print(f"[WARN] Missing: {p}")
            continue
        try:
            rows.append(summarize_file(p, args.threshold))
        except Exception as e:
            print(f"[ERROR] Failed on {p}: {e}")
            continue

    # sort: safety False first, then ratio, then gamma
    def sort_key(r: Dict[str, Any]) -> Tuple[int, float, float]:
        safety = r["safety_layer"]
        safety_rank = 2 if safety is None else (1 if safety else 0)
        ratio = r["ratio"]
        gamma = r["gamma_c"]
        return (
            safety_rank,
            float(ratio) if ratio is not None else 1e9,
            float(gamma) if gamma is not None else 1e9,
        )

    rows = sorted(rows, key=sort_key)

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

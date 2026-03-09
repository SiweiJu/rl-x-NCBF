#!/usr/bin/env python3
"""
Filter a rollout-summary CSV to ONLY include rows whose PKL filepath contains
"eval_safety_False", then WRITE/APPEND them into a target CSV.

Behavior:
- Reads input CSV (must contain a "file" column).
- Filters rows where df["file"] contains args.pattern (default: "eval_safety_False").
- If output CSV exists:
    - Appends only NEW rows (dedup by a stable key; default uses "file" if present,
      otherwise uses a composite of common columns).
- If output CSV doesn't exist:
    - Writes all filtered rows.

Usage:
  python filter_csv_eval_safety_false_append.py /path/to/summary.csv
  python filter_csv_eval_safety_false_append.py /path/to/summary.csv --out_csv /path/to/false_only.csv
  python filter_csv_eval_safety_false_append.py /path/to/summary.csv --key_cols file
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def pick_default_key_cols(df: pd.DataFrame) -> List[str]:
    # Prefer 'file' if available; it's usually unique per PKL.
    if "file" in df.columns:
        return ["file"]

    # Fallback: a composite key over common fields (only if they exist).
    candidates = [
        "experiment", "model",
        "safety_layer", "gamma", "sampling_prob", "seed",
        "name",
    ]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        # last resort: use all columns (can be heavy but will work)
        cols = list(df.columns)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="/home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/no_safety/eval_summary_20260129_193023.csv", help="Input CSV (summary from PKLs).")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="/home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/H50.csv",
        help="Output CSV path. Default: <input>_eval_safety_False.csv",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="eval_safety_False",
        help="Substring to filter on in the 'file' column.",
    )
    ap.add_argument(
        "--key_cols",
        nargs="*",
        default=None,
        help="Columns used to deduplicate when appending. Default: ['file'] if present, else a reasonable composite.",
    )
    args = ap.parse_args()

    in_path = Path(args.csv_path).expanduser().resolve()
    if not in_path.is_file():
        raise SystemExit(f"CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    if "file" not in df.columns:
        raise SystemExit(
            "CSV must contain a 'file' column with the PKL path used to produce the row.\n"
            f"Columns found: {list(df.columns)}"
        )

    # Filter by pattern
    before = len(df)
    df_f = df[df["file"].astype(str).str.contains(args.pattern, na=False)].copy()
    after = len(df_f)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
    else:
        out_path = in_path.with_name(in_path.stem + f"_{args.pattern}.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if after == 0:
        print(f"[WARN] No rows matched pattern='{args.pattern}'. Nothing to write/append.")
        print(f"[INFO] Input rows: {before}")
        print(f"[INFO] Output target: {out_path}")
        return

    # Determine dedup key columns
    key_cols = args.key_cols if args.key_cols else pick_default_key_cols(df_f)
    missing = [c for c in key_cols if c not in df_f.columns]
    if missing:
        raise SystemExit(f"key_cols missing in input CSV: {missing}\nColumns found: {list(df_f.columns)}")

    # Append-or-write
    wrote_new_file = False
    appended = 0
    kept_existing = 0

    if out_path.is_file():
        df_out = pd.read_csv(out_path)

        # Align columns: union, preserve output order first
        all_cols = list(df_out.columns)
        for c in df_f.columns:
            if c not in all_cols:
                all_cols.append(c)

        df_out = df_out.reindex(columns=all_cols)
        df_f2 = df_f.reindex(columns=all_cols)

        # Dedup: keep rows from df_f2 that aren't already in df_out by key
        # Build comparable key tuples (stringify to avoid dtype mismatch)
        def make_key(df0: pd.DataFrame) -> pd.Series:
            return df0[key_cols].astype(str).agg("||".join, axis=1)

        out_keys = set(make_key(df_out).tolist())
        f_keys = make_key(df_f2).tolist()

        is_new = [k not in out_keys for k in f_keys]
        df_new = df_f2[is_new].copy()

        appended = len(df_new)
        kept_existing = len(df_out)

        if appended > 0:
            df_merged = pd.concat([df_out, df_new], ignore_index=True)
            df_merged.to_csv(out_path, index=False)
        else:
            # Nothing new; keep file unchanged
            df_merged = df_out

    else:
        df_f.to_csv(out_path, index=False)
        wrote_new_file = True
        appended = len(df_f)

    # Reporting
    print(f"[OK] Input rows:     {before}")
    print(f"[OK] Matched rows:  {after}  (pattern='{args.pattern}')")
    print(f"[OK] Key cols:      {key_cols}")
    if wrote_new_file:
        print(f"[OK] Wrote new:     {out_path}  (+{appended} rows)")
    else:
        print(f"[OK] Target exists: {out_path}")
        print(f"[OK] Existing rows: {kept_existing}")
        print(f"[OK] Appended rows: {appended}")

    # Quick breakdown
    df_final = pd.read_csv(out_path)
    if len(df_final) > 0:
        for col in ["experiment", "model", "gamma", "sampling_prob"]:
            if col in df_final.columns:
                print(f"  - unique {col}: {df_final[col].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
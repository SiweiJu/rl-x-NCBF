#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List, Optional

import wandb


def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^\w.\-+=@(){}\[\], ]+", "_", s)
    s = s.replace(" ", "_")
    return s[:200] if s else "run"


def parse_tags(tag_args: Optional[List[str]]) -> List[str]:
    if not tag_args:
        return []
    out: List[str] = []
    for t in tag_args:
        if not t:
            continue
        out.extend([p.strip() for p in str(t).split(",") if p.strip()])
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description='Download a file (default: latest.model) from runs matching tag filters; save as <run_name>.model'
    )
    ap.add_argument("--entity", default="catherineju-rwth-aachen-university", help="W&B entity (user or team)")
    ap.add_argument("--project", default="202601_paper", help="W&B project name")
    ap.add_argument("--out", default="/home/siwei/Downloads/RSS/models/ours", help="Output directory")
    ap.add_argument("--state", default=None, help='Optional run state filter, e.g. "finished"')
    ap.add_argument("--run-name-regex", default=None, help="Optional regex to filter runs by name/display_name")
    ap.add_argument("--tag", action="append", default="ours",
                    help='Tag filter. Repeatable: --tag foo --tag bar OR comma list: --tag "foo,bar"')
    ap.add_argument("--tag-mode", choices=["any", "all"], default="all",
                    help='If "all": run must contain all tags. If "any": run must contain at least one tag.')
    ap.add_argument("--replace", action="store_true", help="Overwrite existing files")
    ap.add_argument("--file", default="latest.model", help='Which file to download from each run (default: latest.model)')
    ap.add_argument("--debug", action="store_true", help="Print debug info about returned runs/tags")
    args = ap.parse_args()

    tags = parse_tags(args.tag)

    os.makedirs(args.out, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    # --- SERVER-SIDE FILTERS (important) ---
    filters = {}
    if args.state:
        filters["state"] = args.state

    # W&B public API filters are Mongo-ish. tags filtering works well here.
    if tags:
        if args.tag_mode == "all":
            filters["tags"] = {"$all": tags}
        else:
            filters["tags"] = {"$in": tags}

    run_re = re.compile(args.run_name_regex) if args.run_name_regex else None

    runs = api.runs(path, filters=filters)

    n_total = 0
    n_after_name = 0
    n_found = 0
    n_saved = 0

    # Print a few runs for sanity
    debug_print_left = 10 if args.debug else 0

    for run in runs:
        n_total += 1

        run_name = (getattr(run, "name", None) or "")
        disp_name = (getattr(run, "display_name", None) or "")
        chosen_name = disp_name or run_name or run.id

        run_tags = list(getattr(run, "tags", None) or [])

        if debug_print_left > 0:
            print(f"[DBG ] run={run.id} name={chosen_name} tags={run_tags}")
            debug_print_left -= 1

        if run_re and not (run_re.search(run_name) or run_re.search(disp_name)):
            continue

        n_after_name += 1

        base = safe_filename(chosen_name)
        dst_file = os.path.join(args.out, f"{base}.model")

        if os.path.exists(dst_file) and not args.replace:
            dst_file = os.path.join(args.out, f"{base}__{run.id}.model")
            if os.path.exists(dst_file) and not args.replace:
                print(f"[SKIP] exists: {dst_file}")
                continue

        try:
            f = run.file(args.file)
            tmp_dir = os.path.join(args.out, f".tmp_{run.id}")
            os.makedirs(tmp_dir, exist_ok=True)

            print(f"[GET ] run={run.id} name={chosen_name} -> {args.file}")
            rel_path = f.download(root=tmp_dir, replace=True).name
            src_file = os.path.join(tmp_dir, rel_path)

            if not os.path.exists(src_file):
                src_file = rel_path if os.path.isabs(rel_path) else src_file

            if not os.path.exists(src_file):
                print(f"[WARN] download returned but file not found for run={run.id}")
                continue

            if os.path.exists(dst_file):
                os.remove(dst_file)
            os.replace(src_file, dst_file)

            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

            print(f"[OK  ] saved: {dst_file}")
            n_found += 1
            n_saved += 1

        except Exception as e:
            print(f"[MISS] run={run.id} name={chosen_name}: {e}")

    print("\nSummary")
    print(f"  - Query: {path}")
    print(f"  - Server tag filter: {tags} (mode={args.tag_mode})")
    if args.state:
        print(f"  - State filter: {args.state}")
    if args.run_name_regex:
        print(f"  - Name regex: {args.run_name_regex}")
    print(f"  - Runs returned by API (after server filters): {n_total}")
    print(f"  - Runs after name-regex filter:              {n_after_name}")
    print(f"  - Found '{args.file}':                        {n_found}")
    print(f"  - Saved:                                     {n_saved}")
    print(f"  - Out:                                       {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
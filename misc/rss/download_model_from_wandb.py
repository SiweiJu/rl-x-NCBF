#!/usr/bin/env python3
import argparse
import os
import re
import sys
import wandb


def safe_filename(s: str) -> str:
    s = (s or "").strip()
    # Replace path separators and bad chars
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^\w.\-+=@(){}\[\], ]+", "_", s)
    s = s.replace(" ", "_")
    # Avoid empty names
    return s[:200] if s else "run"


def main():
    ap = argparse.ArgumentParser(
        description='Download "latest.model" from all runs and save as <run_name>.model'
    )
    ap.add_argument("--entity", default="catherineju-rwth-aachen-university", help="W&B entity (user or team)")
    ap.add_argument("--project", default="202601_paper", help="W&B project name")
    ap.add_argument("--out", default="/home/siwei/Downloads/RSS/models/safefall", help="Output directory")
    ap.add_argument("--state", default=None, help='Optional run state filter, e.g. "finished"')
    ap.add_argument("--run-name-regex", default="safe_fall_safety", help="Optional regex to filter runs by name/display_name")
    ap.add_argument("--replace", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    filters = {}
    if args.state:
        filters["state"] = args.state

    run_re = re.compile(args.run_name_regex) if args.run_name_regex else None

    runs = api.runs(path, filters=filters)

    n_runs = 0
    n_found = 0
    n_saved = 0

    for run in runs:
        run_name = (getattr(run, "name", None) or "")
        disp_name = (getattr(run, "display_name", None) or "")
        chosen_name = disp_name or run_name or run.id

        if run_re and not (run_re.search(run_name) or run_re.search(disp_name)):
            continue

        n_runs += 1

        base = safe_filename(chosen_name)
        dst_file = os.path.join(args.out, f"{base}.model")

        # If name collides and not replacing, use run id suffix for uniqueness
        if os.path.exists(dst_file) and not args.replace:
            dst_file = os.path.join(args.out, f"{base}__{run.id}.model")
            if os.path.exists(dst_file) and not args.replace:
                print(f"[SKIP] exists: {dst_file}")
                continue

        try:
            f = run.file("latest.model")
            tmp_dir = os.path.join(args.out, f".tmp_{run.id}")
            os.makedirs(tmp_dir, exist_ok=True)

            print(f"[GET ] run={run.id} name={chosen_name} -> latest.model")
            rel_path = f.download(root=tmp_dir, replace=True).name
            src_file = os.path.join(tmp_dir, rel_path)

            if not os.path.exists(src_file):
                # rare fallback if .name is absolute
                src_file = rel_path if os.path.isabs(rel_path) else src_file

            if not os.path.exists(src_file):
                print(f"[WARN] download returned but file not found for run={run.id}")
                continue

            if os.path.exists(dst_file):
                os.remove(dst_file)
            os.replace(src_file, dst_file)

            # Cleanup tmp dir if empty
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

            print(f"[OK  ] saved: {dst_file}")
            n_found += 1
            n_saved += 1

        except Exception as e:
            print(f"[MISS] run={run.id} name={chosen_name}: {e}")

    print(f"\nDone. Considered {n_runs} runs; found {n_found} latest.model files; saved {n_saved}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

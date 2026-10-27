import glob, json, os, sys, numpy as np
root = sys.argv[1]
schema = json.load(open(os.path.join(root, "schema.json")))
fields = schema["fields"]
total = 0; bad = []; n = 0
for meta_path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "meta.json"))):
    d = os.path.dirname(meta_path); n += 1
    m = json.load(open(meta_path))
    T, N = m["nr_steps"], m["nr_envs"]
    probs = []
    if not os.path.exists(os.path.join(d, "summary.json")): probs.append("no summary")
    size = 0
    for name, spec in fields.items():
        p = os.path.join(d, name + ".npy")
        if not os.path.exists(p):
            probs.append(f"missing {name}"); continue
        st = os.path.getsize(p); size += st
        try:
            a = np.load(p, mmap_mode="r")
        except Exception as e:
            probs.append(f"load {name}: {e}"); continue
        expect = a.dtype.itemsize * int(np.prod(a.shape))
        if st < expect: probs.append(f"truncated {name}")
        if spec["shape"][:2] == ["T", "N"] and a.shape[:2] != (T, N): probs.append(f"shape {name} {a.shape}")
        if name == "oracle_u0" and a.shape[:2] != (T // 10, min(256, N)): probs.append(f"oracle shape {a.shape}")
    # full read of a few arrays to catch I/O garbage
    try:
        t = np.load(os.path.join(d, "terminated.npy")); mg = np.load(os.path.join(d, "margin_next.npy"))
        if not np.array_equal(mg < 0, t): probs.append("margin mismatch")
        x = np.load(os.path.join(d, "x_plus.npy"))
        if not np.isfinite(x).all(): probs.append("nonfinite x_plus")
        xi = np.load(os.path.join(d, "xi_table.npy")); idx = np.load(os.path.join(d, "xi_idx.npy"))
        if idx.max() >= xi.shape[0] or idx.min() < 0: probs.append("xi idx range")
    except Exception as e:
        probs.append(f"read: {e}")
    total += size
    if probs: bad.append((d, probs))
    print(f"{m['split']:5s} {m['checkpoint_label']:28s} {m['setting']:18s} {size/1e9:5.2f} GB {'OK' if not probs else probs}", flush=True)
print(f"shards {n}, total {total/1e9:.1f} GB (apparent file sizes), bad {len(bad)}")
print("leftover tmp dirs:", glob.glob(os.path.join(root, "*", "*", "*.tmp")) + glob.glob(os.path.join(root, "*", "*", "*.old_*")))

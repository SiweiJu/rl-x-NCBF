import glob, json, os, sys, numpy as np
from collections import defaultdict
root = sys.argv[1]
agg = defaultdict(lambda: defaultdict(float)); lat = defaultdict(list)
for mp in sorted(glob.glob(os.path.join(root, "*", "*", "*", "meta.json"))):
    d = os.path.dirname(mp); m = json.load(open(mp)); key = (m["method"], m["setting"])
    term = np.load(f"{d}/terminated.npy"); done = np.load(f"{d}/done.npy")
    fallen = np.load(f"{d}/margin_physical_next.npy") < 0
    T, N = term.shape
    first = np.full(N, -1); a = agg[key]
    for t in range(T):
        first = np.where(fallen[t] & (first < 0), t, first)
        end = done[t] | (t == T - 1)
        e = np.nonzero(end)[0]
        f = first[e] >= 0; tm = term[t, e]
        a["episodes"] += e.size; a["term"] += tm.sum()
        a["fell_any"] += f.sum(); a["fell_any_term"] += (f & tm).sum(); a["fell_any_no_term"] += (f & ~tm).sum()
        a["term_without_prior_fall"] += (tm & ~f).sum()
        lat[key].extend((t - first[e][f & tm] + 1).tolist())
        first = np.where(end, -1, first)
rows = []
tot = defaultdict(float); alll = []
for key in sorted(agg):
    a = agg[key]; l = np.array(lat[key])
    for k, v in a.items(): tot[k] += v
    alll.extend(lat[key])
    print(f"| {key[0]} | {key[1]} | {int(a['episodes'])} | {int(a['term'])} | {int(a['fell_any'])} | {int(a['fell_any_no_term'])} | {int(a['term_without_prior_fall'])} | {np.median(l):.0f} / {np.percentile(l,90):.0f} |")
l = np.array(alll)
print("TOTAL", {k: int(v) for k, v in tot.items()}, "latency median/p90/max", np.median(l), np.percentile(l, 90), l.max())

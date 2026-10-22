#!/usr/bin/env python3
"""Recover the plotted numbers of the paper's result figures from the vector PDFs (T03/G0).

The eval pkls / CSVs behind Fig. 2 (exps.pdf), Fig. 4 (exps_H.pdf) and the appendix
figures are not on IAS, so this reads the matplotlib PDFs directly:
  - every axes = one clip rectangle; y tick marks + labels give the y mapping,
  - x positions of the data points are mapped to the x tick labels of the same column,
  - each stroked line (lw 1.4) is a method mean; the filled band around it is mean +- SE
    (plot scripts: fill_between(mean - se, mean + se), se = std/sqrt(n_seeds)),
  - the legend maps stroke colours to method names.

Usage: python3 -I extract_paper_curves.py <fig.pdf> [...] > paper_curves.csv
Only the standard library is needed.
"""
import re
import sys
import zlib


def streams(path):
    b = open(path, "rb").read()
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):
        try:
            yield zlib.decompress(m.group(1)).decode("latin1")
        except Exception:
            pass


NUM = r"-?\d+(?:\.\d+)?"


def parse(content):
    """Return list of events in drawing order."""
    ev = []
    clip = None
    stroke = fill = None
    width = None
    path = []
    toks = re.findall(r"\[(?:\\.|[^\]\\])*\]\s*TJ|\((?:\\.|[^)\\])*\)\s*Tj|/\w+|" + NUM + r"|[A-Za-z\*']+", content)
    stack = []
    gs = []
    cm = (0.0, 0.0)
    for t in toks:
        if re.fullmatch(NUM, t):
            stack.append(float(t))
            continue
        if t == "re":
            rect = stack[-4:]
        elif t == "W":
            clip = tuple(rect)
        elif t == "q":
            gs.append(clip)
        elif t == "Q":
            clip = gs.pop() if gs else None
        elif t == "RG":
            stroke = tuple(round(v, 3) for v in stack[-3:])
        elif t == "rg":
            fill = tuple(round(v, 3) for v in stack[-3:])
        elif t == "G":
            stroke = (round(stack[-1], 3),) * 3
        elif t == "g":
            fill = (round(stack[-1], 3),) * 3
        elif t == "w":
            width = stack[-1]
        elif t == "cm":
            cm = (stack[-2], stack[-1])
        elif t == "m":
            path = [(stack[-2], stack[-1])]
        elif t == "l":
            path.append((stack[-2], stack[-1]))
        elif t in ("S", "B"):
            ev.append(("stroke", clip, stroke, width, list(path)))
            path = []
        elif t in ("f", "f*"):
            ev.append(("fill", clip, fill, width, list(path)))
            path = []
        elif t.endswith("TJ") or t.endswith("Tj"):
            s = "".join(re.findall(r"\(((?:\\.|[^)\\])*)\)", t))
            ev.append(("text", clip, None, None, (cm, s.replace("\\", ""))))
        if t not in ("re",) and not re.fullmatch(NUM, t):
            stack = []
    return ev


def isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract(path):
    content = max(streams(path), key=len)
    ev = parse(content)
    axes = {}  # clip -> dict
    last_clip = None
    pending_tick = None  # (kind, pos)
    legend = []
    last_stroke_color = None
    titles = []
    for kind, clip, color, width, data in ev:
        if clip is not None and clip[2] > 30:
            last_clip = clip
            axes.setdefault(clip, {"yt": [], "xt": [], "lines": [], "bands": []})
        if kind == "stroke" and len(data) == 2:
            (x0, y0), (x1, y1) = data
            if abs(y0 - y1) < 1e-6 and abs(abs(x1 - x0) - 3.5) < 1e-3:
                pending_tick = ("y", y0, last_clip)
            elif abs(x0 - x1) < 1e-6 and abs(abs(y1 - y0) - 3.5) < 1e-3:
                pending_tick = ("x", x0, last_clip)
            else:
                last_stroke_color = color
                pending_tick = None
        elif kind == "stroke" and len(data) >= 3 and width and width > 1.0 and clip is not None:
            axes[clip]["lines"].append((color, data))
        elif kind == "stroke" and len(data) >= 2 and width and width > 1.0 and clip is None:
            last_stroke_color = color  # legend handle
        elif kind == "fill" and clip is not None and len(data) >= 6:
            axes[clip]["bands"].append((color, data))
        elif kind == "text":
            (cx, cy), s = data
            if pending_tick and isnum(s):
                k, pos, c = pending_tick
                axes[c]["yt" if k == "y" else "xt"].append((pos, float(s)))
                if k == "x":
                    pending_tick = pending_tick  # x tick label follows each tick
                pending_tick = None
            elif not isnum(s) and s not in ("sampling probability",):
                if " - " in s:
                    titles.append(((cx, cy), s))
                elif s.lower().startswith(("success", "avg")):
                    pass
                else:
                    legend.append((s, last_stroke_color))
    return axes, legend, titles


def lin(ticks):
    (p0, v0), (p1, v1) = ticks[0], ticks[-1]
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def main():
    print("figure,column_title,row,method,x,mean,se_band_half_width")
    for fig in sys.argv[1:]:
        axes, legend, titles = extract(fig)
        col_of = {}
        ax_list = [c for c, a in axes.items() if a["lines"]]
        xs = sorted({round(c[0], 1) for c in ax_list})
        ys = sorted({round(c[1], 1) for c in ax_list}, reverse=True)
        # x ticks: take them from any axes in the same column that has them
        xt_col = {}
        for c, a in axes.items():
            if a["xt"]:
                xt_col[round(c[0], 1)] = sorted(a["xt"])
        tit = sorted(titles, key=lambda t: t[0][0])
        tit_names = [t[1] for t in tit]
        cmap = {}
        for name, col in legend:
            if col is not None and col not in cmap:
                cmap[col] = name
        for c in ax_list:
            a = axes[c]
            col = xs.index(round(c[0], 1))
            row = ys.index(round(c[1], 1))
            ylab = "success_rate" if row == 0 else "avg_return"
            yt = a["yt"] or next((axes[o]["yt"] for o in axes if axes[o]["yt"] and abs(o[1] - c[1]) < 0.5), [])
            fy = lin(sorted(yt))
            xt = xt_col.get(round(c[0], 1), [])
            for color, pts in a["lines"]:
                name = cmap.get(color, str(color))
                band = next((b for bc, b in a["bands"] if bc == color), None)
                for (px, py) in pts:
                    xv = min(xt, key=lambda t: abs(t[0] - px))[1] if xt else px
                    half = ""
                    if band:
                        same = [q[1] for q in band if abs(q[0] - px) < 1e-3]
                        if len(same) >= 2:
                            half = f"{(fy(max(same)) - fy(min(same))) / 2:.4f}"
                    title = tit_names[col] if col < len(tit_names) else f"col{col}"
                    print(f"{fig.split('/')[-1]},{title},{ylab},{name},{xv:g},{fy(py):.4f},{half}")


if __name__ == "__main__":
    main()

"""visualization for auto_symbolic snapping: which edges became closed-form
functions, which stayed splines, and how good the fits were."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter

# palette roles
C_SNAP    = "#2a78d6"
C_SPLINE  = "#eb6834"
C_PRUNED  = "#e1e0d9"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
SURFACE   = "#fcfcfb"

STATUS_ORDER  = ["snapped", "spline", "pruned"]
STATUS_COLOR  = {"snapped": C_SNAP, "spline": C_SPLINE, "pruned": C_PRUNED}
STATUS_LABEL  = {"snapped": "snapped to formula", "spline": "kept as spline", "pruned": "pruned (zero)"}

_RE_FIX  = re.compile(r"fixing \((\d+),(\d+),(\d+)\) with (\S+?), r2=([\d.eE+-]+), c=([\d.eE+-]+)")
_RE_ZERO = re.compile(r"fixing \((\d+),(\d+),(\d+)\) with 0\s*$")
_RE_OMIT = re.compile(r"For \((\d+),(\d+),(\d+)\) the best fit was (\S+?), but r\^2 = ([\d.eE+-]+)")


def parse_symbolic_log(log_text):
    """turn auto_symbolic's stdout into per-edge records"""
    edges = []
    for line in log_text.splitlines():
        line = line.strip()
        if (m := _RE_FIX.search(line)):
            l, i, j, fun, r2, c = m.groups()
            edges.append(dict(l=int(l), i=int(i), j=int(j), status="snapped",
                              fun=fun, r2=float(r2), complexity=float(c)))
        elif (m := _RE_ZERO.search(line)):
            l, i, j = m.groups()
            edges.append(dict(l=int(l), i=int(i), j=int(j), status="pruned",
                              fun="0", r2=None, complexity=0.0))
        elif (m := _RE_OMIT.search(line)):
            l, i, j, fun, r2 = m.groups()
            edges.append(dict(l=int(l), i=int(i), j=int(j), status="spline",
                              fun=fun, r2=float(r2), complexity=None))
    return edges


def _style_axis(ax, grid_axis=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def _panel_coverage(ax, edges, n_layers):
    """stacked bar per layer: how many edges ended in each state"""
    counts = {s: [0] * n_layers for s in STATUS_ORDER}
    for e in edges:
        if e["l"] < n_layers:
            counts[e["status"]][e["l"]] += 1

    y = np.arange(n_layers)
    left = np.zeros(n_layers)
    for status in STATUS_ORDER:
        vals = np.array(counts[status], dtype=float)
        ax.barh(y, vals, left=left, height=0.62, color=STATUS_COLOR[status],
                edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, (v, l0) in enumerate(zip(vals, left)):
            if v >= max(6, 0.07 * (left + vals).max()):
                ax.text(l0 + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=8, color=SURFACE if status != "pruned" else INK_2,
                        fontweight="bold", zorder=4)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([f"layer {i}" for i in range(n_layers)], fontsize=8, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlabel("edges", fontsize=8, color=INK_2)
    ax.set_title("Where snapping happened", fontsize=10, color=INK, loc="left", pad=8)
    _style_axis(ax, grid_axis="x")


def _panel_vocabulary(ax, edges, top_n=8):
    """which closed-form functions the search actually chose"""
    funs = Counter(e["fun"] for e in edges if e["status"] == "snapped" and e["fun"] not in (None, "0"))
    if not funs:
        ax.text(0.5, 0.5, "no edges snapped", ha="center", va="center",
                color=MUTED, fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
        return
    items = funs.most_common()
    if len(items) > top_n:
        other = sum(c for _, c in items[top_n:])
        items = items[:top_n] + [("other", other)]
    names = [n for n, _ in items][::-1]
    vals  = [c for _, c in items][::-1]

    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=C_SNAP, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, str(v), va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("edges", fontsize=8, color=INK_2)
    ax.set_title("Which formulas were chosen", fontsize=10, color=INK, loc="left", pad=8)
    _style_axis(ax, grid_axis="x")


def _panel_r2(ax, edges, r2_threshold):
    """distribution of best-fit quality, split at the accept threshold"""
    snap = [e["r2"] for e in edges if e["status"] == "snapped" and e["r2"] is not None]
    omit = [e["r2"] for e in edges if e["status"] == "spline" and e["r2"] is not None]
    if not snap and not omit:
        ax.text(0.5, 0.5, "no fit scores recorded", ha="center", va="center",
                color=MUTED, fontsize=9, transform=ax.transAxes)
        ax.set_axis_off()
        return

    lo = min(snap + omit + [r2_threshold])
    bins = np.linspace(max(0.0, lo - 0.02), 1.0, 46)
    ax.hist([omit, snap], bins=bins, stacked=True,
            color=[C_SPLINE, C_SNAP], edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.axvline(r2_threshold, color=INK_2, linestyle="--", linewidth=1.4, zorder=4)
    # label sits left of the line, clear of the peak that always abuts it on the right
    ax.text(r2_threshold - (bins[-1] - bins[0]) * 0.012, ax.get_ylim()[1] * 0.97,
            f"threshold {r2_threshold:g} ", fontsize=8, color=INK_2, va="top", ha="right",
            zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    near = sum(1 for r in omit if r >= r2_threshold - 0.1)
    ax.set_xlabel("best-fit R² per edge", fontsize=8, color=INK_2)
    ax.set_ylabel("edges", fontsize=8, color=INK_2)
    ax.set_title(f"Fit quality  —  {len(snap)} accepted, {len(omit)} rejected "
                 f"({near} within 0.1 of the threshold)",
                 fontsize=10, color=INK, loc="left", pad=8)
    _style_axis(ax, grid_axis="y")


def _panel_edge_maps(fig, gs_row, edges, width, species):
    """per-layer grid of every edge, colored by outcome"""
    n_layers = len(width) - 1
    sub = gs_row.subgridspec(1, n_layers, wspace=0.32)
    code = {"pruned": 0, "snapped": 1, "spline": 2}
    cmap = matplotlib.colors.ListedColormap([C_PRUNED, C_SNAP, C_SPLINE])

    for l in range(n_layers):
        ax = fig.add_subplot(sub[0, l])
        grid = np.full((width[l], width[l + 1]), np.nan)
        for e in edges:
            if e["l"] == l and e["i"] < width[l] and e["j"] < width[l + 1]:
                grid[e["i"], e["j"]] = code[e["status"]]

        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, width[l + 1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, width[l], 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.4)
        ax.tick_params(which="minor", length=0)

        # name the real species on the network's two open ends
        if l == 0 and species is not None and len(species) == width[0]:
            ax.set_yticks(range(width[0]))
            ax.set_yticklabels(species, fontsize=6.5, color=INK_2)
        else:
            ax.set_yticks([0, width[l] - 1]); ax.set_yticklabels(["0", str(width[l] - 1)], fontsize=7)
        if l == n_layers - 1 and species is not None and len(species) == width[-1]:
            ax.set_xticks(range(width[-1]))
            ax.set_xticklabels(species, fontsize=6.5, color=INK_2, rotation=90)
        else:
            ax.set_xticks([0, width[l + 1] - 1]); ax.set_xticklabels(["0", str(width[l + 1] - 1)], fontsize=7)

        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_color(BASELINE); s.set_linewidth(0.8)
        ax.set_title(f"layer {l}   {width[l]}→{width[l+1]}", fontsize=8.5, color=INK_2, pad=5)
        if l == 0:
            ax.set_ylabel("input node", fontsize=7.5, color=MUTED)
        ax.set_xlabel("output node", fontsize=7.5, color=MUTED)


def plot_symbolic_snapping(edges, width, species=None, r2_threshold=0.9,
                           out_path="figures/symbolic_snapping.png"):
    """render the full snapping report"""
    n_layers = len(width) - 1
    total   = len(edges)
    n_snap  = sum(1 for e in edges if e["status"] == "snapped")
    n_prune = sum(1 for e in edges if e["status"] == "pruned")
    live    = total - n_prune
    pct     = (100.0 * n_snap / live) if live else 0.0

    fig = plt.figure(figsize=(14, 10.6), facecolor=SURFACE)
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 1.3],
                           hspace=0.42, wspace=0.24,
                           left=0.075, right=0.965, top=0.855, bottom=0.055)

    fig.text(0.075, 0.962, "Symbolic snapping report", fontsize=17, color=INK, fontweight="bold")
    fig.text(0.075, 0.934,
             f"{n_snap} of {live} live edges ({pct:.0f}%) replaced by closed-form functions "
             f"at R² ≥ {r2_threshold:g}   ·   {n_prune} pruned   ·   {total} edges total",
             fontsize=10.5, color=INK_2)

    # one figure-level legend: the same encoding drives both the bars and the edge maps
    fig.legend(handles=[Patch(facecolor=STATUS_COLOR[s], edgecolor=BASELINE, linewidth=0.5,
                              label=STATUS_LABEL[s]) for s in STATUS_ORDER],
               fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.072, 0.912),
               ncol=3, labelcolor=INK_2, handlelength=1.3, handleheight=1.1, columnspacing=1.8)

    _panel_coverage(fig.add_subplot(gs[0, 0]), edges, n_layers)
    _panel_vocabulary(fig.add_subplot(gs[0, 1]), edges)
    _panel_r2(fig.add_subplot(gs[1, :]), edges, r2_threshold)
    _panel_edge_maps(fig, gs[2, :], edges, width, species)

    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _mock_edges(width, seed=0):
    """realistic stand-in metadata at true network scale, for layout testing"""
    rng = np.random.default_rng(seed)
    lib = ["x", "x^2", "sin", "tanh", "exp", "log", "sqrt", "1/x", "x^3", "abs"]
    p   = np.array([28, 18, 13, 12, 9, 7, 5, 4, 2, 2], dtype=float); p /= p.sum()
    edges = []
    for l in range(len(width) - 1):
        for i in range(width[l]):
            for j in range(width[l + 1]):
                u = rng.random()
                if u < 0.12:
                    edges.append(dict(l=l, i=i, j=j, status="pruned", fun="0", r2=None, complexity=0.0))
                elif u < 0.62:
                    edges.append(dict(l=l, i=i, j=j, status="snapped", fun=str(rng.choice(lib, p=p)),
                                      r2=float(min(0.999, 0.9 + rng.beta(1.6, 3.2) * 0.1)),
                                      complexity=float(rng.integers(1, 5))))
                else:
                    edges.append(dict(l=l, i=i, j=j, status="spline", fun=str(rng.choice(lib, p=p)),
                                      r2=float(np.clip(rng.beta(5, 2.2) * 0.9, 0.05, 0.899)), complexity=None))
    return edges


if __name__ == "__main__":
    import json, os, sys
    meta_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kan_probe_ckpt/symbolic_meta.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/symbolic_snapping.png"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"rendering from real metadata: {meta_path}")
        plot_symbolic_snapping(meta["edges"], meta["width"], meta.get("species"),
                               meta.get("r2_threshold", 0.9), out)
    else:
        w = [11, 16, 16, 16, 11]
        sp = ["O3","NO","NO2","HCHO","HO2","H2O2","OH","ALD2","MGLY","MCO3","PAN"]
        print("no real metadata yet — rendering mock at true scale")
        plot_symbolic_snapping(_mock_edges(w), w, sp, 0.9, out)

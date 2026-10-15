"""side-by-side comparison of finished runs, for the write-up.

per-run loss curves and 1:1 grids are written by experiment.py already. this adds the
cross-run view: val curves on one axis and per-species r2 as grouped bars, plus a
markdown table. everything is the unweighted val mse -- a weighted run's own val_mse is
on its own scale and is not comparable with an unweighted run's.

usage: python compare_runs.py runs/deliverable/cpu/A_unreg runs/deliverable/cpu/B_oh60
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    h = np.load(os.path.join(run_dir, "history.npz"), allow_pickle=True)
    return m, h


def curve_key(h):
    # runs from before the weighted-loss change only have val_mse
    return "val_mse_uw" if "val_mse_uw" in h.files else "val_mse"


def plot(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for m, h in runs:
        val = np.array(h[curve_key(h)], dtype=float)
        has = ~np.isnan(val)
        steps = np.arange(1, len(val) + 1)
        ax.semilogy(steps[has], val[has], "o-", ms=3, lw=1.2, label=m["name"])
    for s, name in zip(runs[0][1]["events"], runs[0][1]["event_names"]):
        ax.axvline(int(s), color="0.6", ls="--", lw=1)
        ax.text(int(s), ax.get_ylim()[1], str(name), rotation=90, va="top", ha="right",
                fontsize=8, color="0.4")
    ax.set_xlabel("LBFGS step")
    ax.set_ylabel("val MSE (standardized, unweighted)")
    ax.set_title("validation loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    species = list(runs[0][0]["test_r2"].keys())
    x = np.arange(len(species))
    width = 0.8 / len(runs)
    for i, (m, _) in enumerate(runs):
        r2 = [m["test_r2"][s] for s in species]
        ax.bar(x + i * width - 0.4 + width / 2, r2, width, label=m["name"])
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45)
    ax.set_ylabel("test R²")
    ax.set_ylim(min(0.9, min(min(m["test_r2"].values()) for m, _ in runs) - 0.02), 1.002)
    ax.set_title("per-species test R² (zoomed; floor clipped at 0.9)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def table(runs):
    cols = [("name", "{}"), ("device", "{}"), ("n_params", "{}"), ("steps_run", "{}"),
            ("total_time_s", "{:.0f}"), ("sec_per_step_mean", "{:.1f}"),
            ("best_val_mse_uw", "{:.5f}"), ("test_r2_mean", "{:.4f}"),
            ("test_r2_oh", "{:.4f}"), ("test_r2_mean_no_oh", "{:.4f}")]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |",
             "|" + "---|" * len(cols)]
    for m, _ in runs:
        lines.append("| " + " | ".join(
            fmt.format(m[c]) if m.get(c) is not None else "-" for c, fmt in cols) + " |")
    return "\n".join(lines)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    runs = [load(d) for d in dirs]
    out = os.path.join(os.path.dirname(dirs[0].rstrip("/")), "comparison.png")
    print(plot(runs, out))
    print()
    print(table(runs))


if __name__ == "__main__":
    main()

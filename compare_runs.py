# compares finished runs: writes comparison.png beside the first run dir and prints a markdown table.
#   python compare_runs.py runs/<batch>/cuda/<run1> runs/<batch>/cuda/<run2> ...
import json
import os
import sys

import numpy as np

from smogkan.plots import plot_comparison


def load(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    h = np.load(os.path.join(run_dir, "history.npz"), allow_pickle=True)
    return m, h


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
        sys.exit("usage: python compare_runs.py <run dir> [<run dir> ...]")
    runs = [load(d) for d in dirs]
    out = os.path.join(os.path.dirname(dirs[0].rstrip("/")), "comparison.png")
    print(plot_comparison(runs, out))
    print()
    print(table(runs))


if __name__ == "__main__":
    main()

import argparse
import csv
import functools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import config as cfg
import experiment
import kan_patches

print = functools.partial(print, flush=True)

SUMMARY_COLS = ["name", "device", "width_final", "n_params", "steps_run", "total_time_s",
                "sec_per_step_mean", "closure_evals_mean", "final_train_mse", "best_val_mse",
                "test_mse_scaled", "test_r2_mean"]


def load_runs(path, only):
    with open(path) as f:
        spec = yaml.safe_load(f)
    runs = [{**spec["defaults"], **r} for r in spec["runs"]]
    if only:
        runs = [r for r in runs if r["name"] in only]
    return runs


def write_summary(results, out_dir):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        for m, _ in results:
            w.writerow([m[c] for c in SUMMARY_COLS])

    names = [m["name"] for m, _ in results]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.barh(names, [m["best_val_mse"] for m, _ in results])
    ax.set_xscale("log")
    ax.set_xlabel("best val MSE (standardized)")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.barh(names, [m["sec_per_step_mean"] for m, _ in results])
    ax.set_xlabel("mean seconds / LBFGS step")
    ax.invert_yaxis()

    for m, log in results:
        val = np.array(log["val_mse"], dtype=float)
        has = ~np.isnan(val)
        steps = np.arange(1, len(val) + 1)
        wall = np.cumsum(log["step_time"])
        axes[1, 0].semilogy(steps[has], val[has], lw=1, label=m["name"])
        axes[1, 1].semilogy(wall[has], val[has], lw=1, label=m["name"])
    axes[1, 0].set_xlabel("LBFGS step")
    axes[1, 1].set_xlabel("wall-clock time (s)")
    for ax in axes[1]:
        ax.set_ylabel("val MSE")
        ax.grid(alpha=0.3)
    axes[1, 1].legend(fontsize=7, loc="upper right")
    for ax in axes[0]:
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("phase 1 summary")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(cfg.BASE_DIR, "configs", "phase1.yaml"))
    p.add_argument("--csv", default=cfg.CSV_PATH)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default=os.path.join(cfg.BASE_DIR, "runs", "phase1"))
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    kan_patches.install()

    runs = load_runs(args.config, args.runs)
    if args.steps:
        for r in runs:
            r["steps"] = args.steps
    out_dir = os.path.join(args.out, device.type)
    os.makedirs(out_dir, exist_ok=True)

    n_max = max(r["n_exps"] for r in runs)
    X, D, species = experiment.load_raw(args.csv, n_max)
    splits = experiment.split_experiments(n_max, runs[0]["seed"])
    print(f"experiments: {n_max}  train/val/test: {[len(s) for s in splits]}  "
          f"device: {device}  threads: {torch.get_num_threads()}")

    results = []
    for rc in runs:
        print(f"\n=== {rc['name']} ===")
        print(json.dumps(rc))
        metrics, log = experiment.run(rc, X, D, species, splits, device, os.path.join(out_dir, rc["name"]))
        print(f"  done: {metrics['total_time_s']:.1f}s  {metrics['sec_per_step_mean']:.3f}s/step  "
              f"best_val={metrics['best_val_mse']:.5f}  test_r2_mean={metrics['test_r2_mean']:.3f}")
        results.append((metrics, log))
        write_summary(results, out_dir)

    print(f"\nall runs written to {out_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

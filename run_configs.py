# runs every entry of a config yaml (or the ones named with --runs) and writes one folder per run plus summary.csv.
#   python run_configs.py --config configs/stoich_overnight.yaml --data data/baseline_10k/baseline_10k.npz --runs S13_1200
import argparse
import csv
import json
import os
import sys
import traceback

import torch
import yaml

import kan_patches
from smogkan import paths
from smogkan.data import load_npz, load_raw, split_experiments
from smogkan.plots import plot_summary
from smogkan.runner import run

SUMMARY_COLS = ["name", "device", "width_final", "n_params", "steps_run", "total_time_s",
                "sec_per_step_mean", "closure_evals_mean", "final_train_mse", "best_val_mse",
                "best_val_mse_uw", "test_mse_scaled", "test_r2_mean", "test_r2_mean_no_oh",
                "test_r2_oh", "test_r2_mean_11"]


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

    plot_summary(results, out_dir)


def main():
    p = argparse.ArgumentParser(description="train every run in a config yaml")
    p.add_argument("--config", required=True,
                   help="yaml with defaults: and runs:, e.g. configs/stoich_overnight.yaml")
    p.add_argument("--csv", default=paths.CSV_PATH, help="full csv, used only without --data")
    p.add_argument("--data", default=None,
                   help="prebuilt dataset npz; overrides --csv and the derived split")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--threads", type=int, default=None, help="torch cpu threads")
    p.add_argument("--out", default=None, help="output root, default runs/<config name>")
    p.add_argument("--runs", nargs="*", default=None, help="only these run names")
    p.add_argument("--steps", type=int, default=None,
                   help="override steps for every run; needs at least 1 per stage")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(paths.RUNS_DIR, os.path.splitext(os.path.basename(args.config))[0])

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

    # each run picks the 11- or 16-species view of the dataset, loaded once.
    views = {}

    def view(rc):
        key = rc.get("species", "active11")
        if key not in views:
            views[key] = load_npz(args.data, all_species=key == "all16")[:4]
        return views[key]

    if args.data:
        X, D, species, splits, n_max = load_npz(args.data)
        views["active11"] = (X, D, species, splits)
    elif any(r.get("species") == "all16" for r in runs):
        raise ValueError("species: all16 needs --data (the 16 species come from the fixed dataset)")
    else:
        n_max = max(r["n_exps"] for r in runs)
        X, D, species = load_raw(args.csv, n_max)
        splits = split_experiments(n_max, runs[0]["seed"])
    print(f"experiments: {n_max}  train/val/test: {[len(s) for s in splits]}  "
          f"device: {device}  threads: {torch.get_num_threads()}")

    results, failed = [], []
    for rc in runs:
        print(f"\n=== {rc['name']} ===")
        print(json.dumps(rc))
        if args.data:
            X, D, species, splits = view(rc)
        # one failed run is logged and skipped so an unattended batch still finishes the rest.
        try:
            metrics, log = run(rc, X, D, species, splits, device, os.path.join(out_dir, rc["name"]))
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {rc['name']}")
            failed.append(rc["name"])
            continue
        print(f"  done: {metrics['total_time_s']:.1f}s  {metrics['sec_per_step_mean']:.3f}s/step  "
              f"best_val={metrics['best_val_mse']:.5f}  test_r2_mean={metrics['test_r2_mean']:.3f}")
        results.append((metrics, log))
        write_summary(results, out_dir)

    print(f"\nall runs written to {out_dir}")
    if failed:
        sys.exit(f"failed runs: {', '.join(failed)}")


if __name__ == "__main__":
    # line buffering so logs redirected to a file show each line as it is printed.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()

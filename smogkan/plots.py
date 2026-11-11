import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smogkan.training import cache_acts


def plot_loss(log, events, rc, path):
    steps = np.arange(1, len(log["train_mse"]) + 1)
    val = np.array(log["val_mse"], dtype=float)
    has = ~np.isnan(val)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    weighted = bool(rc.get("species_weights"))
    suffix = " (weighted)" if weighted else ""
    ax.semilogy(steps, log["train_mse"], label=f"train MSE{suffix}", lw=1)
    ax.semilogy(steps[has], val[has], "o-", ms=3, label=f"val MSE{suffix}", lw=1)
    if weighted:
        val_uw = np.array(log["val_mse_uw"], dtype=float)
        ax.semilogy(steps, log["train_mse_uw"], ls="--", lw=1, color="tab:blue",
                    alpha=0.6, label="train MSE (unweighted)")
        ax.semilogy(steps[has], val_uw[has], ls="--", lw=1, color="tab:orange",
                    alpha=0.6, label="val MSE (unweighted)")
    if rc["lamb"] > 0:
        ax2 = ax.twinx()
        ax2.semilogy(steps, np.maximum(log["reg"], 1e-12), color="tab:green", alpha=0.6, lw=1, label="reg")
        ax2.set_ylabel("reg term (unscaled)", color="tab:green")
    for s, name in events:
        ax.axvline(s, color="tab:orange", ls="--", lw=1)
        ax.text(s, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("LBFGS step")
    ax.set_ylabel("MSE (standardized)")
    ax.set_title(f"{rc['name']}: loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, log["step_time"], lw=1)
    ax.set_xlabel("LBFGS step")
    ax.set_ylabel("seconds / step")
    ax.set_title(f"step time (mean {np.mean(log['step_time']):.3f}s)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(pred, true, species, r2, title, path):
    n = len(species)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.scatter(true[:, i], pred[:, i], s=2, alpha=0.3)
        lo, hi = np.percentile(true[:, i], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "r-", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{species[i]}  R²={r2[i]:.3f}", fontsize=10)
        ax.set_xlabel("actual (ppb/step)", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
    fig.suptitle(f"{title}: test set")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_rmse(rmse, r2, species, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(species, rmse)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE (ppb/step)")
    axes[1].bar(species, r2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylim(min(-0.1, np.min(r2) - 0.05), 1.05)
    axes[1].set_ylabel("R²")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{title}: test set per species")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# the per-edge pngs pykan needs go to a temp folder that is deleted afterwards; only kan_graph.png is kept.
def plot_graph(model, x, species, path, out_vars=None):
    try:
        cache_acts(model, x[:2000])
        with tempfile.TemporaryDirectory() as edges:
            model.plot(folder=edges, in_vars=species, out_vars=out_vars or species, scale=1.0, varscale=0.3)
            plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"  kan graph skipped: {e}")
    finally:
        model.save_act = False


# one batch: best val loss and step time per run, then val loss against steps and wall-clock time.
def plot_summary(results, out_dir):
    names = [m["name"] for m, _ in results]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.barh(names, [m["best_val_mse_uw"] for m, _ in results])
    ax.set_xscale("log")
    ax.set_xlabel("best val MSE (standardized, unweighted)")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.barh(names, [m["sec_per_step_mean"] for m, _ in results])
    ax.set_xlabel("mean seconds / LBFGS step")
    ax.invert_yaxis()

    for m, log in results:
        val = np.array(log["val_mse_uw"], dtype=float)
        has = ~np.isnan(val)
        steps = np.arange(1, len(val) + 1)
        wall = np.cumsum(log["step_time"])
        axes[1, 0].semilogy(steps[has], val[has], lw=1, label=m["name"])
        axes[1, 1].semilogy(wall[has], val[has], lw=1, label=m["name"])
    axes[1, 0].set_xlabel("LBFGS step")
    axes[1, 1].set_xlabel("wall-clock time (s)")
    for ax in axes[1]:
        ax.set_ylabel("val MSE (unweighted)")
        ax.grid(alpha=0.3)
    axes[1, 1].legend(fontsize=7, loc="upper right")
    for ax in axes[0]:
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("batch summary")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=130)
    plt.close(fig)


def curve_key(h):
    return "val_mse_uw" if "val_mse_uw" in h.files else "val_mse"


def plot_comparison(runs, out_path):
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
    # only species every run predicts, so an 11-species run can sit next to a 16-species one.
    species = [s for s in runs[0][0]["test_r2"] if all(s in m["test_r2"] for m, _ in runs)]
    x = np.arange(len(species))
    width = 0.8 / len(runs)
    for i, (m, _) in enumerate(runs):
        r2 = [m["test_r2"][s] for s in species]
        ax.bar(x + i * width - 0.4 + width / 2, r2, width, label=m["name"])
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45)
    ax.set_ylabel("test R²")
    ax.set_ylim(min(0.9, min(m["test_r2"][s] for m, _ in runs for s in species) - 0.02), 1.002)
    ax.set_title("per-species test R² (zoomed; floor clipped at 0.9)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path

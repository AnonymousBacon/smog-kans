import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config as cfg


def plot_distributions(X_log, Y_raw, names):
    configs = [
        ("log1p (input)",  X_log,                                Y_raw),
        ("MinMaxScaler",   MinMaxScaler().fit_transform(X_log),   MinMaxScaler().fit_transform(Y_raw)),
        ("StandardScaler", StandardScaler().fit_transform(X_log), StandardScaler().fit_transform(Y_raw)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Input/Output Distributions: Raw vs Scalers", fontsize=13)

    for row, (label, X_s, Y_s) in enumerate(configs):
        for col, (data, title) in enumerate([(X_s, f"X — {label}"), (Y_s, f"Y (tendencies) — {label}")]):
            ax = axes[row, col]
            ax.boxplot(data, tick_labels=names, patch_artist=True,
                       flierprops=dict(marker='.', markersize=1, alpha=0.3))
            ax.set_title(title)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel("Value")
            if row == 2:
                p1, p99 = np.percentile(data, [1, 99])
                pad = (p99 - p1) * 0.5
                ax.set_ylim(p1 - pad, p99 + pad)
                ax.set_title(f"{title} (axis clipped to 1–99th pct)")

    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR + "distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/distributions.png")


def plot_loss_curves():
    if os.path.exists(cfg.HIST_PATH):
        _h = np.load(cfg.HIST_PATH)
        train_loss = list(_h["train_loss"])
        test_loss  = list(_h["test_loss"])

        # pykan stores sqrt() of the loss it was given, and it was given
        # weighted_mse, so these series are weighted RMSE and not MSE
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Train vs Test Loss (weighted RMSE, log scale)", fontsize=13)
        ax.semilogy(train_loss, label='train')
        ax.semilogy(test_loss,  label='test', linestyle='--')
        ax.set_xlabel("Steps")
        ax.set_ylabel("weighted RMSE (scaled units)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(cfg.FIG_DIR + "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: figures/loss_curves.png")
    else:
        print("No loss history found — run training first to generate loss_curves.png")


def plot_scatter(preds, actual_ppb, species_names, n_species):
    n_cols = 4
    n_rows = -(-n_species // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()
    for i, (name, ax) in enumerate(zip(species_names, axes)):
        a, p = actual_ppb[:, i], preds[:, i]

        # np.polyfit raises LinAlgError on non-finite input, which would take out
        # the whole figure just because one species went bad
        finite = np.isfinite(a) & np.isfinite(p)
        a, p = a[finite], p[finite]
        if a.size < 2:
            ax.set_title(f"{name}  (no finite predictions)", fontsize=9)
            ax.set_visible(True)
            continue

        ax.scatter(a, p, s=1, alpha=0.3)

        # 1:1 reference line
        lims = [min(a.min(), p.min()), max(a.max(), p.max())]
        ax.plot(lims, lims, 'r--', linewidth=0.8, label='1:1')

        # ols fit
        m, b = np.polyfit(a, p, 1)
        x_fit = np.linspace(lims[0], lims[1], 200)
        ax.plot(x_fit, m * x_fit + b, 'b-', linewidth=0.9, label=f'OLS (m={m:.2f})')

        ss_res = np.sum((p - (m * a + b)) ** 2)
        ss_tot = np.sum((p - p.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.set_title(f"{name}  R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=6)
    for ax in axes[n_species:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR + "scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/scatter.png")


def plot_kan_graph(model, dataset, species_names, n_species):
    model.save_act = True
    with torch.no_grad():
        # seeded so the same run always composites the graph from the same sample
        gen = torch.Generator(device='cpu').manual_seed(cfg.SEED)
        plot_idx = torch.randperm(dataset['train_input'].shape[0], generator=gen)[:8192]
        out = model(dataset['train_input'][plot_idx])

    # plot() turns activations into alpha values, so a non-finite model dies deep
    # inside matplotlib with an opaque "alpha is outside 0-1 range" error
    if not torch.isfinite(out).all():
        print("Skipped KAN graph: model produces non-finite activations, retrain first")
        return

    # pykan's label font size (40*scale*varscale) and label spacing (fig_width/n_species)
    # both scale with `scale`, so only varscale (not scale) can prevent label overlap
    max_label_len = max(len(s) for s in species_names)
    label_varscale = min(1.0, max(0.25, 19 / (n_species * max_label_len)))
    try:
        model.plot(in_vars=species_names, out_vars=species_names, scale=1.0, varscale=label_varscale)
        plt.savefig(cfg.GRAPH_PATH, dpi=200, bbox_inches="tight")
        print(f"Saved KAN graph to {cfg.GRAPH_PATH}")
    except MemoryError as e:
        print(f"Skipped KAN graph: ran out of memory compositing it ({e}). "
              f"This is a pykan scaling limit at this network width, not a training/eval problem — "
              f"everything else already saved successfully.")

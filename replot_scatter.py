import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_PATH = "/Users/baron/Documents/KAN testing/predictions.npz"
FIG_DIR   = "/Users/baron/Documents/KAN testing/figures/"

data         = np.load(SAVE_PATH, allow_pickle=True)
preds        = data["predictions"]
actual_ppb   = data["actuals"]
species_names = list(data["species"])

n_species = len(species_names)
n_cols = 4
n_rows = -(-n_species // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes = axes.flatten()

for i, (name, ax) in enumerate(zip(species_names, axes)):
    a, p = actual_ppb[:, i], preds[:, i]
    ax.scatter(a, p, s=1, alpha=0.3)

    lims = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.plot(lims, lims, 'r--', linewidth=0.8, label='1:1')

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
plt.savefig(FIG_DIR + "scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/scatter.png")

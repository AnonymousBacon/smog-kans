import numpy as np
import matplotlib.pyplot as plt
import os

PRED_PATH  = "/Users/baron/Documents/KAN testing/predictions.npz"
OUT_FOLDER = "/Users/baron/Desktop/smog_analysis"
os.makedirs(OUT_FOLDER, exist_ok=True)

data        = np.load(PRED_PATH, allow_pickle=True)
predictions = data["predictions"]   # shape [N, 11]
actuals     = data["actuals"]       # shape [N, 11]
species     = list(data["species"]) # list of 11 species names

print(f"Loaded {predictions.shape[0]} test samples across {len(species)} species.")
print(f"Species: {species}\n")

# RMSE
rmse_per_species = np.sqrt(np.mean((predictions - actuals) ** 2, axis=0))

print("Per-species RMSE (ppb/step):")
for name, rmse in sorted(zip(species, rmse_per_species), key=lambda x: x[1], reverse=True):
    print(f"  {name:<10}  {rmse:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
sorted_pairs = sorted(zip(species, rmse_per_species), key=lambda x: x[1], reverse=True)
names_sorted, rmse_sorted = zip(*sorted_pairs)
bars = ax.bar(names_sorted, rmse_sorted, color="steelblue", edgecolor="white")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax.set_title("Per-Species Prediction Error (RMSE)", fontsize=14)
ax.set_ylabel("RMSE (ppb / time step)")
ax.set_xlabel("Chemical Species")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, "1_rmse_by_species.png"), dpi=150)
plt.close()
print("\nSaved: 1_rmse_by_species.png")

# prediction vs actual in scatter plots
n_cols = 4
n_rows = (len(species) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
axes = axes.flatten()

for i, name in enumerate(species):
    ax = axes[i]
    ax.scatter(actuals[:, i], predictions[:, i], alpha=0.15, s=4, color="steelblue")
    lo = min(actuals[:, i].min(), predictions[:, i].min())
    hi = max(actuals[:, i].max(), predictions[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Actual (ppb)")
    ax.set_ylabel("Predicted (ppb)")
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Predicted vs Actual — All Species", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, "2_scatter_all_species.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_scatter_all_species.png")

# sample exps
N_STEPS = 12
n_experiments = predictions.shape[0] // N_STEPS
exp_idx = 0  # first test experiment
start   = exp_idx * N_STEPS
end     = start + N_STEPS

# top 3 highest rmse species
top3_idx   = np.argsort(rmse_per_species)[::-1][:3]
top3_names = [species[i] for i in top3_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
time_steps = np.arange(1, N_STEPS + 1)

for ax, idx, name in zip(axes, top3_idx, top3_names):
    ax.plot(time_steps, actuals[start:end, idx],     "o-", label="Actual",    color="royalblue")
    ax.plot(time_steps, predictions[start:end, idx], "s--", label="Predicted", color="tomato")
    ax.set_title(f"{name} — Experiment #{exp_idx + 1}", fontsize=11)
    ax.set_xlabel("Time step (5-min intervals)")
    ax.set_ylabel("Tendency (ppb / step)")
    ax.legend()

fig.suptitle("Time-Series: Predicted vs Actual Tendencies (Top 3 Hardest Species)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, "3_timeseries_top3.png"), dpi=150)
plt.close()
print("Saved: 3_timeseries_top3.png")

# overall accuracy
overall_rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
overall_r2   = []
for i in range(len(species)):
    ss_res = np.sum((actuals[:, i] - predictions[:, i]) ** 2)
    ss_tot = np.sum((actuals[:, i] - np.mean(actuals[:, i])) ** 2)
    overall_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

print(f"\nOverall RMSE across all species: {overall_rmse:.4f} ppb/step")
print("\nR² scores per species (1.0 = perfect, 0 = no better than the mean):")
for name, r2 in sorted(zip(species, overall_r2), key=lambda x: x[1], reverse=True):
    bar = "█" * int(max(r2, 0) * 20)
    print(f"  {name:<10}  {r2:+.4f}  {bar}")

print(f"\nAll plots saved to: {OUT_FOLDER}")

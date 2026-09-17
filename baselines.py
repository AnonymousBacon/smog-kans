import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import time

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(BASE_DIR, "figures", "")
SPLITS     = os.path.join(BASE_DIR, "model", "splits.npz")
SCALER_Y   = os.path.join(BASE_DIR, "model", "scalerY.pkl")
KAN_PREDS  = os.path.join(BASE_DIR, "predictions.npz")
SEED       = 42
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # mps skipped: incomplete op coverage for pykan/lbfgs
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# load preprocessed splits saved by model.py
# these are already log1p + StandardScaler transformed — same as what KAN trained on
d          = np.load(SPLITS, allow_pickle=True)
X_train    = d['X_train']
Y_train    = d['Y_train']
X_test     = d['X_test']
Y_test     = d['Y_test']
species    = list(d['species'])
scalerY    = joblib.load(SCALER_Y)
n_species  = len(species)
print(f"train: {X_train.shape} | test: {X_test.shape}")

# load KAN predictions for comparison
_kan       = np.load(KAN_PREDS, allow_pickle=True)
kan_preds  = _kan['predictions']   # already in ppb
kan_actual = _kan['actuals']       # already in ppb


# xgboost
print("\ntraining xgboost...")
t0 = time.time()
xgb = MultiOutputRegressor(
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                 tree_method='hist', device=str(device),
                 random_state=SEED, verbosity=0),
    n_jobs=-1
)
xgb.fit(X_train, Y_train)
xgb_time = time.time() - t0
print(f"xgboost training time: {xgb_time:.1f}s")

xgb_preds_scaled = xgb.predict(X_test)
xgb_preds  = scalerY.inverse_transform(xgb_preds_scaled)
actual_ppb = scalerY.inverse_transform(Y_test)   # ground truth in ppb


# ffnn
class FFNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)

# weighted loss - reuses the exact per-species weights model.py trained with
_weights = torch.tensor(d['species_weight'], dtype=torch.float32, device=device)

def weighted_mse(pred, target):
    return ((pred - target) ** 2 * _weights).mean()

def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

Xtr, Ytr = t(X_train), t(Y_train)
Xte, Yte = t(X_test),  t(Y_test)

torch.manual_seed(SEED)
ffnn      = FFNN(n_species, n_species).to(device)
optimizer = torch.optim.Adam(ffnn.parameters(), lr=1e-3)
STEPS     = 2000
BATCH     = 1024

print(f"\ntraining ffnn ({STEPS} steps, batch={BATCH})...")
t0 = time.time()
for step in range(STEPS):
    idx    = torch.randint(0, Xtr.shape[0], (BATCH,))
    loss   = weighted_mse(ffnn(Xtr[idx]), Ytr[idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 500 == 0:
        with torch.no_grad():
            val_loss = weighted_mse(ffnn(Xte), Yte).item()
        print(f"  step {step+1}/{STEPS} | val loss: {val_loss:.6f}")
ffnn_time = time.time() - t0
print(f"ffnn training time: {ffnn_time:.1f}s")

with torch.no_grad():
    ffnn_preds_scaled = ffnn(Xte).cpu().numpy()
ffnn_preds = scalerY.inverse_transform(ffnn_preds_scaled)


# rmse comparison
def rmse_per_species(preds, actual):
    return np.sqrt(np.mean((preds - actual) ** 2, axis=0))

kan_rmse  = rmse_per_species(kan_preds,  kan_actual)
xgb_rmse  = rmse_per_species(xgb_preds,  actual_ppb)
ffnn_rmse = rmse_per_species(ffnn_preds, actual_ppb)

print("\nper-species RMSE (ppb/step):")
print(f"  {'species':10s}  {'KAN':>8}  {'XGBoost':>8}  {'FFNN':>8}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for i, name in enumerate(species):
    print(f"  {name:10s}  {kan_rmse[i]:8.4f}  {xgb_rmse[i]:8.4f}  {ffnn_rmse[i]:8.4f}")
print(f"\n  {'mean':10s}  {kan_rmse.mean():8.4f}  {xgb_rmse.mean():8.4f}  {ffnn_rmse.mean():8.4f}")

# grouped bar chart
x  = np.arange(n_species)
w  = 0.26
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w,   kan_rmse,  w, label='KAN')
ax.bar(x,       xgb_rmse,  w, label='XGBoost')
ax.bar(x + w,   ffnn_rmse, w, label='FFNN')
ax.set_xticks(x)
ax.set_xticklabels(species, rotation=45, ha='right')
ax.set_ylabel("RMSE (ppb/step)")
ax.set_title("per-species RMSE comparison: KAN vs XGBoost vs FFNN")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR + "comparison_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: figures/comparison_rmse.png")


# ols scatter per model
def plot_scatter(preds, actual, names, title, filename):
    n_cols = 4
    n_rows = -(-len(names) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    fig.suptitle(title, fontsize=12)
    axes = axes.flatten()
    for i, (name, ax) in enumerate(zip(names, axes)):
        a, p = actual[:, i], preds[:, i]
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
    for ax in axes[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR + filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figures/{filename}")

plot_scatter(xgb_preds,  actual_ppb, species, "XGBoost — predicted vs actual", "scatter_xgb.png")
plot_scatter(ffnn_preds, actual_ppb, species, "FFNN — predicted vs actual",    "scatter_ffnn.png")

import functools
print = functools.partial(print, flush=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


CSV_PATH    = "C:/kan-project/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv"
FIG_DIR     = "C:/kan-project/figures/"
N_EXPS      = 10000
TRAIN_SPLIT = 0.8
SEED        = 42
EPOCHS      = 1000
LR          = 1e-3

os.makedirs(FIG_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv(CSV_PATH, nrows=13 * N_EXPS)
print(f"Loaded {df.shape[0]} rows")

assert all(df.columns[2:].str.endswith("[ppb]")), (
    "Unexpected non-ppb columns after index 2, csv"
)

# load, clean
ignore        = []
active_cols   = [c for c in df.columns if c.split(" ")[0] not in ignore and c.endswith("[ppb]")]
species_names = [c.split(" ")[0] for c in active_cols]
all_ppb_cols  = list(df.iloc[:, 2:].columns)
active_indices = [all_ppb_cols.index(c) for c in active_cols]

C_active = df[active_cols].values
C_all    = df.iloc[:, 2:].values

N, n_pts, n_steps = df.shape[0], 13, 12

assert df.shape[0] == n_pts * N_EXPS, (
    f"Expected {n_pts * N_EXPS} rows, got {df.shape[0]}."
)

D_raw   = np.diff(C_all, axis=0)
D_clean = np.delete(D_raw,    list(range(n_steps, N - 1, n_pts)), axis=0)
X_clean = np.delete(C_active, list(range(n_steps, N,     n_pts)), axis=0)

D_active  = D_clean[:, active_indices]
n_species = len(species_names)
print(f"Species ({n_species}): {species_names}")
print(f"X: {X_clean.shape}  |  D: {D_active.shape}")

# scale
scalerX  = StandardScaler()
scalerY  = StandardScaler()
X_scaled = scalerX.fit_transform(np.log1p(X_clean))
Y_scaled = scalerY.fit_transform(D_active)

n_train          = int(X_scaled.shape[0] * TRAIN_SPLIT)
X_train, Y_train = X_scaled[:n_train], Y_scaled[:n_train]
X_test,  Y_test  = X_scaled[n_train:], Y_scaled[n_train:]
print(f"train: {X_train.shape} | test: {X_test.shape}")

def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

X_train_t, Y_train_t = t(X_train), t(Y_train)
X_test_t,  Y_test_t  = t(X_test),  t(Y_test)

# same architecture as KAN
class FFNN(nn.Module):
    def __init__(self, n_species, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_species, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_species),
        )
    def forward(self, x):
        return self.net(x)

model = FFNN(n_species).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"\nTraining FFNN for {EPOCHS} epochs...")
train_loss_hist, test_loss_hist = [], []
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, Y_train_t)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        test_mse = loss_fn(model(X_test_t), Y_test_t).item()
    
    train_rmse = loss.item() ** 0.5
    test_rmse  = test_mse ** 0.5
    train_loss_hist.append(train_rmse)
    test_loss_hist.append(test_rmse)

    if (epoch + 1) % 200 == 0:
        print(f"  epoch {epoch+1:4d}: train_rmse={train_rmse:.6f}  test_rmse={test_rmse:.6f}")

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("FFNN Train vs Test Loss (RMSE, log scale)", fontsize=13)
ax.semilogy(train_loss_hist, label='train')
ax.semilogy(test_loss_hist,  label='test', linestyle='--')
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE Loss")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR + "ffnn_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/ffnn_loss_curves.png")

# evaluate
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t)
mse       = torch.mean((preds_scaled - Y_test_t) ** 2).item()
preds_ppb = scalerY.inverse_transform(preds_scaled.cpu().numpy())
actual    = scalerY.inverse_transform(Y_test_t.cpu().numpy())
rmse_per  = np.sqrt(np.mean((preds_ppb - actual) ** 2, axis=0))

print(f"\nTest MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
for name, rmse in zip(species_names, rmse_per):
    print(f"  {name:10s}: {rmse:.4f} ppb/step")

# rmse bar chart
x, w = np.arange(n_species), 0.4
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x, rmse_per, w)
ax.set_xticks(x)
ax.set_xticklabels(species_names, rotation=45, ha='right')
ax.set_ylabel("RMSE (ppb/step)")
ax.set_title("FFNN Per-Species RMSE")
plt.tight_layout()
plt.savefig(FIG_DIR + "ffnn_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/ffnn_rmse.png")

# ols scatter: predicted vs actual per species
n_cols = 4
n_rows = -(-n_species // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes = axes.flatten()
for i, (name, ax) in enumerate(zip(species_names, axes)):
    a, p = actual[:, i], preds_ppb[:, i]
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
plt.savefig(FIG_DIR + "ffnn_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/ffnn_scatter.png")

np.savez(
    "C:/kan-project/model/ffnn_predictions.npz",
    predictions=preds_ppb, actuals=actual, species=np.array(species_names), rmse_per=rmse_per,
)
print("Saved predictions to C:/kan-project/model/ffnn_predictions.npz")

import functools
print = functools.partial(print, flush=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# NOTE: keep CSV_PATH / N_EXPS / TRAIN_SPLIT / ignore list in sync with smog_model.py
# so this baseline is trained and evaluated on exactly the same data/split as the KAN.
CSV_PATH    = "C:/kan-project/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv"
FIG_DIR     = "C:/kan-project/figures/"
N_EXPS      = 10000
TRAIN_SPLIT = 0.8
SEED        = 42
EPOCHS      = 2000
LR          = 1e-3

os.makedirs(FIG_DIR, exist_ok=True)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load and clean (identical to smog_model.py)
df = pd.read_csv(CSV_PATH, nrows=13 * N_EXPS)
print(f"Loaded {df.shape[0]} rows")

assert all(df.columns[2:].str.endswith("[ppb]")), (
    "Unexpected non-ppb columns after index 2 — check CSV structure"
)

ignore        = ["H2O", "O2", "HNO3", "CO", "H2"]
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

# scale (identical to smog_model.py)
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

# plain feedforward MLP — same input/output width as the KAN's [n_species, 16, 16, n_species]
class FFNN(nn.Module):
    def __init__(self, n_species, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_species, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_species),
        )
    def forward(self, x):
        return self.net(x)

model = FFNN(n_species).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"\nTraining FFNN for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, Y_train_t)
    loss.backward()
    opt.step()

    if (epoch + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(X_test_t), Y_test_t).item()
        print(f"  epoch {epoch+1:4d}: train_loss={loss.item():.6f}  test_loss={test_loss:.6f}")

# evaluate (identical metric to smog_model.py's eval_model)
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

np.savez(
    "C:/kan-project/model/ffnn_predictions.npz",
    predictions=preds_ppb, actuals=actual, species=np.array(species_names), rmse_per=rmse_per,
)
print("Saved predictions to C:/kan-project/model/ffnn_predictions.npz")

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kan import KAN

# Data
CSV_PATH    = "/Users/baron/Documents/KAN testing/data/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv"
SAVE_PATH   = "/Users/baron/Documents/KAN testing/predictions.npz"
N_EXPS      = 10000
TRAIN_SPLIT = 0.9
DT_MINUTES  = 5        # timestep between rows within one experiment

SEED        = 42
STEPS_ADAM  = 2000
STEPS_LBFGS = 100
LR          = 1e-4
BATCH_SIZE  = 1024
LAMB        = 0        

device = torch.device('cpu')

# Load
df = pd.read_csv(CSV_PATH, nrows=13 * N_EXPS)
print(f"Loaded {df.shape[0]} rows")

print("First 5 columns:", df.columns[:5].tolist())
assert all(df.columns[2:].str.endswith("[ppb]")), (
    "Unexpected non-ppb columns after index 2 — check CSV structure before proceeding"
)

ignore        = ["H2O", "O2", "HNO3", "CO", "H2"]
active_cols   = [c for c in df.columns
                 if c.split(" ")[0] not in ignore and c.endswith("[ppb]")]
species_names = [c.split(" ")[0] for c in active_cols]

all_ppb_cols   = list(df.iloc[:, 2:].columns)
active_indices = [all_ppb_cols.index(c) for c in active_cols]

C_active = df[active_cols].values
C_all    = df.iloc[:, 2:].values

N       = df.shape[0]
n_pts   = 13
n_steps = 12

assert df.shape[0] == n_pts * N_EXPS, (
    f"Expected {n_pts * N_EXPS} rows, got {df.shape[0]}. "
    f"Boundary deletion indices will be wrong."
)

D_raw    = np.diff(C_all, axis=0)
D_clean  = np.delete(D_raw,    list(range(n_steps, N - 1, n_pts)), axis=0)
X_clean  = np.delete(C_active, list(range(n_steps, N,     n_pts)), axis=0)

expected_rows = n_steps * N_EXPS
assert D_clean.shape[0] == expected_rows, f"D_clean shape wrong: {D_clean.shape}"
assert X_clean.shape[0] == expected_rows, f"X_clean shape wrong: {X_clean.shape}"
assert D_clean.shape[0] == X_clean.shape[0], "X and D row counts don't match"

D_active = D_clean[:, active_indices]  # [120000, 11] tendencies for active species

print(f"X: {X_clean.shape}  |  D: {D_active.shape}")

# Scale
scalerX  = StandardScaler()
scalerY  = StandardScaler()
X_scaled = scalerX.fit_transform(X_clean)
Y_scaled = scalerY.fit_transform(D_active)

n_train  = int(X_scaled.shape[0] * TRAIN_SPLIT)
X_train, Y_train = X_scaled[:n_train], Y_scaled[:n_train]
X_test,  Y_test  = X_scaled[n_train:], Y_scaled[n_train:]

def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

dataset = {
    'train_input': t(X_train),
    'train_label': t(Y_train),
    'test_input':  t(X_test),
    'test_label':  t(Y_test),
}

print(f"train_input: {dataset['train_input'].shape}")
print(f"train_label: {dataset['train_label'].shape}")

n_species = len(species_names)


KAN_WIDTH = [n_species, 32, n_species]

# Model
torch.manual_seed(SEED)
model = KAN(width=KAN_WIDTH, grid=5, k=3, seed=SEED, device=device)
print(f"Architecture: {KAN_WIDTH}")

with torch.no_grad():
    loss_before = torch.mean(
        (model(dataset['train_input']) - dataset['train_label']) ** 2
    ).item()

print(f"\nLoss BEFORE: {loss_before:.6f}")
print(f"Dumb baseline: {torch.var(dataset['train_label']).item():.6f}")

# train1
results_adam = model.fit(dataset, opt="Adam", steps=STEPS_ADAM, lamb=LAMB, lr=LR, batch=BATCH_SIZE)
test_key = 'test_loss' if 'test_loss' in results_adam else 'val_loss'
for i, (tl, vl) in enumerate(zip(results_adam['train_loss'], results_adam[test_key]), 1):
    print(f"[Adam]  Epoch {i:4d} | train: {tl:.6f} | test: {vl:.6f}")

results_lbfgs = model.fit(dataset, opt="LBFGS", steps=STEPS_LBFGS, lamb=LAMB)
test_key = 'test_loss' if 'test_loss' in results_lbfgs else 'val_loss'
for i, (tl, vl) in enumerate(zip(results_lbfgs['train_loss'], results_lbfgs[test_key]), 1):
    print(f"[LBFGS] Epoch {i:4d} | train: {tl:.6f} | test: {vl:.6f}")

# Eval
with torch.no_grad():
    preds_scaled = model(dataset['test_input'])
    loss_after   = torch.mean((preds_scaled - dataset['test_label']) ** 2).item()

print(f"\nLoss AFTER  (test): {loss_after:.6f}")
print(f"RMSE        (test): {loss_after**0.5:.6f}")

preds_ppb  = scalerY.inverse_transform(preds_scaled.numpy())
actual_ppb = scalerY.inverse_transform(dataset['test_label'].numpy())

rmse_std = loss_after ** 0.5
print(f"\n--- Standardized space ---")
print(f"RMSE (std): {rmse_std:.6f}")

rmse_ppb_per_species = np.sqrt(np.mean((preds_ppb - actual_ppb) ** 2, axis=0))
print(f"\n--- PPB space (per species) ---")
for name, rmse in zip(species_names, rmse_ppb_per_species):
    print(f"  {name:10s}: {rmse:.4f} ppb/step")
print(f"  {'MEAN':10s}: {rmse_ppb_per_species.mean():.4f} ppb/step")

# Save
np.savez(SAVE_PATH,
         predictions=preds_ppb,
         actuals=actual_ppb,
         species=np.array(species_names))
print(f"Saved predictions to {SAVE_PATH}")

model.plot(in_vars=species_names, out_vars=species_names)
plt.savefig("/Users/baron/Desktop/kan_graph.png", dpi=150, bbox_inches="tight")
print("Saved KAN graph to Desktop/kan_graph.png")


# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False
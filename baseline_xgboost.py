import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

# This evaluates Sturm & Silva's actual trained XGBoost emulator
# (Zenodo 10.5281/zenodo.13385987, smogforest_new.json) on its own proper
# held-out test set, replicating their exact preprocessing so the model
# receives inputs scaled exactly as it was trained on.
#
# NOTE: this is NOT the same test split our KAN was evaluated on — the KAN
# was trained/tested on a 10,000-experiment subset (80/20 split), while this
# model was trained on the full 1,000,000-experiment dataset (90/10 split,
# per the paper). Both are still valid held-out evaluations since all
# experiments are drawn i.i.d. from the same random initial-condition
# distribution, but the numbers are a reproduction of their reported
# baseline, not a row-for-row identical comparison to our KAN's test set.

CSV_PATH   = "C:/kan-project/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv"
MODEL_PATH = "C:/kan-project/model/sturm_silva/smogforest_new.json"

n_pts, n_steps = 13, 12
N_EXPS_TOTAL = 1_000_000
TRAIN_EXPS   = 900_000
TEST_EXPS    = 100_000

ignore_species = ['H2O', 'O2', 'HNO3', 'CO', 'H2']

header   = pd.read_csv(CSV_PATH, nrows=0)
ppb_cols = [c for c in header.columns if c.endswith('[ppb]')]
assert len(ppb_cols) == 16
active_cols    = [c for c in ppb_cols if c.split(' ')[0] not in ignore_species]
active_idx     = [ppb_cols.index(c) for c in active_cols]
species_all    = [c.split(' ')[0] for c in ppb_cols]
species_active = [c.split(' ')[0] for c in active_cols]
n_all, n_active = len(ppb_cols), len(active_cols)
print(f"All species ({n_all}): {species_all}")
print(f"Active/input species ({n_active}): {species_active}")

# --- pass 1: chunked min/max over the TRAINING experiments only, to
# reconstruct the exact MinMaxScaler stats used when the model was trained,
# without loading all 11.7M training rows into memory at once ---
CHUNK_EXPS = 50_000
chunk_rows = CHUNK_EXPS * n_pts
train_rows = TRAIN_EXPS * n_pts

x_min = np.full(n_active, np.inf)
x_max = np.full(n_active, -np.inf)
y_min = np.full(n_all, np.inf)
y_max = np.full(n_all, -np.inf)

rows_read = 0
print(f"\nScanning {TRAIN_EXPS} training experiments in chunks of {CHUNK_EXPS}...")
for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_rows):
    if rows_read >= train_rows:
        break
    C_chunk = chunk.iloc[:, 2:].values
    del_x = list(range(n_steps, C_chunk.shape[0], n_pts))
    X_chunk = np.delete(C_chunk[:, active_idx], del_x, axis=0)
    D_raw = np.diff(C_chunk, axis=0)
    del_d = list(range(n_steps, C_chunk.shape[0] - 1, n_pts))
    D_chunk = np.delete(D_raw, del_d, axis=0)

    x_min = np.minimum(x_min, X_chunk.min(axis=0))
    x_max = np.maximum(x_max, X_chunk.max(axis=0))
    y_min = np.minimum(y_min, D_chunk.min(axis=0))
    y_max = np.maximum(y_max, D_chunk.max(axis=0))

    rows_read += C_chunk.shape[0]
    print(f"  scanned {rows_read}/{train_rows} rows", end="\r")

print(f"\nDone scanning training set for scaler stats.")

# --- pass 2: load only the held-out test experiments (last 10%) ---
test_start_row = TRAIN_EXPS * n_pts
print(f"\nLoading {TEST_EXPS} test experiments (rows {test_start_row}:{test_start_row + TEST_EXPS*n_pts})...")
df_test = pd.read_csv(
    CSV_PATH,
    skiprows=range(1, test_start_row + 1),
    nrows=TEST_EXPS * n_pts,
)
C_test_all = df_test.iloc[:, 2:].values
del_x = list(range(n_steps, C_test_all.shape[0], n_pts))
X_test_raw = np.delete(C_test_all[:, active_idx], del_x, axis=0)
D_raw = np.diff(C_test_all, axis=0)
del_d = list(range(n_steps, C_test_all.shape[0] - 1, n_pts))
Y_test_raw = np.delete(D_raw, del_d, axis=0)
print(f"X_test: {X_test_raw.shape} | Y_test: {Y_test_raw.shape}")

# manual MinMaxScaler transform (feature_range=(0,1), matching sklearn default)
X_test_scaled = (X_test_raw - x_min) / (x_max - x_min)

print(f"\nLoading trained model: {MODEL_PATH}")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

preds_scaled = model.predict(X_test_scaled)
preds_ppb = preds_scaled * (y_max - y_min) + y_min

rmse_per = np.sqrt(np.mean((preds_ppb - Y_test_raw) ** 2, axis=0))
r2_per   = np.array([r2_score(Y_test_raw[:, i], preds_ppb[:, i]) for i in range(n_all)])

print(f"\nSturm & Silva XGBoost — per-species test performance (all {n_all} species):")
for name, rmse, r2 in zip(species_all, rmse_per, r2_per):
    flag = "  <-- overlaps with our KAN" if name in species_active else ""
    print(f"  {name:10s}: RMSE={rmse:.6f} ppb/step  R2={r2:.4f}{flag}")

overlap_idx = [ppb_cols.index(c) for c in active_cols]
print(f"\nMean RMSE over the {n_active} species our KAN also predicts: "
      f"{rmse_per[overlap_idx].mean():.4f} ppb/step")

np.savez(
    "C:/kan-project/model/sturm_silva_xgboost_predictions.npz",
    predictions=preds_ppb, actuals=Y_test_raw,
    species=np.array(species_all), rmse_per=rmse_per, r2_per=r2_per,
)
print("Saved predictions to C:/kan-project/model/sturm_silva_xgboost_predictions.npz")

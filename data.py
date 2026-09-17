import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib

import config as cfg


def load_and_clean_data():
    df = pd.read_csv(cfg.CSV_PATH, nrows=13 * cfg.N_EXPS)
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

    assert df.shape[0] == n_pts * cfg.N_EXPS, (
        f"Expected {n_pts * cfg.N_EXPS} rows, got {df.shape[0]}."
    )

    D_raw    = np.diff(C_all, axis=0)
    D_clean  = np.delete(D_raw,    list(range(n_steps, N - 1, n_pts)), axis=0)
    X_clean  = np.delete(C_active, list(range(n_steps, N,     n_pts)), axis=0)

    assert D_clean.shape[0] == n_steps * cfg.N_EXPS
    assert X_clean.shape[0] == D_clean.shape[0]

    D_active  = D_clean[:, active_indices]
    n_species = len(species_names)
    print(f"Species ({n_species}): {species_names}")
    print(f"X: {X_clean.shape}  |  D: {D_active.shape}")

    return X_clean, D_active, species_names, n_species


def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(cfg.device)


def scale_and_split(X_clean, D_active):
    X_log   = np.log1p(X_clean)
    n_train = int(X_log.shape[0] * cfg.TRAIN_SPLIT)

    # fit on train only: fitting before the split leaks test mean/std into the
    # training inputs, which makes the reported test score optimistic
    scalerX = StandardScaler().fit(X_log[:n_train])
    scalerY = StandardScaler().fit(D_active[:n_train])

    X_train, Y_train = scalerX.transform(X_log[:n_train]), scalerY.transform(D_active[:n_train])
    X_test,  Y_test  = scalerX.transform(X_log[n_train:]), scalerY.transform(D_active[n_train:])
    print(X_train.shape)

    dataset = {
        'train_input': t(X_train), 'train_label': t(Y_train),
        'test_input':  t(X_test),  'test_label':  t(Y_test),
    }
    print(f"train: {dataset['train_input'].shape} | test: {dataset['test_input'].shape}")

    return scalerX, scalerY, X_train, Y_train, X_test, Y_test, dataset


def save_splits(X_train, Y_train, X_test, Y_test, species_names, species_weight, scalerY, scalerX):
    # saved so baselines.py trains on the same data/weights — keep these keys
    # and paths in sync with baselines.py's np.load(SPLITS_PATH) / scalerY load
    np.savez(
        cfg.SPLITS_PATH,
        X_train=X_train, Y_train=Y_train,
        X_test=X_test,   Y_test=Y_test,
        species=np.array(species_names),
        species_weight=species_weight.cpu().numpy(),
    )
    joblib.dump(scalerY, cfg.SCALER_Y_PATH)
    # scalerX is needed to map symbolic formulas back to ppb, so keep it too
    joblib.dump(scalerX, cfg.SCALER_X_PATH)
    print(f"Saved splits + scalers to {os.path.dirname(cfg.SPLITS_PATH)}")


def save_predictions(preds, actual_ppb, species_names):
    np.savez(cfg.SAVE_PATH, predictions=preds, actuals=actual_ppb, species=np.array(species_names))
    print(f"Saved predictions to {cfg.SAVE_PATH}")

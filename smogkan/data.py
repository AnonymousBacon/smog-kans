import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# species the 11-species runs left out, needed back for conservation.
FIXED_SPECIES = ("HNO3", "CO", "H2", "H2O", "O2")
N_STEPS_PER_EXP = 12
N_POINTS_PER_EXP = N_STEPS_PER_EXP + 1


# reads the first n_exps experiments of the full csv; returns the 11 active species at each step and their change.
def load_csv(csv_path, n_exps):
    df = pd.read_csv(csv_path, nrows=N_POINTS_PER_EXP * n_exps)
    print(f"Loaded {df.shape[0]} rows")
    if not all(df.columns[2:].str.endswith("[ppb]")):
        raise ValueError("unexpected non-ppb columns after index 2, check the csv structure")
    if df.shape[0] != N_POINTS_PER_EXP * n_exps:
        raise ValueError(f"expected {N_POINTS_PER_EXP * n_exps} rows, got {df.shape[0]}")

    active_cols = [c for c in df.columns if c.split(" ")[0] not in FIXED_SPECIES and c.endswith("[ppb]")]
    species = [c.split(" ")[0] for c in active_cols]
    all_ppb_cols = list(df.columns[2:])
    active_idx = [all_ppb_cols.index(c) for c in active_cols]

    C_active = df[active_cols].values
    C_all = df.iloc[:, 2:].values
    n = df.shape[0]
    # np.diff runs across experiment boundaries, so the row from one experiment's last point to the next's first is dropped.
    D = np.delete(np.diff(C_all, axis=0), list(range(N_STEPS_PER_EXP, n - 1, N_POINTS_PER_EXP)), axis=0)
    X = np.delete(C_active, list(range(N_STEPS_PER_EXP, n, N_POINTS_PER_EXP)), axis=0)
    D = D[:, active_idx]
    print(f"Species ({len(species)}): {species}")
    print(f"X: {X.shape}  |  D: {D.shape}")
    return X, D, species


def load_raw(csv_path, n_exps):
    return load_csv(csv_path, n_exps)


def load_npz(path, all_species=False):
    d = np.load(path, allow_pickle=True)
    species = [str(s) for s in d["species"]]
    splits = (d["train_ids"], d["val_ids"], d["test_ids"])
    X, D = d["X_raw"], d["D_raw"]
    if all_species:
        X, D, species = load_all_species(path, d["source_exp_number"])
    n_exps = X.shape[0] // N_STEPS_PER_EXP
    print(f"loaded {os.path.basename(path)}: {n_exps} experiments, {X.shape[0]} rows")
    print(f"Species ({len(species)}): {species}")
    return X, D, species, splits, n_exps


# rebuilds all 16 species from the sampled-experiments csv in the same order as the npz split.
def load_all_species(npz_path, exp_numbers):
    csv = os.path.join(os.path.dirname(npz_path), "baseline_10k_experiments.csv")
    df = pd.read_csv(csv)
    cols = [c for c in df.columns if c.endswith("[ppb]")]
    C = df[cols].to_numpy().reshape(-1, N_STEPS_PER_EXP + 1, len(cols))
    if not np.array_equal(df["Experiment number"].to_numpy()[::N_STEPS_PER_EXP + 1], exp_numbers):
        raise ValueError(f"{csv} is not in the npz's experiment order")
    X = C[:, :-1].reshape(-1, len(cols))
    D = np.diff(C, axis=1).reshape(-1, len(cols))
    return X, D, [c.split()[0] for c in cols]


def split_experiments(n_exps, seed, frac=(0.7, 0.15, 0.15)):
    ids = np.random.default_rng(seed).permutation(n_exps)
    n_tr = int(frac[0] * n_exps)
    n_va = int(frac[1] * n_exps)
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def rows_for(exp_ids):
    return (exp_ids[:, None] * N_STEPS_PER_EXP + np.arange(N_STEPS_PER_EXP)).ravel()


def drop_outlier_exps(X_log, train_ids, val_ids, test_ids, max_z):
    probe = StandardScaler().fit(X_log[rows_for(train_ids)])
    z = np.abs(probe.transform(X_log)).reshape(-1, N_STEPS_PER_EXP, X_log.shape[1]).max(axis=(1, 2))
    kept = tuple(ids[z[ids] <= max_z] for ids in (train_ids, val_ids, test_ids))
    dropped = [int(len(a) - len(b)) for a, b in zip((train_ids, val_ids, test_ids), kept)]
    return kept, dropped


def transform_inputs(X, train_rows, kind):
    if kind == "log1p":
        return np.log1p(X)
    if kind == "log10_floor":
        Xt = X[train_rows]
        floor = np.percentile(Xt, 1, axis=0)
        # species that start at exactly zero take their floor from their positive values instead.
        for j in np.where(floor <= 1e-9)[0]:
            pos = Xt[:, j][Xt[:, j] > 0]
            floor[j] = np.percentile(pos, 1) if len(pos) else 1e-9
        return np.log10(X + np.maximum(floor, 1e-9))
    raise ValueError(f"unknown transform {kind!r}")


def build_dataset(X, D, train_ids, val_ids, test_ids, device, dtype, max_z=0.0,
                  transform="log1p"):
    X_log = transform_inputs(X, rows_for(train_ids), transform)
    dropped = [0, 0, 0]
    if max_z:
        (train_ids, val_ids, test_ids), dropped = drop_outlier_exps(
            X_log, train_ids, val_ids, test_ids, max_z)
    tr, va, te = rows_for(train_ids), rows_for(val_ids), rows_for(test_ids)
    scaler_x = StandardScaler().fit(X_log[tr])
    scaler_y = StandardScaler().fit(D[tr])
    # prints any input that lands past 6 sigma or is badly skewed, which is how OH's bad transform was found.
    z = scaler_x.transform(X_log[tr])
    skew = np.mean(z ** 3, axis=0)
    if np.abs(z).max() > 6 or np.abs(skew).max() > 3:
        print("  input check (max|z|, skew): "
              + ", ".join(f"{j}:{np.abs(z[:, j]).max():.1f}/{skew[j]:.1f}" for j in range(z.shape[1])))

    def t(a):
        return torch.tensor(a, dtype=dtype, device=device)

    ds = {}
    for split, rows in (("train", tr), ("val", va), ("test", te)):
        ds[f"{split}_input"] = t(scaler_x.transform(X_log[rows]))
        ds[f"{split}_label"] = t(scaler_y.transform(D[rows]))
    return ds, scaler_y, dropped

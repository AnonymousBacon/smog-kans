import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kan import KAN
import os

# paths
CSV_PATH  = "/Users/baron/Documents/KAN testing/data/experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv"
SAVE_PATH = "/Users/baron/Documents/KAN testing/predictions.npz"
FIG_DIR   = "/Users/baron/Documents/KAN testing/figures/"
CKPT_PATH = "/Users/baron/Documents/KAN testing/model/smog_kan"

os.makedirs(FIG_DIR, exist_ok=True)

# config
N_EXPS               = 10000
TRAIN_SPLIT          = 0.9
SEED                 = 42
STEPS_ADAM           = 2000
STEPS_LBFGS          = 75
LR_ADAM              = 1e-4
BATCH_SIZE           = 1024
LAMB                 = 0
LOAD_FROM_CHECKPOINT = False   # set True to skip training and load saved model

device = torch.device('cpu')

# load and clean
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

D_raw    = np.diff(C_all, axis=0)
D_clean  = np.delete(D_raw,    list(range(n_steps, N - 1, n_pts)), axis=0)
X_clean  = np.delete(C_active, list(range(n_steps, N,     n_pts)), axis=0)

assert D_clean.shape[0] == n_steps * N_EXPS
assert X_clean.shape[0] == D_clean.shape[0]

D_active  = D_clean[:, active_indices]
n_species = len(species_names)
print(f"Species ({n_species}): {species_names}")
print(f"X: {X_clean.shape}  |  D: {D_active.shape}")

# distribution plots
def plot_distributions(X_raw, Y_raw, names):
    configs = [
        ("Raw (ppb)",      X_raw,                               Y_raw),
        ("MinMaxScaler",   MinMaxScaler().fit_transform(X_raw),  MinMaxScaler().fit_transform(Y_raw)),
        ("StandardScaler", StandardScaler().fit_transform(X_raw), StandardScaler().fit_transform(Y_raw)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Input/Output Distributions: Raw vs Scalers", fontsize=13)

    for row, (label, X_s, Y_s) in enumerate(configs):
        for col, (data, title) in enumerate([(X_s, f"X — {label}"), (Y_s, f"Y (tendencies) — {label}")]):
            ax = axes[row, col]
            ax.boxplot(data, labels=names, patch_artist=True,
                       flierprops=dict(marker='.', markersize=1, alpha=0.3))
            ax.set_title(title)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(FIG_DIR + "distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/distributions.png")

plot_distributions(np.log1p(X_clean), D_active, species_names)

# scale
scalerX  = StandardScaler()
scalerY  = StandardScaler()
X_scaled = scalerX.fit_transform(np.log1p(X_clean))
Y_scaled = scalerY.fit_transform(D_active)

n_train          = int(X_scaled.shape[0] * TRAIN_SPLIT)
X_train, Y_train = X_scaled[:n_train], Y_scaled[:n_train]
X_test,  Y_test  = X_scaled[n_train:], Y_scaled[n_train:]

def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

dataset = {
    'train_input': t(X_train), 'train_label': t(Y_train),
    'test_input':  t(X_test),  'test_label':  t(Y_test),
}
print(f"train: {dataset['train_input'].shape} | test: {dataset['test_input'].shape}")

# training
KAN_WIDTH = [n_species, 16, 16, n_species]
print(f"\nArchitecture: {KAN_WIDTH}")

def make_model():
    torch.manual_seed(SEED)
    m = KAN(width=KAN_WIDTH, grid=3, k=3, seed=SEED, device=device, auto_save=False)
    m.speed()
    return m

def loss_key(results):
    return 'test_loss' if 'test_loss' in results else 'val_loss'

if LOAD_FROM_CHECKPOINT:
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model_both  = KAN.loadckpt(CKPT_PATH)
    skip_plots  = True
else:
    print("\n── Adam only ──")
    model_adam  = make_model()
    res_adam    = model_adam.fit(dataset, opt="Adam",  steps=STEPS_ADAM,  lamb=LAMB,
                                 lr=LR_ADAM, batch=BATCH_SIZE)

    print("\n── LBFGS only ──")
    model_lbfgs = make_model()
    res_lbfgs   = model_lbfgs.fit(dataset, opt="LBFGS", steps=STEPS_LBFGS, lamb=LAMB, batch=-1)

    print("\n── Adam → LBFGS ──")
    model_both  = make_model()
    r1 = model_both.fit(dataset, opt="Adam",  steps=STEPS_ADAM,  lamb=LAMB,
                        lr=LR_ADAM, batch=BATCH_SIZE)
    r2 = model_both.fit(dataset, opt="LBFGS", steps=STEPS_LBFGS, lamb=LAMB, batch=-1)

    adam_transition = len(r1['train_loss'])
    res_both = {
        'train_loss': r1['train_loss'] + r2['train_loss'],
        'test_loss':  r1[loss_key(r1)] + r2[loss_key(r2)],
    }

    # grid refinement: 3 → 10 → 20
    refine_transitions = []
    for new_grid in [10, 20]:
        refine_transitions.append(len(res_both['train_loss']))
        print(f"\n── refine grid → {new_grid} ──")
        model_both = model_both.refine(new_grid)
        r_ref = model_both.fit(dataset, opt="LBFGS", steps=100, lamb=0, batch=-1)
        res_both['train_loss'] += r_ref['train_loss']
        res_both['test_loss']  += r_ref[loss_key(r_ref)]

    model_both.saveckpt(CKPT_PATH)
    print(f"Checkpoint saved → {CKPT_PATH}_*")
    print("To skip retraining next run: set LOAD_FROM_CHECKPOINT = True")
    skip_plots = False

# loss curves
if not skip_plots:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Train vs Test Loss (log scale) — Overfitting Check", fontsize=13)

    for ax, train, test, title, vlines in [
        (axes[0], res_adam['train_loss'],  res_adam[loss_key(res_adam)],   "Adam Only",              []),
        (axes[1], res_lbfgs['train_loss'], res_lbfgs[loss_key(res_lbfgs)], "LBFGS Only",             []),
        (axes[2], res_both['train_loss'],  res_both['test_loss'],           "Adam → LBFGS + refine",
         [(adam_transition, 'LBFGS')] + [(s, f'grid→{g}') for s, g in zip(refine_transitions, [10, 20])]),
    ]:
        ax.semilogy(train, label='train')
        ax.semilogy(test,  label='test', linestyle='--')
        for x, label in vlines:
            ax.axvline(x=x, color='gray', linestyle=':', linewidth=1, label=label)
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR + "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/loss_curves.png")

# evaluate
def eval_model(model):
    with torch.no_grad():
        preds_scaled = model(dataset['test_input'])
    mse       = torch.mean((preds_scaled - dataset['test_label']) ** 2).item()
    preds_ppb = scalerY.inverse_transform(preds_scaled.numpy())
    actual    = scalerY.inverse_transform(dataset['test_label'].numpy())
    rmse_per  = np.sqrt(np.mean((preds_ppb - actual) ** 2, axis=0))
    return mse, rmse_per, preds_ppb, actual

mse_both, rmse_both, preds_both, actual_ppb = eval_model(model_both)

if not skip_plots:
    mse_adam,  rmse_adam,  _, _ = eval_model(model_adam)
    mse_lbfgs, rmse_lbfgs, _, _ = eval_model(model_lbfgs)

    print(f"\n{'Optimizer':<15} {'Test MSE':>10} {'Mean RMSE (ppb)':>16}")
    print("-" * 44)
    for name, mse, rmse in [
        ("Adam",       mse_adam,  rmse_adam),
        ("LBFGS",      mse_lbfgs, rmse_lbfgs),
        ("Adam→LBFGS", mse_both,  rmse_both),
    ]:
        print(f"{name:<15} {mse:>10.6f} {rmse.mean():>16.4f}")

    x, w = np.arange(n_species), 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, rmse_adam,  w, label='Adam')
    ax.bar(x,     rmse_lbfgs, w, label='LBFGS')
    ax.bar(x + w, rmse_both,  w, label='Adam→LBFGS')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right')
    ax.set_ylabel("RMSE (ppb/step)")
    ax.set_title("Per-Species RMSE: Optimizer Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR + "rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/rmse_comparison.png")
else:
    print(f"\nTest MSE: {mse_both:.6f} | Mean RMSE: {rmse_both.mean():.4f} ppb/step")
    for name, rmse in zip(species_names, rmse_both):
        print(f"  {name:10s}: {rmse:.4f} ppb/step")

# save predictions and KAN graph
np.savez(SAVE_PATH, predictions=preds_both, actuals=actual_ppb, species=np.array(species_names))
print(f"Saved predictions to {SAVE_PATH}")

model_both.plot(in_vars=species_names, out_vars=species_names)
plt.savefig("/Users/baron/Desktop/kan_graph.png", dpi=150, bbox_inches="tight")
print("Saved KAN graph to Desktop/kan_graph.png")

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
CKPT_PATH      = "/Users/baron/Documents/KAN testing/model/smog_kan"
BEST_CKPT_PATH = "/Users/baron/Documents/KAN testing/model/smog_kan_best"

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
ADAM_CHUNK           = 200     # save best every N adam steps
LBFGS_CHUNK          = 25      # save best every N lbfgs steps
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
def plot_distributions(X_log, Y_raw, names):
    configs = [
        ("log1p (input)",  X_log,                                Y_raw),
        ("MinMaxScaler",   MinMaxScaler().fit_transform(X_log),   MinMaxScaler().fit_transform(Y_raw)),
        ("StandardScaler", StandardScaler().fit_transform(X_log), StandardScaler().fit_transform(Y_raw)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Input/Output Distributions: Raw vs Scalers", fontsize=13)

    for row, (label, X_s, Y_s) in enumerate(configs):
        for col, (data, title) in enumerate([(X_s, f"X — {label}"), (Y_s, f"Y (tendencies) — {label}")]):
            ax = axes[row, col]
            ax.boxplot(data, tick_labels=names, patch_artist=True,
                       flierprops=dict(marker='.', markersize=1, alpha=0.3))
            ax.set_title(title)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel("Value")
            if row == 2:
                p1, p99 = np.percentile(data, [1, 99])
                pad = (p99 - p1) * 0.5
                ax.set_ylim(p1 - pad, p99 + pad)
                ax.set_title(f"{title} (axis clipped to 1–99th pct)")

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

def fit_chunked(model, dataset, opt, total_steps, chunk_size, best_val, **kwargs):
    train_hist, test_hist = [], []
    steps_done = 0
    while steps_done < total_steps:
        s = min(chunk_size, total_steps - steps_done)
        res = model.fit(dataset, opt=opt, steps=s, **kwargs)
        tk = loss_key(res)
        train_hist += res['train_loss']
        test_hist  += res[tk]
        val = res[tk][-1]
        if val < best_val:
            best_val = val
            model.saveckpt(BEST_CKPT_PATH)
            print(f"  best val loss: {val:.6f} → saved")
        steps_done += s
    return train_hist, test_hist, best_val

if LOAD_FROM_CHECKPOINT:
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = KAN.loadckpt(CKPT_PATH)
else:
    model    = make_model()
    best_val = float('inf')

    tl, vl, best_val = fit_chunked(model, dataset, "Adam", STEPS_ADAM, ADAM_CHUNK,
                                   best_val, lamb=LAMB, lr=LR_ADAM, batch=BATCH_SIZE)
    adam_transition = len(tl)

    tl2, vl2, best_val = fit_chunked(model, dataset, "LBFGS", STEPS_LBFGS, LBFGS_CHUNK,
                                     best_val, lamb=LAMB, batch=-1)
    train_loss = tl + tl2
    test_loss  = vl + vl2

    # grid refinement: 3 → 10 → 20
    refine_transitions = []
    for new_grid in [10, 20]:
        refine_transitions.append(len(train_loss))
        print(f"\n── refine grid → {new_grid} ──")
        model.save_act = True
        with torch.no_grad():
            model(dataset['train_input'])
        model = model.refine(new_grid)
        tl_r, vl_r, best_val = fit_chunked(model, dataset, "LBFGS", 100, LBFGS_CHUNK,
                                            best_val, lamb=0, batch=-1)
        train_loss += tl_r
        test_loss  += vl_r

    model.saveckpt(CKPT_PATH)
    print(f"Checkpoint saved → {CKPT_PATH}_*")
    print("To skip retraining next run: set LOAD_FROM_CHECKPOINT = True")

    np.savez(
        "/Users/baron/Documents/KAN testing/model/loss_history.npz",
        train_loss=train_loss,
        test_loss=test_loss,
        adam_transition=adam_transition,
        refine_transitions=refine_transitions,
    )

# loss curves
_hist_path = "/Users/baron/Documents/KAN testing/model/loss_history.npz"
if os.path.exists(_hist_path):
    _h = np.load(_hist_path)
    train_loss       = list(_h["train_loss"])
    test_loss        = list(_h["test_loss"])
    adam_transition  = int(_h["adam_transition"])
    refine_transitions = list(_h["refine_transitions"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Train vs Test Loss (log scale)", fontsize=13)
    ax.semilogy(train_loss, label='train')
    ax.semilogy(test_loss,  label='test', linestyle='--')
    ax.axvline(x=adam_transition, color='gray', linestyle=':', linewidth=1, label='Adam→LBFGS')
    for step, g in zip(refine_transitions, [10, 20]):
        ax.axvline(x=step, color='orange', linestyle=':', linewidth=1, label=f'grid→{g}')
    ax.set_xlabel("Steps")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR + "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/loss_curves.png")
else:
    print("No loss history found — run training first to generate loss_curves.png")

# evaluate
def eval_model(model):
    with torch.no_grad():
        preds_scaled = model(dataset['test_input'])
    mse       = torch.mean((preds_scaled - dataset['test_label']) ** 2).item()
    preds_ppb = scalerY.inverse_transform(preds_scaled.numpy())
    actual    = scalerY.inverse_transform(dataset['test_label'].numpy())
    rmse_per  = np.sqrt(np.mean((preds_ppb - actual) ** 2, axis=0))
    return mse, rmse_per, preds_ppb, actual

mse, rmse_per, preds, actual_ppb = eval_model(model)

print(f"\nTest MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
for name, rmse in zip(species_names, rmse_per):
    print(f"  {name:10s}: {rmse:.4f} ppb/step")

x, w = np.arange(n_species), 0.4
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x, rmse_per, w)
ax.set_xticks(x)
ax.set_xticklabels(species_names, rotation=45, ha='right')
ax.set_ylabel("RMSE (ppb/step)")
ax.set_title("Per-Species RMSE")
plt.tight_layout()
plt.savefig(FIG_DIR + "rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/rmse.png")

# scatter: predicted vs actual per species
n_cols = 4
n_rows = -(-n_species // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes = axes.flatten()
for i, (name, ax) in enumerate(zip(species_names, axes)):
    ax.scatter(actual_ppb[:, i], preds[:, i], s=1, alpha=0.3)
    lims = [min(actual_ppb[:, i].min(), preds[:, i].min()),
            max(actual_ppb[:, i].max(), preds[:, i].max())]
    ax.plot(lims, lims, 'r--', linewidth=0.8)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
for ax in axes[n_species:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR + "scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/scatter.png")

# save predictions and KAN graph
np.savez(SAVE_PATH, predictions=preds, actuals=actual_ppb, species=np.array(species_names))
print(f"Saved predictions to {SAVE_PATH}")

model.plot(in_vars=species_names, out_vars=species_names)
plt.savefig("/Users/baron/Desktop/kan_graph.png", dpi=150, bbox_inches="tight")
print("Saved KAN graph to Desktop/kan_graph.png")

import sys
import functools
sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kan import KAN
import joblib
import os
import io
import contextlib
from symbolic_viz import parse_symbolic_log, plot_symbolic_snapping

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data", "experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv")
SAVE_PATH = os.path.join(BASE_DIR, "predictions.npz")
FIG_DIR   = os.path.join(BASE_DIR, "figures", "")
CKPT_PATH      = os.path.join(BASE_DIR, "model", "smog_kan")
BEST_CKPT_PATH = os.path.join(BASE_DIR, "model", "smog_kan_best")
SPLITS_PATH    = os.path.join(BASE_DIR, "model", "splits.npz")
SCALER_Y_PATH  = os.path.join(BASE_DIR, "model", "scalerY.pkl")
HIST_PATH      = os.path.join(BASE_DIR, "model", "loss_history.npz")
GRAPH_PATH     = os.path.join(BASE_DIR, "kan_graph.png")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

N_EXPS               = 10000
TRAIN_SPLIT          = 0.8
SEED                 = 42
LBFGS_CHUNK          = 25
PRINT_EVERY          = 100
LOAD_FROM_CHECKPOINT = False

WARMUP_STEPS    = 1200
SPARSIFY_STEPS  = 300
SPARSIFY_LAMB   = 1e-3
PRUNE_NODE_TH   = 1e-2
PRUNE_EDGE_TH   = 3e-2
REFINE_GRIDS    = [10, 20]
REFINE_STEPS    = 250

SYMBOLIC_R2_MIN     = 0.9
SYMBOLIC_FIT_STEPS  = 200

device = torch.device('cpu')

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

scalerX  = StandardScaler()
scalerY  = StandardScaler()
X_scaled = scalerX.fit_transform(np.log1p(X_clean))
Y_scaled = scalerY.fit_transform(D_active)

n_train          = int(X_scaled.shape[0] * TRAIN_SPLIT)
X_train, Y_train = X_scaled[:n_train], Y_scaled[:n_train]
X_test,  Y_test  = X_scaled[n_train:], Y_scaled[n_train:]
print(X_train.shape)

def t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)

dataset = {
    'train_input': t(X_train), 'train_label': t(Y_train),
    'test_input':  t(X_test),  'test_label':  t(Y_test),
}
print(f"train: {dataset['train_input'].shape} | test: {dataset['test_input'].shape}")

KAN_WIDTH = [n_species, 16, 16, 16, n_species]
print(f"\nArchitecture: {KAN_WIDTH}")

SPECIES_WEIGHTS = {"OH": 1.5}
species_weight = torch.tensor(
    [SPECIES_WEIGHTS.get(name, 1.0) for name in species_names],
    dtype=torch.float32, device=device,
)

def weighted_mse(pred, target):
    return torch.mean(species_weight * (pred - target) ** 2)

np.savez(
    SPLITS_PATH,
    X_train=X_train, Y_train=Y_train,
    X_test=X_test,   Y_test=Y_test,
    species=np.array(species_names),
    species_weight=species_weight.cpu().numpy(),
)
joblib.dump(scalerY, SCALER_Y_PATH)
print(f"Saved splits + scaler to {os.path.dirname(SPLITS_PATH)}")


def make_model():
    torch.manual_seed(SEED)
    m = KAN(width=KAN_WIDTH, grid=5, k=3, seed=SEED, device=device, auto_save=False, grid_eps=0.0)
    m.speed()
    return m

def loss_key(results):
    return 'test_loss' if 'test_loss' in results else 'val_loss'

def fit_chunked(model, dataset, opt, total_steps, chunk_size, best_val, print_every=PRINT_EVERY, **kwargs):
    kwargs.setdefault("update_grid", False)
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
        if steps_done % print_every == 0 or steps_done == total_steps:
            print(f"  step {steps_done:5d}/{total_steps}  train_loss={res['train_loss'][-1]:.6f}  test_loss={val:.6f}")
    return train_hist, test_hist, best_val

if LOAD_FROM_CHECKPOINT:
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = KAN.loadckpt(CKPT_PATH)
else:
    model    = make_model()
    best_val = float('inf')
    train_loss, test_loss = [], []

    print(f"\nWarmup: {WARMUP_STEPS} steps (speed mode)")
    stage_train, stage_test, best_val = fit_chunked(
        model, dataset, "LBFGS", WARMUP_STEPS, LBFGS_CHUNK,
        best_val, lamb=0, batch=-1, loss_fn=weighted_mse,
    )
    train_loss += stage_train
    test_loss  += stage_test

    model.save_act         = True
    model.symbolic_enabled = True
    print(f"\nSparsifying: {SPARSIFY_STEPS} steps (lamb={SPARSIFY_LAMB})")
    stage_train, stage_test, best_val = fit_chunked(
        model, dataset, "LBFGS", SPARSIFY_STEPS, LBFGS_CHUNK,
        best_val, lamb=SPARSIFY_LAMB, batch=-1, loss_fn=weighted_mse,
    )
    train_loss += stage_train
    test_loss  += stage_test

    print(f"\nPruning (node_th={PRUNE_NODE_TH}, edge_th={PRUNE_EDGE_TH})")
    print(f"  width before: {model.width}")
    model = model.prune(node_th=PRUNE_NODE_TH, edge_th=PRUNE_EDGE_TH)
    print(f"  width after:  {model.width}")
    model.save_act  = False
    model.auto_save = False

    for grid in REFINE_GRIDS:
        model.save_act = True
        model(dataset['train_input'])
        model = model.refine(grid)
        model.save_act = False

        print(f"\nGrid refine: grid={grid} ({REFINE_STEPS} steps)")
        stage_train, stage_test, best_val = fit_chunked(
            model, dataset, "LBFGS", REFINE_STEPS, LBFGS_CHUNK,
            best_val, lamb=0, batch=-1, loss_fn=weighted_mse,
        )
        train_loss += stage_train
        test_loss  += stage_test

    model.saveckpt(CKPT_PATH)
    print(f"Checkpoint saved → {CKPT_PATH}_*")
    print("To skip retraining next run: set LOAD_FROM_CHECKPOINT = True")

    np.savez(
        HIST_PATH,
        train_loss=train_loss,
        test_loss=test_loss,
    )

if os.path.exists(HIST_PATH):
    _h = np.load(HIST_PATH)
    train_loss = list(_h["train_loss"])
    test_loss  = list(_h["test_loss"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Train vs Test Loss (log scale)", fontsize=13)
    ax.semilogy(train_loss, label='train')
    ax.semilogy(test_loss,  label='test', linestyle='--')
    ax.set_xlabel("Steps")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR + "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/loss_curves.png")
else:
    print("No loss history found — run training first to generate loss_curves.png")

def eval_model(model):
    with torch.no_grad():
        preds_scaled = model(dataset['test_input'])
    mse       = torch.mean((preds_scaled - dataset['test_label']) ** 2).item()
    preds_ppb = scalerY.inverse_transform(preds_scaled.cpu().numpy())
    actual    = scalerY.inverse_transform(dataset['test_label'].cpu().numpy())
    rmse_per  = np.sqrt(np.mean((preds_ppb - actual) ** 2, axis=0))
    return mse, rmse_per, preds_ppb, actual

mse, rmse_per, preds, actual_ppb = eval_model(model)

print(f"\nTest MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
for name, rmse in zip(species_names, rmse_per):
    print(f"  {name:10s}: {rmse:.4f} ppb/step")

if np.isnan(mse):
    print("\nSkipping auto_symbolic: model is already producing NaN, retrain first")
else:
    rmse_before = rmse_per.copy()

    model.save_act         = True
    model.symbolic_enabled = True
    with torch.no_grad():
        model(dataset['train_input'])

    _sym_log     = io.StringIO()
    _real_stdout = sys.stdout

    class _Tee:
        def write(self, s):
            _real_stdout.write(s)
            _sym_log.write(s)
            return len(s)

        def flush(self):
            _real_stdout.flush()

    with contextlib.redirect_stdout(_Tee()):
        model.auto_symbolic(r2_threshold=SYMBOLIC_R2_MIN)

    model.fit(dataset, opt="LBFGS", steps=SYMBOLIC_FIT_STEPS, lamb=0, loss_fn=weighted_mse,
              update_grid=False)
    model.save_act = False

    plot_symbolic_snapping(
        parse_symbolic_log(_sym_log.getvalue()),
        [int(w) for w in model.width_in],
        species_names, SYMBOLIC_R2_MIN, FIG_DIR + "symbolic_snapping.png",
    )

    mse, rmse_per, preds, actual_ppb = eval_model(model)
    print(f"\nAfter auto_symbolic: Test MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
    for name, before, after in zip(species_names, rmse_before, rmse_per):
        flag = "  worse" if after > before else ""
        print(f"  {name:10s}: {before:.4f} -> {after:.4f} ppb/step{flag}")


n_cols = 4
n_rows = -(-n_species // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes = axes.flatten()
for i, (name, ax) in enumerate(zip(species_names, axes)):
    a, p = actual_ppb[:, i], preds[:, i]
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
plt.savefig(FIG_DIR + "scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/scatter.png")

np.savez(SAVE_PATH, predictions=preds, actuals=actual_ppb, species=np.array(species_names))
print(f"Saved predictions to {SAVE_PATH}")

model.save_act = True
with torch.no_grad():
    plot_idx = torch.randperm(dataset['train_input'].shape[0])[:8192]
    model(dataset['train_input'][plot_idx])

max_label_len = max(len(s) for s in species_names)
label_varscale = min(1.0, max(0.25, 19 / (n_species * max_label_len)))
try:
    model.plot(in_vars=species_names, out_vars=species_names, scale=1.0, varscale=label_varscale)
    plt.savefig(GRAPH_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved KAN graph to {GRAPH_PATH}")
except MemoryError as e:
    print(f"Skipped KAN graph: ran out of memory compositing it ({e}). "
          f"This is a pykan scaling limit at this network width, not a training/eval problem — "
          f"everything else already saved successfully.")

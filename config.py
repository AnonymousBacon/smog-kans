import os
import torch

# paths — relative to this script's location, so it runs regardless of clone path
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data", "experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv")
SAVE_PATH = os.path.join(BASE_DIR, "predictions.npz")
FIG_DIR   = os.path.join(BASE_DIR, "figures", "")
CKPT_PATH      = os.path.join(BASE_DIR, "model", "smog_kan")
BEST_CKPT_PATH = os.path.join(BASE_DIR, "model", "smog_kan_best")
SPLITS_PATH    = os.path.join(BASE_DIR, "model", "splits.npz")
SCALER_X_PATH  = os.path.join(BASE_DIR, "model", "scalerX.pkl")
SCALER_Y_PATH  = os.path.join(BASE_DIR, "model", "scalerY.pkl")
HIST_PATH      = os.path.join(BASE_DIR, "model", "loss_history.npz")
GRAPH_PATH     = os.path.join(BASE_DIR, "kan_graph.png")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# config
N_EXPS               = 10000
TRAIN_SPLIT          = 0.8
SEED                 = 42
LBFGS_CHUNK          = 25      # save best every N lbfgs steps
PRINT_EVERY          = 100     # periodic progress print during long lbfgs runs
LOAD_FROM_CHECKPOINT = False   # set True to skip training and load saved model

# training pipeline: warmup (fast) -> sparsify -> prune -> grid refine
WARMUP_STEPS    = 1200   # speed mode, no sparsification penalty
SPARSIFY_STEPS  = 300    # save_act on, l1+entropy penalty to concentrate signal onto few edges
SPARSIFY_LAMB   = 1e-3
PRUNE_NODE_TH   = 1e-2
PRUNE_EDGE_TH   = 3e-2
REFINE_GRIDS    = [10, 20]   # spline resolution stages after pruning
REFINE_STEPS    = 250        # lbfgs steps per refine stage

# symbolic fitting: snap well-fit edges to closed-form functions
SYMBOLIC_R2_MIN     = 0.9
SYMBOLIC_FIT_STEPS  = 200   # retune affine constants after snapping
SYMBOLIC_FIT_SAMPLES = 4000  # rows auto_symbolic sees; see the note in evaluate.py

# pykan's default library contains functions with poles (1/x^n, tan) and ones
# undefined for negative inputs (log, sqrt, x^0.5). inputs here are
# standardscaler'd, so activations are centred on zero and take both signs --
# every one of those is guaranteed to be evaluated at a point it blows up on.
# a snapped 1/x^2 turned the whole forward pass to nan the first time symbolic
# fitting ran to completion. these are the ones finite across all of R.
SYMBOLIC_LIB = ['0', 'x', 'x^2', 'x^3', 'x^4', 'x^5',
                'exp', 'sin', 'cos', 'tanh', 'gaussian', 'abs']

# per-species loss weighting
SPECIES_WEIGHTS = {"OH": 1.5} #change to much higher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

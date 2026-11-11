import os
import torch

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
SYMBOLIC_FIT_SAMPLES = 4000

SYMBOLIC_LIB = ['0', 'x', 'x^2', 'x^3', 'x^4', 'x^5',
                'exp', 'sin', 'cos', 'tanh', 'gaussian', 'abs']

SPECIES_WEIGHTS = {"OH": 1.5}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

import os

# repo root, one level above this package.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv")
NPZ_PATH = os.path.join(BASE_DIR, "data", "baseline_10k", "baseline_10k.npz")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

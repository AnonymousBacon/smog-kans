# v1 pipeline (frozen, in archive/v1/)

The first version of the smog KAN: it reads the full csv, uses an 80/20 row split and starts from `model.py`
(`smog_model.py` is the single-file version before that). `run_configs.py` and `smogkan/` replaced it.

These files are kept for reference and are not maintained. They import each other by name (`import config`,
`import data`), so to run one, `cd archive/v1` first. Their outputs (`model/splits.npz`, `predictions.npz`) are not
on the current `baseline_10k` experiment split.

- `archive/v1/figures/`, `archive/v1/gpu_run_results/`: plots and results from the v1 runs, including the September GPU run
- `baselines.py` (XGBoost), `baseline_ffnn.py` (FFNN): baselines on the v1 split; port to `baseline_10k.npz` before
  comparing them with current KAN runs
- `docs/CHANGES_v1.md`: change log for `smog_model.py`

# smog-kans

A Kolmogorov-Arnold network (pykan 0.2.8) that learns one 5-minute step of a 16-species, 13-reaction smog
mechanism: given the concentrations now, predict how much each species changes. An optional stoichiometric
constraint makes the KAN predict one flux per reaction and maps those to species changes through the reaction
matrix, so atoms are conserved.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The dataset is not in git. Put `baseline_10k.npz` and `baseline_10k_experiments.csv` in `data/baseline_10k/`.

## Run

Every experiment is a yaml in `configs/`: a `defaults:` block plus a list of `runs:`, each overriding some defaults.

```bash
python run_configs.py --config configs/stoich_overnight.yaml --data data/baseline_10k/baseline_10k.npz --runs S13_1200
```

- `--steps 6` makes a quick smoke run. It needs at least one step per stage; a stage is the first fit plus one per
  `refine` grid.
- `--device cuda` runs on a GPU. `./run_overnight.sh preflight` and `./run_overnight.sh start` run a whole config
  on a rented GPU box. The header of `run_overnight.sh` lists its options.
- Model checkpoints (`model_state`, `model_cache_data`, `model_config.yml`) and `history.npz` are not in git. After a
  GPU run, copy the results back before you stop or destroy the box:
  `rsync -az root@<box>:<repo>/runs/<config name>/ runs/<config name>/`, and the same for `logs/<config name>/`.
- Results go to `runs/<config name>/<device>/<run name>/`: `config.json`, `metrics.json`, `history.npz`, the model
  checkpoint and `loss_curves.png`, `scatter.png`, `rmse_r2.png`. Each batch also gets `summary.csv` and
  `summary.png`.
- `python compare_runs.py <run dir> <run dir> ...` writes `comparison.png` and prints a markdown table.
- Add `kan_graph: true` to a config to also draw the KAN graph. It is off by default because it is slow.

## Layout

| path | what |
|---|---|
| `run_configs.py` | entry point: trains every run in a config |
| `smogkan/data.py` | loading the npz or csv, the experiment split, input transforms, scaling |
| `smogkan/training.py` | model and LBFGS setup, the training loop with early stopping and safe ctrl-c |
| `smogkan/metrics.py` | test-set RMSE, R², atom residual, flux stats |
| `smogkan/plots.py` | every plot |
| `smogkan/runner.py` | `run()`: one config entry from data to saved results |
| `stoich.py` | the reaction matrix, `StoichHead`, and Daniel's fixed symbolic last layer |
| `kan_patches.py` | fixes for pykan 0.2.8 (rank-safe spline fit, skip empty symbolic layers) |
| `experiment.py`, `phase1.py` | old names, kept so existing imports and commands still work |
| `compare_runs.py` | compare finished runs |
| `analysis/` | one-off analyses (OH predictability) |
| `configs/` | one yaml per batch of runs |
| `daniel_updates/` | Daniel's notebook for the symbolic stoich layer |
| `docs/` | this readme, `ROADMAP.md`, notes on the archived v1 pipeline, logs from past batches |
| `archive/v1/` | the first pipeline, frozen (see `docs/archive_v1.md`) |

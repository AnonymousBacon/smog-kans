# Roadmap

Goal for the week: find out why turning on regularization/pruning makes the loss worse
and training slower, and why GPU and CPU runs disagree. Get there by starting from the
smallest possible KAN and adding one feature at a time, measuring both loss and time
at every step.

---

## What the code says about the problems

These are things found by reading `train.py`, `data.py` and pykan 0.2.8's source. Phase 1
will confirm or rule out each one with measurements.

### 1. Why regularization costs more compute (this is expected)

- With `lamb > 0`, pykan keeps `save_act=True` during `fit()`. Every forward pass then
  computes `torch.std` over each layer's `(batch, out, in)` activation tensor (96k × 16 × 16
  for the hidden layers) to get per-edge scales (`MultKAN.py` ~L811-830).
- The reg term (`acts_scale_spline`) stays in the autograd graph, so backward also goes
  through those reductions.
- LBFGS with `strong_wolfe` line search calls the closure several times per "step", so
  the extra cost is multiplied.
- With `lamb = 0`, pykan sets `save_act=False` inside `fit()` and skips all of this. So
  "reg on" vs "reg off" is not just a different loss. It's a different, heavier forward
  pass.

### 2. Why regularization makes train/val loss worse

- The objective is `MSE + lamb * (l1 + 2 * entropy)`, summed over **every edge**. With
  `lamb=1e-3` on a [11,16,16,16,11] net, the penalty can be the same size as the
  standardized MSE (~0.1). If that happens, the optimizer gives up accuracy to shrink edges.
  The logged `train_loss` is MSE only, so it looks like the model got worse for no reason.
- Right now reg is switched on **partway through** (warmup 1200 steps at lamb=0, then
  sparsify 300 steps at lamb=1e-3). Changing the objective mid-run throws LBFGS off.
  This is the "regularization: all or nothing" point: run reg from step 0, or not at all.
- **LBFGS memory is wiped every 25 steps.** `fit_chunked` calls `model.fit()` once per
  chunk, and each `fit()` builds a new `LBFGS` optimizer (`MultKAN.py` L1496). The
  curvature history is lost every `LBFGS_CHUNK` steps. This hurts every run, but it may
  hurt more once the objective has the extra reg term.

### 3. GPU vs CPU disagreements

- Already found: cuda `lstsq` returned NaNs after `prune()` (patched in `kan_patches.py`).
- float32 on GPU isn't bit-for-bit deterministic, and LBFGS with line search (tolerances
  set to 1e-32) magnifies tiny differences. Two runs can drift apart even when both are
  correct. We need a float64 check to tell drift apart from a real bug.
- For a very small model, the GPU may well be **slower** than the CPU because of kernel
  launch overhead. Measure it, don't assume.

### 4. Data loading is "nice" to the model

- The train/test split is sequential: the first 8,000 experiments train and the last
  2,000 test. It's split by experiment (good), but not shuffled.
- Old logs (`_train16_log.txt`) show test loss consistently **below** train loss
  (e.g. 0.33 vs 0.19). That points to an easier test block, not a good model.
- The best checkpoint is chosen on the test set, so the reported test score is optimistic.
- Fix: shuffle **experiment IDs** with a fixed seed, then split train/val/test 70/15/15
  by experiment. Pick checkpoints on val and report test once at the end.

---

## Phase 0: Harness (code to write first, ~1 day)

Everything else depends on runs being comparable and timed.

- [x] **Config-driven runs.** One script, `phase1.py --config configs/<name>.yaml`,
      where the config sets: width, grid, k, lamb, prune on/off, refine grids, steps,
      n_exps, device, dtype, seed. Each run writes to `runs/<name>/`: config copy,
      loss history (MSE and reg logged separately), timings, final metrics.
- [x] **Own training loop** in place of `model.fit()`: one LBFGS optimizer for the whole
      stage (fixes the reset every 25 steps), reg either on from step 0 or never, val
      evaluated every N steps rather than every step.
- [x] **Timing.** Seconds per step (with `torch.cuda.synchronize()` on GPU), time per
      stage, total wall-clock time. Peak memory is still not recorded.
- [x] **Unfriendly data split.** Shuffled experiment-level train/val/test split, scalers
      fit on train only, checkpoints picked on val. The flag to reproduce the old
      sequential split was not added -- nothing has needed it yet.
- [ ] **Mudd SLURM script** (`slurm/run_cpu.sbatch`) that runs one config on CPU, so CPU
      hours can be used as soon as they're approved.

## Phase 1: Smallest, simplest model

**1a. Baseline:** width `[11, 1, 11]` (1 hidden layer, 1 node), grid=3, k=3,
lamb=0, no prune, no refine, no symbolic. Fixed LBFGS step budget.

- [ ] Run on CPU and GPU, same seed. Compare the loss curves and sec/step. (CPU done,
      GPU side still outstanding.)
- [x] Run on CPU in float32 and float64 to measure how much drift is just numerics.
- [x] Scale data 1K → 2K → 5K → 10K experiments and check that time/step grows ~linearly.

**1b. Add one thing at a time** (each run = baseline + exactly one change):

| Run | Change | Question it answers |
|---|---|---|
| 1b-reg | lamb ∈ {1e-5, 1e-4, 1e-3} from step 0 | How much does reg alone cost in time and loss? At which lamb does reg ≈ MSE? |
| 1b-grid | simple refine 3 → 5 → 10 | Does refinement help a tiny model, and what does it cost? |
| 1b-width | width `[11, 4, 11]` | Is 1 node simply too small to learn anything? |
| 1b-prune | `[11, 4, 11]` + reg + prune | Does prune remove live edges? Loss before/after prune |

Output: a table of {final val MSE, sec/step, total time} per run on CPU and GPU. This
table tells us which lamb to use in Phase 2 and whether the reg slowdown comes from
pykan's overhead (expected) or from something in our code.

Answered on CPU by `configs/phase1_reg.yaml` → `runs/phase1_reg/cpu/summary.csv`: lamb
up to 1e-3 changes val MSE by less than the seed-to-seed spread and costs 8-20% more
time per step; lamb=1e-2 collapses the model to predicting the mean. So Phase 2's λ\*
is 1e-4 at the most, and reg is not worth its cost on a 418-parameter net.

## Phase 2: Full model, 10K experiments

Width `[11, 16, 16, 16, 11]`, 10K experiments, unfriendly split, same seed and the same
total step budget for all three runs:

- [ ] **2A: No L1.** lamb=0 the whole time.
- [ ] **2B: L1 the whole time.** lamb=λ* (picked in Phase 1) from step 0 to the end.
- [ ] **2C: L1 + prune.** Same as 2B, then prune, then retrain the pruned net at lamb=0
      (optional grid refine).

Report: val/test MSE, RMSE per species in ppb, sec/step, total wall-clock time, width
after pruning, and baselines (XGBoost/FFNN from `baselines.py`) on the same split.

## Phase 3: Scale out

- [ ] Vast.ai: run 2A/2B/2C in parallel on GPU instances (needs the API, see below).
- [ ] Mudd CPU hours: run the same three configs on CPU with the sbatch script for a
      direct CPU vs GPU comparison at full size.

---

## Open questions

1. **Vast API:** what we'll need to automate: launch an instance with a chosen GPU and
   image, copy the repo and data CSV up, run a config, pull `runs/<name>/` back, then
   destroy the instance. Is the data CSV allowed to live on Vast machines?
2. **Mudd:** is it a SLURM cluster? What partition/account, what core limit per job, and
   is there a GPU partition?
3. **Step budget:** fixed number of LBFGS steps per run, or a fixed wall-clock budget?
   (Fixed steps makes the time comparison cleaner.)
4. "1 hidden fxn": `[11, 1, 11]` is assumed above. If it meant a single edge/function
   (e.g. one input → one output), we'll adjust 1a.

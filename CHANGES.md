# Changes to smog_model.py

A summary of every meaningful change made to the model, with plain-English explanations of what each one does and why it matters.

---

## 1. Input preprocessing: log transform before scaling

**What changed:** Before feeding concentration values into the model, we now apply `log1p(x)` (which means "take the natural log of x + 1") to the input concentrations. Then we scale them with StandardScaler (subtract the mean, divide by standard deviation — so the data is centered at zero).

**Why it matters:** Atmospheric concentrations span wildly different magnitudes. O3 might be 100 ppb while OH is 0.0001 ppb. Without the log transform, the StandardScaler produced values 100–150 standard deviations away from the mean for some species. The model would essentially be looking at numbers like `-120` and `+140` as inputs, which makes it nearly impossible to learn anything useful. After log1p, all species end up with distributions that look roughly bell-shaped and are in a comparable range. This is visible in the distribution plot — the bottom-left panel should have neat, centered boxes after this fix.

**Technical note:** `log1p` instead of plain `log` handles the case where a concentration might be exactly 0 (log of 0 is undefined; log of 0+1 = log(1) = 0).

---

## 2. Output (tendency) scaling: StandardScaler only, no log

**What changed:** The output tendencies (how much each species changes per time step) are scaled with StandardScaler alone. No log transform on the outputs.

**Why:** Tendencies can be negative (a species decreasing), so log doesn't apply. StandardScaler handles negative values fine — it just centers and normalizes each species' tendency distribution independently.

---

## 3. Turned off regularization (LAMB = 0)

**What changed:** `LAMB` (short for lambda, the regularization coefficient in pykan) is set to 0.

**Regularization explained:** Regularization is a penalty added to the loss that discourages the model from becoming too complex. It pushes activations toward zero so the network can later be pruned (simplified). The trade-off is that regularization reduces raw accuracy.

**Why we turned it off:** Our goal right now is accuracy. Pruning and interpretability are a future concern. With `LAMB = 0`, the model uses its full capacity to fit the data rather than wasting effort on sparsity.

---

## 4. Two-phase optimizer: Adam first, then LBFGS

**What changed:** Training now runs in two phases. Phase 1 uses Adam for 2000 steps, then phase 2 switches to LBFGS for 75 steps.

**What these optimizers are:**
- **Adam** is a general-purpose optimizer. It updates the model using small random batches of data (1024 samples at a time here). It's fast per step and good at navigating the loss landscape globally, especially early in training.
- **LBFGS** is a more mathematically precise optimizer. It uses the full training dataset every step and takes "smarter" steps based on the curvature of the loss surface. It's slow per step but converges much more tightly to a minimum once Adam has gotten you to a good region.

**Why both:** Starting with Adam gets the model into the right ballpark quickly. LBFGS then fine-tunes it to a much lower loss than Adam alone could reach. This is a well-known strategy for KANs (the pykan authors recommend it).

---

## 5. Network architecture: [11, 16, 16, 11]

**What changed:** The network has four layers: 11 inputs → 16 hidden nodes → 16 hidden nodes → 11 outputs.

**Why this size:** The original architecture had 32 nodes in each hidden layer (`[11, 32, 32, 11]`). KANs are fundamentally different from standard neural networks — each connection between nodes is its own learned function (a spline), not just a number. This means `11×32 + 32×32 + 32×11` = 1,728 splines, each needing to be updated every training step with the full dataset. With 108,000 training samples, this made LBFGS steps take 3+ minutes each. Cutting to 16 nodes reduces to 928 splines — about half the computation.

---

## 6. Speed mode: model.speed()

**What changed:** After creating the model, we call `model.speed()`.

**What it does:** KANs have two branches — a learned spline branch and a symbolic branch (which tries to fit exact mathematical formulas like sin, log, etc.). The symbolic branch is expensive and rarely helps during early training. `speed()` disables it.

**Result:** 2–5× faster training steps with no meaningful accuracy cost during the learning phase.

---

## 7. Disabled auto-saving of network visualizations (auto_save=False)

**What changed:** When creating the KAN model, we pass `auto_save=False`.

**What it was doing:** By default, pykan saves a PNG image of the network structure after every training chunk. With hundreds of training steps chunked into groups, this was generating hundreds of `sp_0_0_0.png`, `sp_0_0_1.png`, etc. files into the working directory.

**Why we disabled it:** These files clutter the folder and slow down training. We generate our own plots explicitly.

---

## 8. Grid refinement: 3 → 10 → 20

**What changed:** After the Adam+LBFGS training, the model goes through two "refinement" steps that increase the resolution of the splines from grid=3 to grid=10, then from grid=10 to grid=20.

**Spline/grid explained:** KAN connections are B-splines — smooth curves fitted to data. The `grid` parameter controls how many "control points" each spline has. A grid of 3 means a simple, smooth curve with limited flexibility. A grid of 20 means the curve can capture much finer detail. But starting with a fine grid from scratch causes overfitting because the model memorizes noise. Starting coarse and refining lets it learn the overall shape first, then the details.

**How refinement works:** `model.refine(new_grid)` takes the already-learned spline shapes and re-parameterizes them at higher resolution, preserving what was learned. Then we run another 100 LBFGS steps to fine-tune the higher-resolution splines.

**Bug fix included:** Calling `model.speed()` earlier disables activation caching, which `model.refine()` needs internally. Before each refinement, we re-enable caching (`model.save_act = True`) and run a forward pass through the training data so the cache is populated. Without this, `refine()` crashes with an `IndexError`.

---

## 9. Best validation loss checkpointing

**What changed:** During training, whenever the model achieves a new best score on the test set (data it hasn't trained on), it saves a separate checkpoint to `model/smog_kan_best`.

**Why:** Training optimizers don't always finish at their best point. LBFGS might overshoot a good minimum, or the final grid refinement might slightly hurt performance. This ensures we always have the single best model captured, even if the final model is slightly worse.

**How it works:** The `fit_chunked()` function breaks training into small chunks (e.g., 200 Adam steps at a time, 25 LBFGS steps at a time), checks the test loss after each chunk, and saves if it improved.

---

## 10. Loss history saved to disk

**What changed:** After training, the full loss history — all training and test losses across all phases — is saved to `model/loss_history.npz`. The loss curve plot now loads from this file.

**Why:** Previously, the loss curve could only be generated immediately after training. If you loaded the model from a checkpoint (to skip retraining), the loss history was gone and the plot wouldn't generate. Now the history persists and the loss curve always renders.

---

## 11. Figures produced

**Distribution plot (`figures/distributions.png`):**
Shows the data before and after preprocessing in three rows. The bottom row (StandardScaler, axis clipped to 1–99th percentile) is what the model actually sees. Use this to verify preprocessing is working correctly.

**Loss curves (`figures/loss_curves.png`):**
Plots training loss and test loss on a log scale over all training steps. Vertical lines mark where Adam switches to LBFGS (gray) and where each grid refinement happens (orange). You want to see both lines decreasing and tracking each other closely (no gap = no overfitting).

**RMSE per species (`figures/rmse.png`):**
Bar chart showing how well the model predicts each of the 11 species. RMSE (Root Mean Squared Error) is the average prediction error in ppb/step — lower is better. Species with very small true tendencies (like OH) will have very small RMSE regardless of whether the model is learning, so interpret these bars relative to the typical magnitude of each species.

**Scatter plots (`figures/scatter.png`):**
One panel per species. Each dot is a test sample: x-axis = actual tendency, y-axis = model prediction. A perfect model would have all dots on the red diagonal line. A cloud of dots means the model is guessing randomly. Tight clustering along the diagonal = good.

**KAN graph (`Desktop/kan_graph.png`):**
A visualization of the network itself — each node and connection, with the learned spline shapes drawn on each edge. This is unique to KANs and shows which connections the model relied on most.

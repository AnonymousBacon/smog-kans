import copy
import time

import numpy as np
import torch
from kan import KAN
from kan.LBFGS import LBFGS

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def make_model(width, grid, k, seed, device, train_input, grid_eps=0.0, symbolic=False):
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device,
                auto_save=False, grid_eps=grid_eps, symbolic_enabled=symbolic, save_act=False)
    model.update_grid(train_input)
    model.save_act = False
    return model


def make_weights(species, weights, device, dtype):
    if not weights:
        return None
    w = torch.ones(len(species), device=device, dtype=dtype)
    for name, val in weights.items():
        if name not in species:
            raise ValueError(f"species_weights names {name!r}, not in {species}")
        w[species.index(name)] = float(val)
    return w / w.mean()


def mse(pred, target, w=None):
    se = (pred - target) ** 2
    if w is not None:
        se = se * w
    return torch.mean(se)


# applies the stoichiometric head when the run has one.
def predict(model, x, head=None):
    out = model(x)
    return out if head is None else head(out)


# the kan's 13 flux outputs: its final output with the head, or the input to the fixed last layer without.
@torch.no_grad()
def fluxes(model, x, symbolic):
    if not symbolic:
        return model(x)
    save_act = model.save_act
    model.save_act = True
    model(x)
    model.save_act = save_act
    return model.acts[-2]


@torch.no_grad()
def eval_mse(model, x, y, w=None, head=None):
    save_act = model.save_act
    model.save_act = False
    pred = predict(model, x, head)
    out = (mse(pred, y, w).item(), mse(pred, y).item())
    model.save_act = save_act
    return out


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_opt(model):
    return LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


def cache_acts(model, x):
    model.save_act = True
    with torch.no_grad():
        model(x)


# fraction of each spline layer's activations inside its grid. skip_last drops the fixed stoich layer in symbolic
# mode, whose splines are masked off, so the last entry lines up with the head runs' last layer.
def grid_coverage(model, x, skip_last=False):
    model.save_act = True
    with torch.no_grad():
        model(x)
    cov = []
    for l, layer in enumerate(model.act_fun):
        a = model.acts[l]
        lo = layer.grid[:, layer.k][None, :]
        hi = layer.grid[:, -layer.k - 1][None, :]
        cov.append(float(((a >= lo) & (a <= hi)).float().mean()))
    model.save_act = False
    return cov[:-1] if skip_last else cov


# set once a run is interrupted (timeout sends sigint): the current step finishes, the stage ends, later stages are
# skipped, and the best checkpoint so far is still evaluated and saved. a handler instead of KeyboardInterrupt so
# the signal can't land mid-eval or mid-save. read it as training.INTERRUPTED; a from-import would copy the old value.
INTERRUPTED = False


def on_sigint(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True


def new_log():
    return {k: [] for k in ("stage", "train_mse", "train_mse_uw", "reg", "val_mse",
                            "val_mse_uw", "step_time", "closure_evals", "early_stops")}


def train_stage(model, ds, steps, lamb, eval_every, device, log, stage, grid_update=(0, 0),
                w=None, head=None, patience=0, min_delta=0.0):
    model.save_act = lamb > 0
    opt = make_opt(model)
    x, y = ds["train_input"], ds["train_label"]
    gu_every, gu_until = grid_update
    state = {"evals": 0}

    def closure():
        opt.zero_grad()
        pred = predict(model, x, head)
        loss = mse(pred, y, w)
        if lamb > 0:
            reg = model.get_reg("edge_forward_spline_n", 1.0, 2.0, 0.0, 0.0)
        else:
            reg = torch.zeros((), device=device)
        objective = loss + lamb * reg
        objective.backward()
        state["evals"] += 1
        state["loss"], state["reg"] = loss.item(), reg.item()
        state["loss_uw"] = mse(pred.detach(), y).item()
        return objective

    best_val, best_state, stale = float("inf"), None, 0
    for step in range(1, steps + 1):
        state["evals"] = 0
        sync(device)
        t0 = time.perf_counter()
        if INTERRUPTED:
            print(f"  [{stage}] interrupted before step {step}, saving best checkpoint so far")
            log["early_stops"].append((stage, step))
            break
        if gu_every and step % gu_every == 1 % gu_every and step <= gu_until:
            model.update_grid(x)
            opt = make_opt(model)
        opt.step(closure)
        sync(device)
        dt = time.perf_counter() - t0

        val, val_uw = np.nan, np.nan
        if step % eval_every == 0 or step == steps:
            val, val_uw = eval_mse(model, ds["val_input"], ds["val_label"], w, head)
            # early stopping counts evals since the last real val improvement, but not while grid updates are still
            # running: each one resets lbfgs and can bump val, which would end a stage before the grid has settled.
            improved = val < best_val * (1 - min_delta)
            stale = 0 if improved or (gu_every and step <= gu_until) else stale + 1
            if val < best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
            print(f"  [{stage}] step {step:4d}/{steps}  train_mse={state['loss']:.5f}  "
                  f"reg={state['reg']:.3e}  val_mse={val:.5f}  "
                  f"val_mse_uw={val_uw:.5f}  {dt:.3f}s/step")

        log["stage"].append(stage)
        log["train_mse"].append(state["loss"])
        log["train_mse_uw"].append(state["loss_uw"])
        log["reg"].append(state["reg"])
        log["val_mse"].append(val)
        log["val_mse_uw"].append(val_uw)
        log["step_time"].append(dt)
        log["closure_evals"].append(state["evals"])

        if not np.isfinite(state["loss"]):
            print(f"  [{stage}] non-finite loss at step {step}, stopping stage")
            break
        if patience and stale >= patience:
            print(f"  [{stage}] early stop at step {step}: no val improvement in "
                  f"{patience} evals")
            log["early_stops"].append((stage, step))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_act = False
    return model

import copy
import functools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from kan import KAN
from kan.LBFGS import LBFGS
from sklearn.preprocessing import StandardScaler

import config as cfg
import data

DTYPES = {"float32": torch.float32, "float64": torch.float64}
N_STEPS_PER_EXP = 12

# phase1.py flushes its own prints but this module's did not, so a redirected log
# showed nothing until the buffer filled
print = functools.partial(print, flush=True)


def load_raw(csv_path, n_exps):
    cfg.CSV_PATH = csv_path
    cfg.N_EXPS = n_exps
    X, D, species, _ = data.load_and_clean_data()
    return X, D, species


def load_npz(path):
    # prebuilt dataset: a random subsample of the full csv with the experiment-level
    # split already fixed. reading the csv instead takes whichever experiments happen
    # to come first and re-derives the split from a seed, so two runs only match if
    # they agree on n_exps and seed. raw arrays are stored untransformed on purpose --
    # build_dataset applies the transform and fits the scalers on train rows itself
    d = np.load(path, allow_pickle=True)
    species = [str(s) for s in d["species"]]
    splits = (d["train_ids"], d["val_ids"], d["test_ids"])
    X, D = d["X_raw"], d["D_raw"]
    n_exps = X.shape[0] // N_STEPS_PER_EXP
    print(f"loaded {os.path.basename(path)}: {n_exps} experiments, {X.shape[0]} rows")
    print(f"Species ({len(species)}): {species}")
    return X, D, species, splits, n_exps


def split_experiments(n_exps, seed, frac=(0.7, 0.15, 0.15)):
    ids = np.random.default_rng(seed).permutation(n_exps)
    n_tr = int(frac[0] * n_exps)
    n_va = int(frac[1] * n_exps)
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def rows_for(exp_ids):
    return (exp_ids[:, None] * N_STEPS_PER_EXP + np.arange(N_STEPS_PER_EXP)).ravel()


def drop_outlier_exps(X_log, train_ids, val_ids, test_ids, max_z):
    probe = StandardScaler().fit(X_log[rows_for(train_ids)])
    z = np.abs(probe.transform(X_log)).reshape(-1, N_STEPS_PER_EXP, X_log.shape[1]).max(axis=(1, 2))
    kept = tuple(ids[z[ids] <= max_z] for ids in (train_ids, val_ids, test_ids))
    dropped = [int(len(a) - len(b)) for a, b in zip((train_ids, val_ids, test_ids), kept)]
    return kept, dropped


def transform_inputs(X, train_rows, kind):
    # log1p is only a log where x >> 1. every OH value is below 1 ppb, so log1p leaves
    # OH essentially raw: skew 87 and max |z| 128 on the 10k set, against < 6 for every
    # other species. that single species is what max_z was dropping experiments for.
    # a per-species floor fixes it without breaking the species that sit near zero --
    # a flat log10(x + 1e-12) moves the problem onto HCHO/H2O2/ALD2 instead
    if kind == "log1p":
        return np.log1p(X)
    if kind == "log10_floor":
        floor = np.maximum(np.percentile(X[train_rows], 1, axis=0), 1e-9)
        return np.log10(X + floor)
    raise ValueError(f"unknown transform {kind!r}")


def build_dataset(X, D, train_ids, val_ids, test_ids, device, dtype, max_z=0.0,
                  transform="log1p"):
    X_log = transform_inputs(X, rows_for(train_ids), transform)
    dropped = [0, 0, 0]
    if max_z:
        (train_ids, val_ids, test_ids), dropped = drop_outlier_exps(
            X_log, train_ids, val_ids, test_ids, max_z)
    tr, va, te = rows_for(train_ids), rows_for(val_ids), rows_for(test_ids)
    scaler_x = StandardScaler().fit(X_log[tr])
    scaler_y = StandardScaler().fit(D[tr])

    def t(a):
        return torch.tensor(a, dtype=dtype, device=device)

    ds = {}
    for split, rows in (("train", tr), ("val", va), ("test", te)):
        ds[f"{split}_input"] = t(scaler_x.transform(X_log[rows]))
        ds[f"{split}_label"] = t(scaler_y.transform(D[rows]))
    return ds, scaler_y, dropped


def make_model(width, grid, k, seed, device, train_input, grid_eps=0.0):
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device,
                auto_save=False, grid_eps=grid_eps, symbolic_enabled=False, save_act=False)
    # one-time fit of the grid to the standardized inputs; default range is [-1, 1]
    model.update_grid(train_input)
    model.save_act = False
    return model


def make_weights(species, weights, device, dtype):
    # normalized to mean 1 so the weighted loss stays on the same scale as the plain
    # one. without that, weighting OH 60:1 multiplies the objective by ~6 and changes
    # the lbfgs step sizes, which would confound the weighting with the optimizer
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


@torch.no_grad()
def eval_mse(model, x, y, w=None):
    # returns (weighted, unweighted). the weighted one is what training minimizes and
    # what picks the checkpoint; the unweighted one is the only number comparable
    # across runs with different weights
    save_act = model.save_act
    model.save_act = False
    pred = model(x)
    out = (mse(pred, y, w).item(), mse(pred, y).item())
    model.save_act = save_act
    return out


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_opt(model):
    return LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe",
                 tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


def grid_coverage(model, x):
    # fraction of each layer's inputs that land inside that layer's spline domain
    # outside it only the base function contributes
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
    return cov


def train_stage(model, ds, steps, lamb, eval_every, device, log, stage, grid_update=(0, 0),
                w=None):
    model.save_act = lamb > 0
    opt = make_opt(model)
    x, y = ds["train_input"], ds["train_label"]
    gu_every, gu_until = grid_update
    state = {"evals": 0}

    def closure():
        opt.zero_grad()
        pred = model(x)
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

    best_val, best_state = float("inf"), None
    for step in range(1, steps + 1):
        state["evals"] = 0
        sync(device)
        t0 = time.perf_counter()
        if gu_every and step % gu_every == 1 % gu_every and step <= gu_until:
            # refit the grid to where the activations are now; the coefficients are
            # refit too, so the old lbfgs curvature history no longer applies
            model.update_grid(x)
            opt = make_opt(model)
        opt.step(closure)
        sync(device)
        dt = time.perf_counter() - t0

        val, val_uw = np.nan, np.nan
        if step % eval_every == 0 or step == steps:
            val, val_uw = eval_mse(model, ds["val_input"], ds["val_label"], w)
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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_act = False
    return model


def cache_acts(model, x):
    model.save_act = True
    with torch.no_grad():
        model(x)


def run(rc, X, D, species, splits, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dtype = DTYPES[rc["dtype"]]
    torch.set_default_dtype(dtype)
    torch.manual_seed(rc["seed"])

    train_ids, val_ids, test_ids = splits
    # only subsets when the config asks for fewer experiments than the dataset holds,
    # so the n_exps sweeps in phase1.yaml still work and a full-size run uses the split
    # exactly as given
    n_train = int(0.7 * rc["n_exps"])
    if n_train < len(train_ids):
        train_ids = train_ids[:n_train]
    ds, scaler_y, dropped = build_dataset(X, D, train_ids, val_ids, test_ids, device,
                                          dtype, rc.get("max_z", 0.0),
                                          rc.get("transform", "log1p"))
    if any(dropped):
        print(f"  dropped outlier experiments (|z|>{rc['max_z']}): "
              f"train {dropped[0]}, val {dropped[1]}, test {dropped[2]}")

    n_sp = len(species)
    w = make_weights(species, rc.get("species_weights"), device, dtype)
    if w is not None:
        print("  species weights (mean-normalized): "
              + ", ".join(f"{s}={v:.3f}" for s, v in zip(species, w.tolist())))
    width = [n_sp] + list(rc["width_hidden"]) + [n_sp]
    model = make_model(width, rc["grid"], rc["k"], rc["seed"], device, ds["train_input"],
                       rc.get("grid_eps", 0.0))
    gu = (rc.get("grid_update_every", 0), rc.get("grid_update_until", 0))
    # refine() already repositions the knots, so re-running update_grid inside a refine
    # stage mostly just resets the lbfgs curvature history again. default stays True to
    # reproduce the phase2/deliverable runs
    gu_refine = gu if rc.get("grid_update_refine", True) else (0, 0)

    n_stages = 1 + len(rc["refine"]) + (1 if rc["prune"] else 0)
    stage_steps = rc["steps"] // n_stages
    log = {k: [] for k in ("stage", "train_mse", "train_mse_uw", "reg", "val_mse",
                           "val_mse_uw", "step_time", "closure_evals")}
    events = []

    t_start = time.perf_counter()
    model = train_stage(model, ds, stage_steps, rc["lamb"], rc["eval_every"], device, log,
                        "train", gu, w)

    if rc["prune"]:
        cache_acts(model, ds["train_input"])
        before = list(model.width)
        model = model.prune(node_th=rc["prune_node_th"], edge_th=rc["prune_edge_th"])
        model.auto_save = False
        events.append((len(log["stage"]), "prune"))
        print(f"  pruned width {before} -> {model.width}")
        model = train_stage(model, ds, stage_steps, 0.0, rc["eval_every"], device, log,
                            "post_prune", (0, 0), w)

    # refine keeps the run's lamb rather than dropping to 0: switching the objective
    # partway through is the thing "regularization: all or nothing" rules out, and it
    # throws lbfgs off. post_prune above is the deliberate exception -- phase 2c
    # retrains the pruned net unregularized on purpose
    for g in rc["refine"]:
        cache_acts(model, ds["train_input"])
        model = model.refine(g)
        model.auto_save = False
        events.append((len(log["stage"]), f"grid={g}"))
        model = train_stage(model, ds, stage_steps, rc["lamb"], rc["eval_every"], device,
                            log, f"grid{g}", gu_refine, w)
    total_time = time.perf_counter() - t_start

    with torch.no_grad():
        model.save_act = False
        pred = model(ds["test_input"]).cpu().numpy()
    pred_ppb = scaler_y.inverse_transform(pred)
    true_ppb = scaler_y.inverse_transform(ds["test_label"].cpu().numpy())
    rmse = np.sqrt(np.mean((pred_ppb - true_ppb) ** 2, axis=0))
    ss_res = np.sum((pred_ppb - true_ppb) ** 2, axis=0)
    ss_tot = np.sum((true_ppb - true_ppb.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot

    step_time = np.array(log["step_time"])
    metrics = {
        "name": rc["name"],
        "device": str(device),
        "width_final": [int(w if np.isscalar(w) else w[0]) for w in model.width],
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "steps_run": len(step_time),
        "total_time_s": total_time,
        "sec_per_step_mean": float(step_time.mean()),
        "sec_per_step_median": float(np.median(step_time)),
        "closure_evals_mean": float(np.mean(log["closure_evals"])),
        "final_train_mse": float(log["train_mse"][-1]),
        "final_train_mse_uw": float(log["train_mse_uw"][-1]),
        "best_val_mse": float(np.nanmin(np.array(log["val_mse"], dtype=float))),
        "best_val_mse_uw": float(np.nanmin(np.array(log["val_mse_uw"], dtype=float))),
        "species_weights": rc.get("species_weights") or {},
        "transform": rc.get("transform", "log1p"),
        "test_mse_scaled": float(np.mean((pred - ds["test_label"].cpu().numpy()) ** 2)),
        "test_rmse_ppb": dict(zip(species, map(float, rmse))),
        "test_r2": dict(zip(species, map(float, r2))),
        "test_r2_mean": float(np.mean(r2)),
        "test_r2_mean_no_oh": float(np.mean([v for s, v in zip(species, r2) if s != "OH"])),
        "test_r2_oh": float(r2[species.index("OH")]) if "OH" in species else float("nan"),
        "dropped_exps": dropped,
        "grid_coverage": grid_coverage(model, ds["train_input"]),
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(rc, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out_dir, "history.npz"),
             **{k: np.array(v) for k, v in log.items()},
             events=np.array([e[0] for e in events]), event_names=np.array([e[1] for e in events]))
    model.saveckpt(os.path.join(out_dir, "model"))

    plot_loss(log, events, rc, os.path.join(out_dir, "loss_curves.png"))
    plot_scatter(pred_ppb, true_ppb, species, r2, rc["name"], os.path.join(out_dir, "scatter.png"))
    plot_rmse(rmse, r2, species, rc["name"], os.path.join(out_dir, "rmse_r2.png"))
    plot_graph(model, ds["train_input"], species, os.path.join(out_dir, "kan_graph.png"))

    torch.set_default_dtype(torch.float32)
    return metrics, log


def plot_loss(log, events, rc, path):
    steps = np.arange(1, len(log["train_mse"]) + 1)
    val = np.array(log["val_mse"], dtype=float)
    has = ~np.isnan(val)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    weighted = bool(rc.get("species_weights"))
    suffix = " (weighted)" if weighted else ""
    ax.semilogy(steps, log["train_mse"], label=f"train MSE{suffix}", lw=1)
    ax.semilogy(steps[has], val[has], "o-", ms=3, label=f"val MSE{suffix}", lw=1)
    if weighted:
        # the unweighted curves are the ones comparable with the unweighted run
        val_uw = np.array(log["val_mse_uw"], dtype=float)
        ax.semilogy(steps, log["train_mse_uw"], ls="--", lw=1, color="tab:blue",
                    alpha=0.6, label="train MSE (unweighted)")
        ax.semilogy(steps[has], val_uw[has], ls="--", lw=1, color="tab:orange",
                    alpha=0.6, label="val MSE (unweighted)")
    if rc["lamb"] > 0:
        ax2 = ax.twinx()
        ax2.semilogy(steps, np.maximum(log["reg"], 1e-12), color="tab:green", alpha=0.6, lw=1, label="reg")
        ax2.set_ylabel("reg term (unscaled)", color="tab:green")
    for s, name in events:
        ax.axvline(s, color="tab:orange", ls="--", lw=1)
        ax.text(s, ax.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("LBFGS step")
    ax.set_ylabel("MSE (standardized)")
    ax.set_title(f"{rc['name']}: loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, log["step_time"], lw=1)
    ax.set_xlabel("LBFGS step")
    ax.set_ylabel("seconds / step")
    ax.set_title(f"step time (mean {np.mean(log['step_time']):.3f}s)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(pred, true, species, r2, title, path):
    n = len(species)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.scatter(true[:, i], pred[:, i], s=2, alpha=0.3)
        lo, hi = np.percentile(true[:, i], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "r-", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{species[i]}  R²={r2[i]:.3f}", fontsize=10)
        ax.set_xlabel("actual (ppb/step)", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
    fig.suptitle(f"{title}: test set")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_rmse(rmse, r2, species, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(species, rmse)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE (ppb/step)")
    axes[1].bar(species, r2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylim(min(-0.1, np.min(r2) - 0.05), 1.05)
    axes[1].set_ylabel("R²")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{title}: test set per species")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_graph(model, x, species, path):
    try:
        cache_acts(model, x[:2000])
        model.plot(folder=os.path.join(os.path.dirname(path), "kan_edges"),
                   in_vars=species, out_vars=species, scale=1.0, varscale=0.3)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"  kan graph skipped: {e}")
    finally:
        model.save_act = False

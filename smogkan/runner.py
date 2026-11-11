import json
import os
import signal
import time

import numpy as np
import torch

import stoich
from smogkan import plots, training
from smogkan.data import build_dataset, rows_for
from smogkan.metrics import rmse_r2, run_stats, test_scores
from smogkan.training import DTYPES, fluxes, make_model, make_weights, predict, train_stage


# trains one config entry rc end to end and writes config.json, metrics.json, history.npz, the checkpoint and the
# summary plots to out_dir. returns (metrics, log).
def run(rc, X, D, species, splits, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    signal.signal(signal.SIGINT, training.on_sigint)
    dtype = DTYPES[rc["dtype"]]
    torch.set_default_dtype(dtype)
    torch.manual_seed(rc["seed"])

    train_ids, val_ids, test_ids = splits
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
    # with stoich on, the last kan layer outputs one flux per reaction instead of one tendency per species.
    # stoich: symbolic is daniel's version, where one more kan layer fixed to S replaces the head.
    head, n_out = None, n_sp
    symbolic = rc.get("stoich") == "symbolic"
    if symbolic and (rc.get("flux_positive") or rc["prune"]):
        raise ValueError("stoich: symbolic does not support flux_positive or prune")
    if rc.get("stoich"):
        S = stoich.stoich_matrix(species)
        D_train = D[rows_for(train_ids)]
        if not stoich.check_S(S, D_train, species):
            raise ValueError("stoichiometric matrix does not match the data, see check above")
        head = stoich.make_head(S, D_train, scaler_y, device, dtype, rc.get("head_scale", "equal"),
                                rc.get("flux_positive", False))
        n_out = S.shape[1]
    width = [n_sp] + list(rc["width_hidden"]) + [n_out] + ([n_sp] if symbolic else [])
    model = make_model(width, rc["grid"], rc["k"], rc["seed"], device, ds["train_input"],
                       rc.get("grid_eps", 0.0), symbolic)
    flux_head = head
    if symbolic:
        stoich.fix_stoich_layer(model, head)
        head = None
    gu = (rc.get("grid_update_every", 0), rc.get("grid_update_until", 0))
    gu_refine = gu if rc.get("grid_update_refine", True) else (0, 0)

    n_stages = 1 + len(rc["refine"]) + (1 if rc["prune"] else 0)
    stage_steps = rc["steps"] // n_stages
    log = training.new_log()
    events = []
    es = {"head": head, "patience": rc.get("patience", 0), "min_delta": rc.get("min_delta", 0.0)}

    t_start = time.perf_counter()
    model = train_stage(model, ds, stage_steps, rc["lamb"], rc["eval_every"], device, log,
                        "train", gu, w, **es)

    if rc["prune"] and not training.INTERRUPTED:
        training.cache_acts(model, ds["train_input"])
        before = list(model.width)
        model = model.prune(node_th=rc["prune_node_th"], edge_th=rc["prune_edge_th"])
        model.auto_save = False
        events.append((len(log["stage"]), "prune"))
        print(f"  pruned width {before} -> {model.width}")
        model = train_stage(model, ds, stage_steps, 0.0, rc["eval_every"], device, log,
                            "post_prune", (0, 0), w, **es)

    for g in rc["refine"]:
        if training.INTERRUPTED:
            break
        training.cache_acts(model, ds["train_input"])
        model = model.refine(g)
        model.auto_save = False
        if symbolic:
            stoich.freeze_stoich_layer(model)
        events.append((len(log["stage"]), f"grid={g}"))
        model = train_stage(model, ds, stage_steps, rc["lamb"], rc["eval_every"], device,
                            log, f"grid{g}", gu_refine, w, **es)
    total_time = time.perf_counter() - t_start

    with torch.no_grad():
        model.save_act = False
        pred = predict(model, ds["test_input"], head).cpu().numpy()
        flux = None if flux_head is None else flux_head.flux_ppb(
            fluxes(model, ds["test_input"], symbolic)).cpu().numpy()
    pred_ppb = scaler_y.inverse_transform(pred)
    true_ppb = scaler_y.inverse_transform(ds["test_label"].cpu().numpy())
    rmse, r2 = rmse_r2(pred_ppb, true_ppb)

    metrics = run_stats(rc, model, log, device, total_time)
    metrics["test_mse_scaled"] = float(np.mean((pred - ds["test_label"].cpu().numpy()) ** 2))
    metrics.update(test_scores(species, pred_ppb, true_ppb, rmse, r2, flux))
    metrics["dropped_exps"] = dropped
    metrics["grid_coverage"] = training.grid_coverage(model, ds["train_input"], symbolic)
    metrics["interrupted"] = training.INTERRUPTED
    metrics["early_stops"] = [list(e) for e in log["early_stops"]]

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(rc, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out_dir, "history.npz"),
             **{k: np.array(v) for k, v in log.items()},
             events=np.array([e[0] for e in events]), event_names=np.array([e[1] for e in events]))
    model.saveckpt(os.path.join(out_dir, "model"))

    plots.plot_loss(log, events, rc, os.path.join(out_dir, "loss_curves.png"))
    plots.plot_scatter(pred_ppb, true_ppb, species, r2, rc["name"], os.path.join(out_dir, "scatter.png"))
    plots.plot_rmse(rmse, r2, species, rc["name"], os.path.join(out_dir, "rmse_r2.png"))
    # off by default: pykan draws the graph by saving one png per spline edge (~1000 per run) and reading them back.
    if rc.get("kan_graph", False):
        out_vars = species if head is None else [f"R{j + 1}" for j in range(n_out)]
        plots.plot_graph(model, ds["train_input"], species, os.path.join(out_dir, "kan_graph.png"), out_vars)

    torch.set_default_dtype(torch.float32)
    return metrics, log

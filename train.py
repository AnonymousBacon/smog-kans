import numpy as np
import torch
from kan import KAN

import config as cfg


def build_kan_width(n_species):
    return [n_species, 16, 16, 16, n_species]


def build_species_weight(species_names, device):
    return torch.tensor(
        [cfg.SPECIES_WEIGHTS.get(name, 1.0) for name in species_names],
        dtype=torch.float32, device=device,
    )


def make_weighted_mse(species_weight):
    def weighted_mse(pred, target):
        return torch.mean(species_weight * (pred - target) ** 2)
    return weighted_mse


def make_model(kan_width, seed, device):
    torch.manual_seed(seed)
    m = KAN(width=kan_width, grid=5, k=3, seed=seed, device=device, auto_save=False, grid_eps=0.0)
    m.speed()
    return m


def loss_key(results):
    return 'test_loss' if 'test_loss' in results else 'val_loss'


def fit_chunked(model, dataset, opt, total_steps, chunk_size, best_val, print_every=cfg.PRINT_EVERY, **kwargs):
    # pykan defaults to update_grid=True, which rebuilds each layer's grid from
    # sample quantiles every grid_update_freq(5) steps. that jitters the loss every
    # 5 steps and goes NaN outright once the grid is refined past 5 intervals, so
    # the grid stays fixed here and is only changed deliberately by refine()
    kwargs.setdefault("update_grid", False)
    train_hist, test_hist = [], []
    steps_done = 0
    while steps_done < total_steps:
        s = min(chunk_size, total_steps - steps_done)
        res = model.fit(dataset, opt=opt, steps=s, **kwargs)
        tk = loss_key(res)
        train_hist += res['train_loss']
        test_hist  += res[tk]
        val = res[tk][-1]
        if val < best_val:
            best_val = val
            model.saveckpt(cfg.BEST_CKPT_PATH)
            print(f"  best val loss: {val:.6f} → saved")
        steps_done += s
        if steps_done % print_every == 0 or steps_done == total_steps:
            print(f"  step {steps_done:5d}/{total_steps}  train_loss={res['train_loss'][-1]:.6f}  test_loss={val:.6f}")
    return train_hist, test_hist, best_val


def run_training_pipeline(dataset, kan_width, weighted_mse):
    if cfg.LOAD_FROM_CHECKPOINT:
        print(f"\nLoading checkpoint: {cfg.CKPT_PATH}")
        model = KAN.loadckpt(cfg.CKPT_PATH)
        return model, [], [], float('inf')

    model    = make_model(kan_width, cfg.SEED, cfg.device)
    best_val = float('inf')
    train_loss, test_loss = [], []

    # warmup: fast training in speed mode, no sparsification yet
    print(f"\nWarmup: {cfg.WARMUP_STEPS} steps (speed mode)")
    stage_train, stage_test, best_val = fit_chunked(
        model, dataset, "LBFGS", cfg.WARMUP_STEPS, cfg.LBFGS_CHUNK,
        best_val, lamb=0, batch=-1, loss_fn=weighted_mse,
    )
    train_loss += stage_train
    test_loss  += stage_test

    # sparsify: leave speed mode so activations get cached for the l1/entropy
    # penalty. also re-enable the symbolic branch here for good, since speed()
    # turns it off permanently — if left off, auto_symbolic's snapped edges
    # would silently contribute nothing to the forward pass later on
    model.save_act         = True
    model.symbolic_enabled = True
    print(f"\nSparsifying: {cfg.SPARSIFY_STEPS} steps (lamb={cfg.SPARSIFY_LAMB})")
    stage_train, stage_test, best_val = fit_chunked(
        model, dataset, "LBFGS", cfg.SPARSIFY_STEPS, cfg.LBFGS_CHUNK,
        best_val, lamb=cfg.SPARSIFY_LAMB, batch=-1, loss_fn=weighted_mse,
    )
    train_loss += stage_train
    test_loss  += stage_test

    # prune dead nodes/edges, then drop activation caching again for speed
    print(f"\nPruning (node_th={cfg.PRUNE_NODE_TH}, edge_th={cfg.PRUNE_EDGE_TH})")
    print(f"  width before: {model.width}")
    model = model.prune(node_th=cfg.PRUNE_NODE_TH, edge_th=cfg.PRUNE_EDGE_TH)
    print(f"  width after:  {model.width}")
    model.save_act  = False
    model.auto_save = False

    # prune changed the parameter shapes, so a best checkpoint from before it can
    # no longer be loaded into this model. reset so BEST_CKPT_PATH stays loadable
    best_val = float('inf')

    # grid refinement: train the pruned network at increasing spline
    # resolution, retraining after each refine step
    for grid in cfg.REFINE_GRIDS:
        model.save_act = True
        model(dataset['train_input'])
        model = model.refine(grid)
        model.save_act = False   # refine() resets save_act to its default (True)
        best_val = float('inf')  # grid change resizes coef, same reason as after prune

        print(f"\nGrid refine: grid={grid} ({cfg.REFINE_STEPS} steps)")
        stage_train, stage_test, best_val = fit_chunked(
            model, dataset, "LBFGS", cfg.REFINE_STEPS, cfg.LBFGS_CHUNK,
            best_val, lamb=0, batch=-1, loss_fn=weighted_mse,
        )
        train_loss += stage_train
        test_loss  += stage_test

    model.saveckpt(cfg.CKPT_PATH)
    print(f"Checkpoint saved → {cfg.CKPT_PATH}_*")
    print("To skip retraining next run: set LOAD_FROM_CHECKPOINT = True")

    np.savez(
        cfg.HIST_PATH,
        train_loss=train_loss,
        test_loss=test_loss,
    )

    return model, train_loss, test_loss, best_val

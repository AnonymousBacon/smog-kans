# quick cpu check of daniel's symbolic stoich layer before paying for a gpu run.
# checks: the frozen last layer equals StoichHead, it stays frozen through lbfgs, loss drops, atoms are conserved.
# usage: .venv/bin/python smoke_symbolic.py [steps] [n_exps]
import sys

import numpy as np
import torch

import kan_patches
import stoich
from smogkan import paths, training
from smogkan.data import build_dataset, load_npz, rows_for
from smogkan.training import eval_mse, fluxes, make_model, make_weights, train_stage

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_EXPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DATA = paths.NPZ_PATH
CPU = torch.device("cpu")


# what reaches the output: the symbolic +-x coefficients, the node and subnode affine, and the spline mask.
# grid updates rewrite the last layer's spline grid and coef, but the mask keeps those splines at zero.
def frozen_state(model):
    L = model.depth - 1
    ps = list(model.symbolic_fun[L].parameters()) + [model.act_fun[L].mask, model.node_bias[L],
                                                       model.node_scale[L], model.subnode_bias[L],
                                                       model.subnode_scale[L]]
    return [p.detach().clone() for p in ps]


def check(dtype_name):
    dtype = training.DTYPES[dtype_name]
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)
    X, D, species, (tr, va, te), _ = load_npz(DATA, all_species=True)
    n = N_EXPS
    tr, va, te = tr[:int(0.7 * n)], va[:int(0.15 * n)], te[:int(0.15 * n)]
    ds, scaler_y, _ = build_dataset(X, D, tr, va, te, CPU, dtype, 0.0, "log10_floor")

    S = stoich.stoich_matrix(species)
    head = stoich.make_head(S, D[rows_for(tr)], scaler_y, CPU, dtype, "equal")
    model = make_model([16, 16, 16, 16, 13, 16], 3, 3, 42, CPU, ds["train_input"], 0.0, symbolic=True)
    stoich.fix_stoich_layer(model, head)
    w = make_weights(species, {"OH": 5}, CPU, dtype)
    x = ds["test_input"]

    def gap():
        with torch.no_grad():
            return (model(x) - head(fluxes(model, x, True))).abs().max().item()

    gap0 = gap()
    before = frozen_state(model)
    trainable_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    val0 = eval_mse(model, ds["val_input"], ds["val_label"], w)[0]

    log = training.new_log()
    model = train_stage(model, ds, STEPS, 0.0, 10, CPU, log, "train", (10, 50), w)

    val1 = eval_mse(model, ds["val_input"], ds["val_label"], w)[0]
    still_frozen = all(torch.equal(a, b) for a, b in zip(before, frozen_state(model)))
    moved = any(not torch.equal(a, p) for a, p in zip(trainable_before,
                                                       [p for p in model.parameters() if p.requires_grad]))
    with torch.no_grad():
        pred_ppb = scaler_y.inverse_transform(model(x).double().numpy())
    atom = np.abs(pred_ppb @ stoich.atoms(species).T).max(axis=0)
    # residual relative to the size of the predicted tendencies, so float32 and float64 compare fairly
    rel = atom.max() / np.abs(pred_ppb).max()

    print(f"\n[{dtype_name}] {len(tr)} train exps, {STEPS} lbfgs steps")
    print(f"  symbolic layer vs StoichHead, max gap: before {gap0:.2e}  after {gap():.2e}")
    print(f"  last layer unchanged by training: {still_frozen}   other weights moved: {moved}")
    print(f"  val mse: {val0:.4f} -> {val1:.4f}")
    print(f"  atom residual ppb (C,H,N,O): {', '.join(f'{a:.1e}' for a in atom)}   relative: {rel:.1e}")
    return still_frozen and moved and val1 < val0


if __name__ == "__main__":
    kan_patches.install()
    ok = [check(d) for d in ("float32", "float64")]
    print("\nsmoke test", "passed" if all(ok) else "FAILED")
    sys.exit(0 if all(ok) else 1)

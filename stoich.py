# fixed stoichiometric output layer from sturm and wexler 2022; run this file to check S against the data.
import os

import numpy as np
import torch
import torch.nn as nn

SPECIES16 = ["O3", "NO", "NO2", "HCHO", "HO2", "H2O2", "OH", "HNO3", "CO", "H2",
             "ALD2", "MGLY", "MCO3", "PAN", "H2O", "O2"]

# carbon, hydrogen, nitrogen and oxygen atoms in each species.
COMPOSITION = {
    "O3": (0, 0, 0, 3), "NO": (0, 0, 1, 1), "NO2": (0, 0, 1, 2), "HCHO": (1, 2, 0, 1),
    "HO2": (0, 1, 0, 2), "H2O2": (0, 2, 0, 2), "OH": (0, 1, 0, 1), "HNO3": (0, 1, 1, 3),
    "CO": (1, 0, 0, 1), "H2": (0, 2, 0, 0), "ALD2": (2, 4, 0, 1), "MGLY": (3, 4, 0, 2),
    "MCO3": (2, 3, 0, 3), "PAN": (2, 3, 1, 5), "H2O": (0, 2, 0, 1), "O2": (0, 0, 0, 2),
}

# the 13 reactions of the julia photochemical mechanism in table 1 order, products positive and reactants negative.
REACTIONS = [
    {"NO2": -1, "O2": -1, "NO": 1, "O3": 1},
    {"O3": -1, "NO": -1, "NO2": 1, "O2": 1},
    {"HCHO": -1, "O2": -2, "HO2": 2, "CO": 1},
    {"HCHO": -1, "H2": 1, "CO": 1},
    {"HCHO": -1, "OH": -1, "O2": -1, "HO2": 1, "CO": 1, "H2O": 1},
    {"HO2": -1, "NO": -1, "OH": 1, "NO2": 1},
    {"OH": -1, "NO2": -1, "HNO3": 1},
    {"H2O2": -1, "OH": 2},
    {"H2O2": -1, "OH": -1, "H2O": 1, "HO2": 1},
    {"ALD2": -1, "OH": -1, "O2": -1, "MCO3": 1, "H2O": 1},
    {"MGLY": -1, "O2": -2, "MCO3": 1, "HO2": 1, "CO": 1},
    {"MCO3": -1, "NO2": -1, "PAN": 1},
    {"PAN": -1, "MCO3": 1, "NO2": 1},
]


def atoms(species):
    return np.array([COMPOSITION[s] for s in species], dtype=float).T


def stoich_matrix(species, reactions=REACTIONS):
    if not reactions:
        raise ValueError("REACTIONS in stoich.py is empty -- fill in the mechanism first")
    S = np.zeros((len(species), len(reactions)))
    for j, r in enumerate(reactions):
        for name, nu in r.items():
            if name not in species:
                raise ValueError(f"reaction R{j + 1} names {name!r}, not in {species}")
            S[species.index(name), j] = nu
    return S


# fails if a reaction loses atoms or if the training tendencies fall outside what S can produce.
def check_S(S, D, species, tol=1e-6):
    imbalance = atoms(species) @ S
    bad = [f"R{j + 1}: C,H,N,O off by {imbalance[:, j].tolist()}"
           for j in range(S.shape[1]) if np.abs(imbalance[:, j]).max() > 1e-9]
    flux, *_ = np.linalg.lstsq(S, D.T, rcond=None)
    resid = np.linalg.norm(D.T - S @ flux) / np.linalg.norm(D)
    rank_S = np.linalg.matrix_rank(S)
    rank_D = np.linalg.matrix_rank(D, tol=1e-8 * np.abs(D).max())
    ok = not bad and resid < tol and rank_S >= rank_D
    print(f"stoich check: {S.shape[1]} reactions, rank(S)={rank_S}, rank(data)={rank_D}, "
          f"relative residual of data outside span(S)={resid:.2e}  ->  {'ok' if ok else 'FAILED'}")
    for line in bad:
        print("  unbalanced " + line)
    if resid >= tol:
        worst = np.abs(D.T - S @ flux).mean(axis=1) / (np.abs(D).mean(axis=0) + 1e-30)
        order = np.argsort(worst)[::-1][:5]
        print("  species the data moves but S can't: "
              + ", ".join(f"{species[i]} ({worst[i]:.2e})" for i in order))
    return ok


# turns the kan's 13 flux outputs into standardized tendencies, applying S in ppb so atoms are conserved exactly.
class StoichHead(nn.Module):
    def __init__(self, S, flux_scale, y_mean, y_scale, device, dtype, positive=False, mix=None):
        super().__init__()
        self.positive = positive

        def t(a):
            return torch.tensor(np.asarray(a), dtype=dtype, device=device)

        self.register_buffer("S", t(S))
        self.register_buffer("flux_scale", t(flux_scale))
        self.register_buffer("y_mean", t(y_mean))
        self.register_buffer("y_scale", t(y_scale))
        # mixes the kan outputs into reaction fluxes; identity unless the head is whitened.
        self.register_buffer("mix", t(np.eye(S.shape[1]) if mix is None else mix))

    # every reaction in the mechanism runs one way only, so positive=True keeps fluxes >= 0 through a softplus.
    def flux_ppb(self, f):
        if self.positive:
            f = nn.functional.softplus(f)
        return (f @ self.mix.T) * self.flux_scale

    def forward(self, f):
        return (self.flux_ppb(f) @ self.S.T - self.y_mean) / self.y_scale


# gives every reaction a unit that moves the standardized loss equally, since OH's tiny scale otherwise makes its reactions hundreds of times more sensitive.
# "whiten" goes further: the equal-scaled map from kan outputs to standardized tendencies still has singular values
# from ~2.4 down to ~1e-3, so the kan had to output values in the hundreds along the weak directions. mixing through
# V / sigma makes each of the 11 kan outputs that reach the loss move it along one orthonormal direction; the 2 null
# directions keep unit weight and never touch the loss. S is still applied in ppb, so atoms balance as before.
def make_head(S, D_train, scaler_y, device, dtype, scaling="equal", positive=False):
    mix = None
    if scaling in ("equal", "whiten"):
        scale = 1.0 / np.linalg.norm(S / scaler_y.scale_[:, None], axis=0)
    else:
        scale = (np.linalg.pinv(S) @ D_train.T).std(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    if scaling == "whiten":
        if positive:
            raise ValueError("head_scale: whiten mixes fluxes, so it does not support flux_positive")
        _, sig, Vt = np.linalg.svd(S * scale[None, :] / scaler_y.scale_[:, None])
        inv = np.where(sig > 1e-9 * sig[0], 1.0 / np.maximum(sig, 1e-300), 1.0)
        mix = Vt.T * inv[None, :]
    return StoichHead(S, scale, scaler_y.mean_, scaler_y.scale_, device, dtype, positive, mix)


# daniel getter's version: rather than a head after the kan, the kan's own last layer is fixed to S through pykan's
# symbolic edges, so plot/prune/symbolic tools see the stoichiometry. edge (flux i -> species j) is c * x with the
# same coefficients StoichHead uses, so both compute the same function; the coefficient sits in the edge's affine
# slot under the library name "x" because a lambda saves as "anonymous" and loadckpt can't read it back.
def fix_stoich_layer(model, head):
    L = model.depth - 1
    layer = model.symbolic_fun[L]
    if (layer.in_dim, layer.out_dim) != tuple(head.S.T.shape):
        raise ValueError(f"last layer is {layer.in_dim}->{layer.out_dim}, S needs {tuple(head.S.T.shape)}")
    for i in range(layer.in_dim):
        for j in range(layer.out_dim):
            model.fix_symbolic(L, i, j, "x", fit_params_bool=False, verbose=False, log_history=False)
    with torch.no_grad():
        # affine is (out, in, 4) holding a, b, c, d of c * f(a * x + b) + d.
        layer.affine[:, :, 2] = (head.S * head.flux_scale[None, :]) @ head.mix / head.y_scale[:, None]
        model.node_bias[L].copy_(-head.y_mean / head.y_scale)
    freeze_stoich_layer(model)


# refine() builds a fresh last layer with trainable splines (masked off, but still counted), so call this after it too.
def freeze_stoich_layer(model):
    L = model.depth - 1
    for p in list(model.act_fun[L].parameters()) + list(model.symbolic_fun[L].parameters()):
        p.requires_grad_(False)


if __name__ == "__main__":
    import experiment

    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "baseline_10k", "baseline_10k.npz")
    X, D, species, (train_ids, _, _), _ = experiment.load_npz(path, all_species=True)
    D_train = D[experiment.rows_for(train_ids)]
    print("data atom residual (C,H,N,O):", np.abs(atoms(species) @ D_train.T).max(axis=1))
    check_S(stoich_matrix(species), D_train, species)

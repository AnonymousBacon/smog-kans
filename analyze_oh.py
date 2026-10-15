import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression

N_STEPS_PER_EXP = 12
EPS = 1e-12


def rows_for(ids):
    return (ids[:, None] * N_STEPS_PER_EXP + np.arange(N_STEPS_PER_EXP)).ravel()


def r2(pred, actual):
    return float(1 - ((pred - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())


def fit_hgb(xtr, ytr, xte):
    mu, sd = ytr.mean(), ytr.std()
    m = HGB(max_iter=400, random_state=0).fit(xtr, (ytr - mu) / sd)
    return m, m.predict(xte) * sd + mu


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/baseline_10k/baseline_10k.npz"
    d = np.load(path, allow_pickle=True)
    species = [str(s) for s in d["species"]]
    X, D = d["X_raw"], d["D_raw"]
    R, E = rows_for(d["train_ids"]), rows_for(d["test_ids"])

    oh = species.index("OH")
    others = [i for i in range(len(species)) if i != oh]
    lx = np.log10(X + EPS)
    out = {}

    lin = LinearRegression().fit(lx[R][:, others], lx[R][:, oh])
    out["ss_map_linear"] = r2(lin.predict(lx[E][:, others]), lx[E][:, oh])
    ss, ss_pred = fit_hgb(lx[R][:, others], lx[R][:, oh], lx[E][:, others])
    out["ss_map_nonlinear"] = r2(ss_pred, lx[E][:, oh])

    lx_next = np.log10(X + D + EPS)
    mu, sd = lx[R][:, oh].mean(), lx[R][:, oh].std()
    oh_next = 10 ** (ss.predict(lx_next[E][:, others]) * sd + mu)
    out["ss_route_dOH_ceiling"] = r2(oh_next - X[E][:, oh], D[E][:, oh])

    _, direct = fit_hgb(lx[R], D[R][:, oh], lx[E])
    out["direct_dOH_gbm"] = r2(direct, D[E][:, oh])

    out["median_abs_dOH_over_median_OH"] = float(
        np.median(np.abs(D[:, oh])) / np.median(X[:, oh]))

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", "oh_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"steady-state map, linear      log10[OH] = f(other 10) : R2 {out['ss_map_linear']:7.4f}")
    print(f"steady-state map, nonlinear   log10[OH] = f(other 10) : R2 {out['ss_map_nonlinear']:7.4f}")
    print(f"dOH via steady state, true future others (CEILING)    : R2 {out['ss_route_dOH_ceiling']:7.4f}")
    print(f"dOH direct, gradient boosting on the same 11 inputs   : R2 {out['direct_dOH_gbm']:7.4f}")
    print(f"median |dOH| / median OH                              :    {out['median_abs_dOH_over_median_OH']:7.4f}")
    print()
    print("for comparison, the KAN on the same split: 0.8989 unweighted, 0.9477 at OH 60:1")


if __name__ == "__main__":
    main()

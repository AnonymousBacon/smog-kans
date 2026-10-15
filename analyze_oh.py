"""is OH better predicted directly, or from a steady-state assumption?

OH is the one species the KAN struggles with (test R2 0.899 unweighted). the fallback on
the todo list was to predict it from a steady-state assumption instead. this decides
that, without training a KAN -- everything here is seconds of sklearn.

the steady-state idea: OH is a radical with a lifetime far shorter than the 5-minute
step, so its concentration should be slaved to the slower species around it. if
[OH] = f(other ten) holds, we can predict the other ten (which the KAN does at
R2 >= 0.9955), map through f to get OH at t+1, and subtract to get the tendency.

writes runs/oh_analysis.json.

usage: python analyze_oh.py [path/to/baseline_10k.npz]
"""
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
    # standardize the target. raw dOH is ~1e-3 ppb and the boosting model's split
    # tolerances are absolute, so an unstandardized target quietly handicaps it and
    # would make the KAN look better than it is
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

    # 1. does the steady-state relation hold at all? regress log[OH] on the other ten
    #    at the SAME timestep, linear first then nonlinear
    lin = LinearRegression().fit(lx[R][:, others], lx[R][:, oh])
    out["ss_map_linear"] = r2(lin.predict(lx[E][:, others]), lx[E][:, oh])
    ss, ss_pred = fit_hgb(lx[R][:, others], lx[R][:, oh], lx[E][:, others])
    out["ss_map_nonlinear"] = r2(ss_pred, lx[E][:, oh])

    # 2. the ceiling on the steady-state route: feed it the TRUE future concentrations
    #    of the other ten, which no model would ever have. anything the real pipeline
    #    achieves is below this
    lx_next = np.log10(X + D + EPS)
    mu, sd = lx[R][:, oh].mean(), lx[R][:, oh].std()
    oh_next = 10 ** (ss.predict(lx_next[E][:, others]) * sd + mu)
    out["ss_route_dOH_ceiling"] = r2(oh_next - X[E][:, oh], D[E][:, oh])

    # 3. direct prediction, same inputs the KAN gets, as a non-KAN reference point
    _, direct = fit_hgb(lx[R], D[R][:, oh], lx[E])
    out["direct_dOH_gbm"] = r2(direct, D[E][:, oh])

    # 4. how fast does OH actually move? a radical at steady state barely drifts per step
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

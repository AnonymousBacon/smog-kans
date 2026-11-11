import numpy as np

import stoich
from smogkan.data import FIXED_SPECIES


def rmse_r2(pred_ppb, true_ppb):
    rmse = np.sqrt(np.mean((pred_ppb - true_ppb) ** 2, axis=0))
    ss_res = np.sum((pred_ppb - true_ppb) ** 2, axis=0)
    ss_tot = np.sum((true_ppb - true_ppb.mean(axis=0)) ** 2, axis=0)
    return rmse, 1 - ss_res / ss_tot


def test_scores(species, pred_ppb, true_ppb, rmse, r2, flux=None):
    m = {
        "test_rmse_ppb": dict(zip(species, map(float, rmse))),
        "test_r2": dict(zip(species, map(float, r2))),
        "test_r2_mean": float(np.mean(r2)),
        "test_r2_mean_no_oh": float(np.mean([v for s, v in zip(species, r2) if s != "OH"])),
        "test_r2_oh": float(r2[species.index("OH")]) if "OH" in species else float("nan"),
    }
    # r2 over the original 11 species keeps 16-species runs comparable with best_oh5.
    active = [i for i, s in enumerate(species) if s not in FIXED_SPECIES]
    m["test_r2_mean_11"] = float(np.mean(r2[active]))
    if len(active) < len(species):
        E = stoich.atoms(species)
        turnover = np.abs(true_ppb) @ np.abs(E.T)
        m["atom_residual_pred"] = dict(zip("CHNO", map(float, np.abs(pred_ppb @ E.T).max(axis=0)
                                                       / turnover.mean(axis=0))))
    if flux is not None:
        m["flux_ppb"] = {f"R{j + 1}": {"mean": float(flux[:, j].mean()),
                                       "std": float(flux[:, j].std()),
                                       "frac_negative": float((flux[:, j] < 0).mean())}
                         for j in range(flux.shape[1])}
    return m


def run_stats(rc, model, log, device, total_time):
    step_time = np.array(log["step_time"])
    return {
        "name": rc["name"],
        "device": str(device),
        "width_final": [int(w if np.isscalar(w) else w[0]) for w in model.width],
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "steps_run": len(step_time),
        "total_time_s": total_time,
        "sec_per_step_mean": float(step_time.mean()),
        "sec_per_step_median": float(np.median(step_time)),
        "closure_evals_mean": float(np.mean(log["closure_evals"])),
        "final_train_mse": float(log["train_mse"][-1]) if log["train_mse"] else float("nan"),
        "final_train_mse_uw": float(log["train_mse_uw"][-1]) if log["train_mse_uw"] else float("nan"),
        "best_val_mse": float(np.nanmin(np.array(log["val_mse"], dtype=float))),
        "best_val_mse_uw": float(np.nanmin(np.array(log["val_mse_uw"], dtype=float))),
        "species_weights": rc.get("species_weights") or {},
        "transform": rc.get("transform", "log1p"),
    }

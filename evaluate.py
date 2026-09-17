import sys
import io
import contextlib
import numpy as np
import torch

from symbolic_viz import parse_symbolic_log, plot_symbolic_snapping
import config as cfg


def eval_model(model, dataset, scalerY):
    with torch.no_grad():
        preds_scaled = model(dataset['test_input'])
    mse       = torch.mean((preds_scaled - dataset['test_label']) ** 2).item()
    preds_ppb = scalerY.inverse_transform(preds_scaled.cpu().numpy())
    actual    = scalerY.inverse_transform(dataset['test_label'].cpu().numpy())
    rmse_per  = np.sqrt(np.mean((preds_ppb - actual) ** 2, axis=0))
    return mse, rmse_per, preds_ppb, actual


def run_symbolic_fitting(model, dataset, scalerY, species_names, mse, rmse_per, weighted_mse):
    # symbolic fitting: snap well-fit edges to closed-form functions for
    # interpretability, then briefly retune their affine constants.
    # edges killed by pruning get snapped to the constant '0' automatically.
    if np.isnan(mse):
        print("\nSkipping auto_symbolic: model is already producing NaN, retrain first")
        preds, actual_ppb = eval_model(model, dataset, scalerY)[2:]
        return model, mse, rmse_per, preds, actual_ppb

    rmse_before = rmse_per.copy()

    model.save_act         = True
    model.symbolic_enabled = True
    with torch.no_grad():
        model(dataset['train_input'])

    # tee auto_symbolic's per-edge decisions: they stream to the console as usual
    # and are also kept so the snapping figure can be built from them
    _sym_log     = io.StringIO()
    _real_stdout = sys.stdout

    class _Tee:
        def write(self, s):
            _real_stdout.write(s)
            _sym_log.write(s)
            return len(s)

        def flush(self):
            _real_stdout.flush()

    with contextlib.redirect_stdout(_Tee()):
        model.auto_symbolic(r2_threshold=cfg.SYMBOLIC_R2_MIN)

    model.fit(dataset, opt="LBFGS", steps=cfg.SYMBOLIC_FIT_STEPS, lamb=0, loss_fn=weighted_mse,
              update_grid=False)
    model.save_act = False

    plot_symbolic_snapping(
        parse_symbolic_log(_sym_log.getvalue()),
        [int(w) for w in model.width_in],
        species_names, cfg.SYMBOLIC_R2_MIN, cfg.FIG_DIR + "symbolic_snapping.png",
    )

    mse, rmse_per, preds, actual_ppb = eval_model(model, dataset, scalerY)
    print(f"\nAfter auto_symbolic: Test MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
    for name, before, after in zip(species_names, rmse_before, rmse_per):
        flag = "  worse" if after > before else ""
        print(f"  {name:10s}: {before:.4f} -> {after:.4f} ppb/step{flag}")

    return model, mse, rmse_per, preds, actual_ppb

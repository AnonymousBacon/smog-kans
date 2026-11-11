# kept so `import experiment` and `experiment.run(...)` keep working; the code lives in smogkan/.
from smogkan import training
from smogkan.data import (FIXED_SPECIES, N_STEPS_PER_EXP, build_dataset, drop_outlier_exps, load_all_species,
                          load_csv, load_npz, load_raw, rows_for, split_experiments, transform_inputs)
from smogkan.plots import plot_graph, plot_loss, plot_rmse, plot_scatter
from smogkan.runner import run
from smogkan.training import (DTYPES, cache_acts, eval_mse, fluxes, grid_coverage, make_model, make_opt,
                              make_weights, mse, predict, sync, train_stage)


# experiment.INTERRUPTED reads the live flag in smogkan.training instead of a copy taken at import.
def __getattr__(name):
    if name == "INTERRUPTED":
        return training.INTERRUPTED
    raise AttributeError(f"module 'experiment' has no attribute {name!r}")

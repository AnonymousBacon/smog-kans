import sys
import functools
sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import numpy as np

import config as cfg
import data
import model
import train
import evaluate
import visualize

X_clean, D_active, species_names, n_species = data.load_and_clean_data()

# distribution plots
visualize.plot_distributions(np.log1p(X_clean), D_active, species_names)

# scale
scalerX, scalerY, X_train, Y_train, X_test, Y_test, dataset = data.scale_and_split(X_clean, D_active)

# training
KAN_WIDTH = model.build_kan_width(n_species)
print(f"\nArchitecture: {KAN_WIDTH}")

species_weight = model.build_species_weight(species_names, cfg.device)
weighted_mse   = model.make_weighted_mse(species_weight)

# save preprocessed splits + scaler + weights so baselines.py trains on the same data/weights
data.save_splits(X_train, Y_train, X_test, Y_test, species_names, species_weight, scalerY)

trained_model, train_loss, test_loss, best_val = train.run_training_pipeline(dataset, KAN_WIDTH, weighted_mse)

# loss curves
visualize.plot_loss_curves()

# evaluate
mse, rmse_per, preds, actual_ppb = evaluate.eval_model(trained_model, dataset, scalerY)

print(f"\nTest MSE: {mse:.6f} | Mean RMSE: {rmse_per.mean():.4f} ppb/step")
for name, rmse in zip(species_names, rmse_per):
    print(f"  {name:10s}: {rmse:.4f} ppb/step")

trained_model, mse, rmse_per, preds, actual_ppb = evaluate.run_symbolic_fitting(
    trained_model, dataset, scalerY, species_names, mse, rmse_per, weighted_mse,
)

# ols scatter: predicted vs actual per species
visualize.plot_scatter(preds, actual_ppb, species_names, n_species)

# save predictions and KAN graph
data.save_predictions(preds, actual_ppb, species_names)
visualize.plot_kan_graph(trained_model, dataset, species_names, n_species)

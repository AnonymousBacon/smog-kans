import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

from sklearn.preprocessing import MinMaxScaler

import joblib

n_points_per_experiment = 13  # 1 hours * 1 step/5 minutes + 1 for the first step
n_steps_per_experiment = n_points_per_experiment - 1

OUTPUT_DIR = "/Users/baron/Documents/KAN testing/data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)   # creates the folder if it doesn't exist

def load_data(folder="./", file="...", nrows=None):
    ...
    df = pd.read_csv(folder + file, nrows=nrows)   # nrows=None reads everything
    # Load reference model output:
    #   C -- concentration values

    # Additionally, calculate and output
    #   D -- tendency values (units of concentration)
    print("loading concentration data")
    # read dataframe from csv
    df = pd.read_csv(folder + file)

    # Get concentration values C, which are all columns except the first two
    C = df.iloc[:, 2:].values

    # Get tendency values D, which are the difference between consecutive concentration values
    D = np.delete(
        np.diff(C, axis=0),
        list(range(n_steps_per_experiment, C.shape[0] - 1, n_points_per_experiment)),
        axis=0,
    )

    # Get C of active species, ignoring H2O, O2, and buildup HNO3, CO, H2
    ignore_species = ["H2O", "O2", "HNO3", "CO", "H2"]
    active_species_columns = [
        col
        for col in df.columns
        if col.split(" ")[0] not in ignore_species and col.endswith("[ppb]")
    ]
    C_active = df[active_species_columns].values

    return df, C, D, C_active




def ppb_to_molec_cm3(ppb, tempk, press):
    """
    Convert concentration from parts per billion (ppb) to molecules per cubic centimeter (molec/cm^3).

    Parameters:
    - ppb: Concentration in parts per billion (float).
    - tempk: Temperature in Kelvin (float).
    - press: Pressure in atm (float).

    Returns:
    - Concentration in molecules per cubic centimeter (float).
    """
    molec_cm3 = ppb / 1e9 * press / 0.0821 / tempk * 6.022e23 / 1e3
    return molec_cm3


# def createIO(
#     C, D, C_active, conversion=True, conversion_fxn=ppb_to_molec_cm3, downsampling=True
# ):
#     # This function creates input X and output Y train and test sets, as well as test C and J
#     # Inputs:
#     #   C -- concentration values (ppb)
#     #   D -- tendency values (units of concentration) over a 5 minute time step

#     # Outputs:
#     #   X_train --  scaled input values for training and validating NN
#     #   Y_train --  target values for training and validating NN
#     #   X_test  --  unscaled input values for testing/evaluating NN
#     #   Y_test  --  target values for testing/evaluating NN
#     #   C_test  --  concentrations of test data
#     #   scalerX  --  scaling function used to scale NN inputs X_train and X_test

#     print("creating input and output data")
#     X = C_active
#     # delete the last step of each experiment
#     X = np.delete(
#         C_active,
#         list(range(n_steps_per_experiment, C_active.shape[0], n_points_per_experiment)),
#         axis=0,
#     )

#     Y = D

#     ## convert to ppb?
#     if conversion:  # convert c_active to number density
#         print("converting to num density")
#         T, P = 298.0, 1.0
#         C_active = conversion_fxn(C_active, T, P)

#     # Create a train/test split
#     split = 0.90
#     trainsplit = int(split * X.shape[0])
#     print("train size:", trainsplit)
#     testsplit = int(round(1 - split, 2) * X.shape[0])
#     print("test size:", testsplit)
#     num_test_exps = int(testsplit / (n_steps_per_experiment))
#     print("number of test experiments:", num_test_exps)

#     X_train_raw = X[0:trainsplit, :]
#     Y_train_raw = Y[0:trainsplit, :]
#     X_test_raw = X[trainsplit:, :]
#     Y_test_raw = Y[trainsplit:, :]
#     C_test_raw = C[int(trainsplit * 13 / 12) :, :]

#     # Scale input data
#     scalerX = MinMaxScaler()
#     scalerX.fit(X_train_raw)
#     X_train = scalerX.transform(X_train_raw)

#     # Scale output data
#     scalerY = MinMaxScaler()
#     scalerY.fit(Y_train_raw)
#     Y_train = scalerY.transform(Y_train_raw)

#     if downsampling:
#         # Reshape training data into days for sampling
#         X_train = np.reshape(
#             X_train_raw, [num_test_exps, n_steps_per_experiment, C_active.shape[1]]
#         )

#         Y_train = np.reshape(
#             Y_train_raw, [num_test_exps, n_steps_per_experiment, C.shape[1]]
#         )

#     # Reshape testing data into days
#     X_test = np.reshape(
#         X_test_raw, [num_test_exps, n_steps_per_experiment, C_active.shape[1]]
#     )
#     Y_test = np.reshape(Y_test_raw, [num_test_exps, n_steps_per_experiment, C.shape[1]])
#     C_test = np.reshape(
#         C_test_raw, [num_test_exps, n_points_per_experiment, C.shape[1]]
#     )

#     return X_train, Y_train, X_test, Y_test, C_test, scalerX, scalerY


def createIO_DG(C, D, C_active, conversion_fxn):
    # This function creates input X and output Y train and test sets, as well as test C and J
    # Inputs:
    #   C -- concentration values (ppb)
    #   D -- tendency values (units of concentration) over a 5 minute time step

    # Outputs:
    #   X_train --  scaled input values for training and validating NN
    #   Y_train --  target values for training and validating NN
    #   X_test  --  unscaled input values for testing/evaluating NN
    #   Y_test  --  target values for testing/evaluating NN
    #   C_test  --  concentrations of test data
    #   scalerX  --  scaling function used to scale NN inputs X_train and X_test

    print("creating input and output data")
    # 1: create X and Y datasets
    X = C_active
    # delete the last step of each experiment
    X = np.delete(
        C_active,
        list(range(n_steps_per_experiment, C_active.shape[0], n_points_per_experiment)),
        axis=0,
    )

    Y = D

    # 2: Create a train/test split
    split = 0.90
    trainsplit = int(split * X.shape[0])
    print("train size:", trainsplit)
    num_train_exps = int(trainsplit / (n_steps_per_experiment))
    print("num train experiments:", num_train_exps)
    testsplit = int(round(1 - split, 2) * X.shape[0])
    print("test size:", testsplit)
    num_test_exps = int(testsplit / (n_steps_per_experiment))
    print("number of test experiments:", num_test_exps)

    X_train_raw = X[0:trainsplit, :]
    Y_train_raw = Y[0:trainsplit, :]
    X_test_raw = X[trainsplit:, :]
    Y_test_raw = Y[trainsplit:, :]
    C_test_raw = C[int(trainsplit * 13 / 12) :, :]

    # 3: DO DOWNSAMPLING
    # 3a sample 11800 runs
    samp_arr = np.random.randint(0, X_train_raw.shape[0], size=11800)

    Xtrain_clipped = X_train_raw[samp_arr, :]
    # and reshape
    # Xtrain_reshaped = Xtrain_clipped.reshape(Xtrain_clipped.shape[0]*Xtrain_clipped.shape[1], -1)

    Ytrain_clipped = Y_train_raw[samp_arr, :]
    # and reshape
    # Ytrain_reshaped = Ytrain_clipped.reshape(Ytrain_clipped.shape[0]*Ytrain_clipped.shape[1], -1)

    # 3b now look at high OH and H2O2 TENDENCIES (i.e. Y_train)
    OH_cond = lambda x: np.abs(x) > 0.005  # x here are arrays. output should be array
    H2O2_cond = lambda x: np.abs(x) > 0.5

    xt_hi_OH = X_train_raw[OH_cond(Y_train_raw[:, 6])]
    yt_hi_OH = Y_train_raw[OH_cond(Y_train_raw[:, 6])]
    print("Hi OH:", xt_hi_OH.shape)

    xt_hi_H2O2 = X_train_raw[H2O2_cond(Y_train_raw[:, 5])]
    yt_hi_H2O2 = Y_train_raw[H2O2_cond(Y_train_raw[:, 5])]
    print("Hi H2O2:", yt_hi_H2O2.shape)

    # 3c concat X's and Y's all together
    Xtrain_ds = np.concatenate((Xtrain_clipped, xt_hi_OH, xt_hi_H2O2), axis=0)
    Xtrain_fin = np.unique(Xtrain_ds, axis=0)

    Ytrain_ds = np.concatenate((Ytrain_clipped, yt_hi_OH, yt_hi_H2O2), axis=0)
    Ytrain_fin = np.unique(Ytrain_ds, axis=0)

    # 4 convert to num density
    T, P = 298.0, 1.0
    Xtrain_fin = conversion_fxn(Xtrain_fin, T, P)
    Ytrain_fin = conversion_fxn(Ytrain_fin, T, P)
    X_test_raw = conversion_fxn(X_test_raw, T, P)
    Y_test_raw = conversion_fxn(Y_test_raw, T, P)
    C_test_raw = conversion_fxn(C_test_raw, T, P)

    ## Split xtrain, ytrain into training and val datasets!!
    len_train = Xtrain_fin.shape[0]
    val_split = 0.8

    # shuffle training data prior to splitting
    perm = np.random.permutation(len_train)
    Xtrain_fin, Ytrain_fin = Xtrain_fin[perm], Ytrain_fin[perm]

    # now split data
    X_train, Y_train = (
        Xtrain_fin[: int(len_train * val_split)],
        Ytrain_fin[: int(len_train * val_split)],
    )
    X_val, Y_val = (
        Xtrain_fin[int(len_train * val_split) :],
        Ytrain_fin[int(len_train * val_split) :],
    )

    # 6 Scale TRAIN AND VAL data

    # XTrain
    scalerXT = MinMaxScaler()
    scalerXT.fit(X_train)
    X_train = scalerXT.transform(X_train)

    # YTrain
    scalerYT = MinMaxScaler()
    scalerYT.fit(Y_train)
    Y_train = scalerYT.transform(Y_train)

    # XVal
    scalerXV = MinMaxScaler()
    scalerXV.fit(X_val)
    X_val = scalerXV.transform(X_val)

    # YVal
    scalerYV = MinMaxScaler()
    scalerYV.fit(Y_val)
    Y_val = scalerYV.transform(Y_val)

    # Reshape testing data into days
    X_test = np.reshape(
        X_test_raw, [num_test_exps, n_steps_per_experiment, C_active.shape[1]]
    )
    Y_test = np.reshape(Y_test_raw, [num_test_exps, n_steps_per_experiment, C.shape[1]])
    C_test = np.reshape(
        C_test_raw, [num_test_exps, n_points_per_experiment, C.shape[1]]
    )

    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        C_test,
        scalerXT,
        scalerYT,
        scalerXV,
        scalerYV,
    )


def molec_cm3_to_ppb(molec_cm3, tempk, press):
    """
    Convert concentration from molecules per cubic centimeter (molec/cm^3) to parts per billion (ppb).

    Parameters:
    - molec_cm3: Concentration in molecules per cubic centimeter (float).
    - tempk: Temperature in Kelvin (float).
    - press: Pressure in atm (float).

    Returns:
    - Concentration in parts per billion (float).
    """
    ppb = molec_cm3 / 6.022e23 * 1e3 * 0.0821 * tempk / press * 1e9
    return ppb


if __name__ == "__main__":
    # Load data and create input/output data for NNs
    # Provide own path if reproducing on another system
    INPUT_DIR = "/Users/baron/Documents/KAN testing/data/"   # where the CSV actually lives
    filepath = INPUT_DIR
    df, C, D, C_active = load_data(
        folder=filepath,
        file="experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv",
        nrows=13000,   # 13 rows/experiment × 1000 experiments
    )

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        C_test,
        scalerXT,
        scalerYT,
        scalerXV,
        scalerYV,
    ) = createIO_DG(C, D, C_active, ppb_to_molec_cm3)

    print("train data is now of dimension:", X_train.shape)
    # space to save datasets for easy loading/parsing later
    np.savez(
        OUTPUT_DIR + "/split_data_0.34_SHUFFLED.npz",
        xtrain=X_train,
        ytrain=Y_train,
        xval=X_val,
        yval=Y_val,
        xtest=X_test,
        ytest=Y_test,
        ctest=C_test,
    )

    # ## convert to torch-able data format and save these!!
    # xtrain_t, ytrain_t = torch.Tensor(X_train), torch.Tensor(Y_train)
    # xtest_t, ytest_t = torch.Tensor(X_test), torch.Tensor(Y_test)
    # ctest_t = torch.Tensor(C_test)
    # train_dataset = TensorDataset(xtrain_t, ytrain_t)
    # test_dataset = TensorDataset(xtest_t, ytest_t)
    # torch.save(train_dataset, "data/train_tensor.pt")
    # torch.save(test_dataset, "data/test_tensor.pt")
    # torch.save(ctest_t, "data/ctest_tensor.pt")

    # ... and save scalers
    scalerxt, scaleryt = "scalerXT.save", "scalerYT.save"
    scalerxv, scaleryv = "scalerXV.save", "scalerYV.save"
    joblib.dump(
        scalerXT, OUTPUT_DIR + scalerxt
    )
    joblib.dump(
        scalerYT, OUTPUT_DIR + scaleryt
    )
    joblib.dump(
        scalerXV, OUTPUT_DIR + scalerxv
    )
    joblib.dump(
        scalerYV, OUTPUT_DIR + scaleryv
    )
    # reference for loading scalers
    # scaler = joblib.load(scaler_filename)

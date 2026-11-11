import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

from sklearn.preprocessing import MinMaxScaler

import joblib

n_points_per_experiment = 13
n_steps_per_experiment = n_points_per_experiment - 1

OUTPUT_DIR = "/Users/baron/Documents/KAN testing/data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(folder="./", file="...", nrows=None):
    ...
    df = pd.read_csv(folder + file, nrows=nrows)

    print("loading concentration data")
    df = pd.read_csv(folder + file)

    C = df.iloc[:, 2:].values

    D = np.delete(
        np.diff(C, axis=0),
        list(range(n_steps_per_experiment, C.shape[0] - 1, n_points_per_experiment)),
        axis=0,
    )

    ignore_species = ["H2O", "O2", "HNO3", "CO", "H2"]
    active_species_columns = [
        col
        for col in df.columns
        if col.split(" ")[0] not in ignore_species and col.endswith("[ppb]")
    ]
    C_active = df[active_species_columns].values

    return df, C, D, C_active


def ppb_to_molec_cm3(ppb, tempk, press):
    molec_cm3 = ppb / 1e9 * press / 0.0821 / tempk * 6.022e23 / 1e3
    return molec_cm3


def createIO_DG(C, D, C_active, conversion_fxn):


    print("creating input and output data")
    X = C_active
    X = np.delete(
        C_active,
        list(range(n_steps_per_experiment, C_active.shape[0], n_points_per_experiment)),
        axis=0,
    )

    Y = D

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

    samp_arr = np.random.randint(0, X_train_raw.shape[0], size=11800)

    Xtrain_clipped = X_train_raw[samp_arr, :]

    Ytrain_clipped = Y_train_raw[samp_arr, :]

    OH_cond = lambda x: np.abs(x) > 0.005
    H2O2_cond = lambda x: np.abs(x) > 0.5

    xt_hi_OH = X_train_raw[OH_cond(Y_train_raw[:, 6])]
    yt_hi_OH = Y_train_raw[OH_cond(Y_train_raw[:, 6])]
    print("Hi OH:", xt_hi_OH.shape)

    xt_hi_H2O2 = X_train_raw[H2O2_cond(Y_train_raw[:, 5])]
    yt_hi_H2O2 = Y_train_raw[H2O2_cond(Y_train_raw[:, 5])]
    print("Hi H2O2:", yt_hi_H2O2.shape)

    Xtrain_ds = np.concatenate((Xtrain_clipped, xt_hi_OH, xt_hi_H2O2), axis=0)
    Xtrain_fin = np.unique(Xtrain_ds, axis=0)

    Ytrain_ds = np.concatenate((Ytrain_clipped, yt_hi_OH, yt_hi_H2O2), axis=0)
    Ytrain_fin = np.unique(Ytrain_ds, axis=0)

    T, P = 298.0, 1.0
    Xtrain_fin = conversion_fxn(Xtrain_fin, T, P)
    Ytrain_fin = conversion_fxn(Ytrain_fin, T, P)
    X_test_raw = conversion_fxn(X_test_raw, T, P)
    Y_test_raw = conversion_fxn(Y_test_raw, T, P)
    C_test_raw = conversion_fxn(C_test_raw, T, P)

    len_train = Xtrain_fin.shape[0]
    val_split = 0.8

    perm = np.random.permutation(len_train)
    Xtrain_fin, Ytrain_fin = Xtrain_fin[perm], Ytrain_fin[perm]

    X_train, Y_train = (
        Xtrain_fin[: int(len_train * val_split)],
        Ytrain_fin[: int(len_train * val_split)],
    )
    X_val, Y_val = (
        Xtrain_fin[int(len_train * val_split) :],
        Ytrain_fin[int(len_train * val_split) :],
    )


    scalerXT = MinMaxScaler()
    scalerXT.fit(X_train)
    X_train = scalerXT.transform(X_train)

    scalerYT = MinMaxScaler()
    scalerYT.fit(Y_train)
    Y_train = scalerYT.transform(Y_train)

    scalerXV = MinMaxScaler()
    scalerXV.fit(X_val)
    X_val = scalerXV.transform(X_val)

    scalerYV = MinMaxScaler()
    scalerYV.fit(Y_val)
    Y_val = scalerYV.transform(Y_val)

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
    ppb = molec_cm3 / 6.022e23 * 1e3 * 0.0821 * tempk / press * 1e9
    return ppb


if __name__ == "__main__":
    INPUT_DIR = "/Users/baron/Documents/KAN testing/data/"
    filepath = INPUT_DIR
    df, C, D, C_active = load_data(
        folder=filepath,
        file="experiments_11e5_1hour_5mins_falsecombinatoricratelaws.csv",
        nrows=13000,
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

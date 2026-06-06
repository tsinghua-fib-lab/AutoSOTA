import pandas as pd
import numpy as np
import torch
import uuid

import torchtuples as tt  # Some useful functions
import torch.nn as nn
import torch.nn.functional as F

from pycox.models.cox_time import MLPVanillaCoxTime

from pycox.models import (
    CoxTime,
    PMF,
    MTLR,
    BCESurv,
    DeepHitSingle,
    CoxCC,
    CoxPH,
    PCHazard,
    LogisticHazard,
)

from benchmark.dysurv_functions import DySurv, Loss
from benchmark.dqs.torch.distribution import DistributionLinear
from benchmark.dqs.torch.loss import NegativeLogLikelihood
from benchmark.dqs.torch.survival_analysis_cnll import MLP

batch_size = 256
epochs = 1000
num_nodes = [16, 16]
batch_norm = True
dropout = 0.1


def find_surv_at_new_times(surv_at_train_times, new_times):
    new_times = pd.DataFrame({"time": new_times})
    new_times["original_index"] = new_times.index

    surv_df = surv_at_train_times.copy()
    surv_df = surv_df.reset_index().rename(
        columns={"index": "time"}
    )  # make time a column
    surv_df = surv_df.rename(columns={"duration": "time"})
    surv_df["time"] = surv_df["time"].astype(np.float32)

    new_times_sorted = new_times.sort_values("time")
    surv_df_sorted = surv_df.sort_values("time")

    merged = pd.merge_asof(
        new_times_sorted, surv_df_sorted, on="time", direction="backward"
    )
    merged = merged.sort_values("original_index").drop(
        columns=["time", "original_index"]
    )

    surv_new_times = merged.reset_index(drop=True).transpose()
    return np.clip(surv_new_times.to_numpy(), 0.0, 1.0)


def fit_CoxTime(data_train, data_val, data_test, new_times):

    print("\nFitting CoxTime ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = CoxTime.label_transform()
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    # Hyperparameter search for learning rate
    lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
    model.optimizer.set_lr(lrfinder.get_best_lr())

    # Train model
    model.fit(
        x_train,
        y_train,
        batch_size,
        epochs,
        val_data=val.repeat(10).cat(),
        verbose=False,
    )

    # Predict
    _ = model.compute_baseline_hazards()
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_PMF(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting PMF ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = PMF.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = PMF(net, tt.optim.Adam, duration_index=labtrans.cuts)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=4)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_MTLR(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting MTLR ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = MTLR.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=6)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_BCESurv(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting BCESurv ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = BCESurv.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))
    y_test = labtrans.transform(*(df_test["time"], df_test["event"]))

    train = tt.tuplefy(x_train, y_train)
    val = tt.tuplefy(x_val, y_val)
    test = tt.tuplefy(x_test, y_test)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = BCESurv(net, tt.optim.Adam(), duration_index=labtrans.cuts)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=8)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Fit
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_DeepHitSingle(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting DeepHitSingle ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = DeepHitSingle(
        net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts
    )

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=4)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_CoxCC(data_train, data_val, data_test, new_times):

    print("\nFitting CoxCC ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    y_train = (df_train["time"], df_train["event"])
    y_val = (df_val["time"], df_val["event"])

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = 1
    output_bias = False
    net = tt.practical.MLPVanilla(
        in_features,
        num_nodes,
        out_features,
        batch_norm,
        dropout,
        output_bias=output_bias,
    )
    model = CoxCC(net, tt.optim.Adam)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(
        x_train,
        y_train,
        batch_size,
        epochs,
        val_data=val.repeat(10).cat(),
        verbose=False,
    )

    # Predict
    _ = model.compute_baseline_hazards()
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_Deepsurv(data_train, data_val, data_test, new_times):

    print("\nFitting Deepsurv ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    y_train = (df_train["time"], df_train["event"])
    y_val = (df_val["time"], df_val["event"])

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = 1
    output_bias = False
    net = tt.practical.MLPVanilla(
        in_features,
        num_nodes,
        out_features,
        batch_norm,
        dropout,
        output_bias=output_bias,
    )
    model = CoxPH(net, tt.optim.Adam)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(
        x_train, y_train, batch_size, tolerance=10, verbose=False
    )
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(
        x_train,
        y_train,
        batch_size,
        epochs,
        val_data=val,
        val_batch_size=batch_size,
        verbose=False,
    )

    # Predict
    _ = model.compute_baseline_hazards()
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_PCHazard(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting PCHazard ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = PCHazard.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = PCHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=8)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_LogisticHazard(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting LogisticHazard ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = LogisticHazard.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    val = tt.tuplefy(x_val, y_val)

    # simple MLP with two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )
    model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)

    # Hyperparameter search for learning rate
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=8)
    model.optimizer.set_lr(lr_finder.get_best_lr())

    # Train model
    _ = model.fit(x_train, y_train, batch_size, epochs, val_data=val, verbose=False)

    # Predict
    surv_at_train_times = model.predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_Dysurv(data_train, data_val, data_test, new_times, num_durations=10):

    print("\nFitting Dysurv ...")

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = df_train["x"]
    x_test = df_test["x"]
    x_val = df_val["x"]

    labtrans = LogisticHazard.label_transform(num_durations)
    y_train = labtrans.fit_transform(*(df_train["time"], df_train["event"]))
    y_val = labtrans.transform(*(df_val["time"], df_val["event"]))

    # train_target = np.zeros((x_train.shape[0], x_train.shape[1], 1))
    # train_target[:, :, -1] = y_train[0].reshape(-1, 1)
    # x_train = np.append(x_train, train_target, axis=2)
    # val_target = np.zeros((x_val.shape[0], x_val.shape[1], 1))
    # val_target[:, :, -1] = y_val[0].reshape(-1, 1)
    # x_val = np.append(x_val, val_target, axis=2)
    # train = tt.tuplefy(x_train, (y_train, x_train))
    # val = tt.tuplefy(x_val, (y_val, x_val))

    train = tt.tuplefy(x_train, (y_train, x_train))
    val = tt.tuplefy(x_val, (y_val, x_val))

    in_features = x_train.shape[1]
    encoded_features = 20  # use 20 latent factors
    out_features = labtrans.out_features
    net = DySurv(in_features, encoded_features, out_features)

    loss = Loss([0.33, 0, 0])
    model = LogisticHazard(
        net, tt.optim.Adam(0.001), duration_index=labtrans.cuts, loss=loss
    )

    metrics = dict(
        loss_surv=Loss([1, 0, 0]), loss_ae=Loss([0, 1, 0]), loss_kd=Loss([0, 0, 1])
    )
    callbacks = [tt.cb.EarlyStopping()]

    batch_size = 256
    epochs = 1000
    _ = model.fit(
        *train,
        batch_size,
        epochs,
        val_data=val,
        metrics=metrics,
        callbacks=callbacks,
        verbose=False
    )

    surv_at_train_times = model.interpolate(10).predict_surv_df(x_test)

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_at_train_times, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_at_train_times, new_times)

    return -surv_test, surv_test, surv_new_times


def fit_sumo_net(data_train, data_val, data_test, new_times):

    # load only for sumo net used because it imports a local version if pycox
    # from benchmark.sumo_net.hyperopt_class import *
    # from benchmark.sumo_net.dataloaders import get_dataloader

    print("\nFitting Sumo Net ...")

    df_test = data_test["np.array"]
    temp_folder = str(uuid.uuid4())

    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        "bounding_op": [torch.relu],  # torch.sigmoid, torch.relu, torch.exp,
        "transformation": [torch.nn.Tanh()],
        "depth_x": [2],
        "width_x": [16],  # adapt for smaller time net
        "depth_t": [1],
        "width_t": [1],  # ads
        "depth": [2],
        "width": [16],
        "bs": [50],
        "lr": [1e-2],
        "direct_dif": ["autograd"],
        "dropout": [0.1],
        "eps": [1e-4],
        "weight_decay": [0],
        "T_losses": [90],
        "alpha": [0.2],
        "sigma": [0.1],
        "num_dur": [20],
    }
    dataloader = get_dataloader(
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        bs=hyper_param_space["bs"][0],
    )

    job_params = {
        "d_out": 1,
        "dataset_string": "support",
        "seed": 1,  # ,np.random.randint(0,9999),
        "total_epochs": 50,
        "device": "cpu",
        "patience": 50,
        "hyperits": 5,
        "selection_criteria": "train",
        "grid_size": 100,
        "test_grid_size": 100,
        "validation_interval": 2,
        # 'net_type':'survival_net_basic',
        # 'net_type':'weibull_net',
        # 'net_type':'lognormal_net',
        "net_type": "survival_net_basic",
        # 'net_type':'cox_time_benchmark',
        # 'net_type':'deephit_benchmark',
        # 'net_type':'cox_linear_benchmark',
        # 'net_type':'deepsurv_benchmark',
        "objective": "S_mean",
        "fold_idx": 1,
        "savedir": temp_folder,
        "use_sotle": False,
        "chunks": 50,
        "max_series_accumulation": 25000,
        "validate_train": False,
        "dataloader": dataloader,
    }
    training_obj = hyperopt_training(
        job_param=job_params, hyper_param_space=hyper_param_space
    )
    # training_obj.debug=True
    training_obj.run()
    results = training_obj.post_process()
    surv_on_grid = results["survival"]

    # Find surv at test times
    surv_test = find_surv_at_new_times(surv_on_grid, df_test["time"])

    # Find surv at new times
    surv_new_times = find_surv_at_new_times(surv_on_grid, new_times)

    # Delete temporary folder
    folder_path = os.path.join(os.getcwd(), temp_folder)
    shutil.rmtree(folder_path)

    return -surv_test, surv_test, surv_new_times


def fit_dqs(data_train, data_val, data_test, new_times):

    df_train = data_train["np.array"]
    df_test = data_test["np.array"]
    df_val = data_val["np.array"]

    x_train = torch.from_numpy(df_train["x"]).clone()
    x_test = torch.from_numpy(df_test["x"]).clone()

    time_train = torch.from_numpy(df_train["time"]).clone()
    time_test = torch.from_numpy(df_test["time"]).clone()

    event_train = torch.from_numpy(df_train["event"]).clone() == 1

    new_times_tensor = torch.from_numpy(new_times).clone()

    # prepare model
    max_time = max(max(time_train), max(time_test), max(new_times_tensor))
    boundaries = torch.linspace(0.0, max_time + 1, 5)
    dist = DistributionLinear(boundaries)
    loss_fn = NegativeLogLikelihood(dist, boundaries)
    mlp = MLP(
        input_len=x_train.shape[1],
        n_output=4,
        n_hidden_layers=2,
        hidden_units=16,
        dropout_rate=dropout,
        use_batch_norm=batch_norm,
    )

    # train model
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(100):
        pred = mlp(x_train)
        loss = loss_fn.loss(pred, time_train, event_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("epoch=%d, loss=%f" % (epoch, loss))

    # Predict nn output on test set
    pred_test = mlp(x_test)

    # Predict survival at test times
    surv_test = torch.zeros(
        (pred_test.shape[0], time_test.shape[0])
    )  # preallocate output tensor
    for i in range(pred_test.shape[0]):
        for j in range(time_test.shape[0]):
            val = torch.stack(
                loss_fn._compute_F(pred_test[i, :].reshape(1, -1), time_test[j]), dim=0
            ).mean(dim=0)
            surv_test[i, j] = 1 - val[0][0].item()
    surv_test = np.clip(surv_test.detach().cpu().numpy(), 0.0, 1.0)

    # Predict survival at new times
    surv_new_times = torch.zeros(
        (pred_test.shape[0], new_times_tensor.shape[0])
    )  # preallocate output tensor
    for i in range(pred_test.shape[0]):
        for j in range(new_times_tensor.shape[0]):
            val = torch.stack(
                loss_fn._compute_F(pred_test[i, :].reshape(1, -1), new_times_tensor[j]),
                dim=0,
            ).mean(dim=0)
            surv_new_times[i, j] = 1 - val[0][0].item()
    surv_new_times = np.clip(surv_new_times.detach().cpu().numpy(), 0.0, 1.0)

    return -surv_test, surv_test, surv_new_times

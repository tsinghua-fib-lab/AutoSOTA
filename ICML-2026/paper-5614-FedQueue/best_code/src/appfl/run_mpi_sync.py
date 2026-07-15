"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import copy
import time
import logging
import numpy as np
import torch.nn as nn
from mpi4py import MPI
from typing import Any
from appfl.misc.data import Dataset
from appfl.misc.utils import (
    validation,
    save_model_iteration,
    client_log,
    create_custom_logger,
    get_appfl_algorithm,
)
from appfl.comm.mpi import MpiCommunicator
from appfl.compressor import Compressor
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def run_server(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_dataset: Dataset = Dataset(),
    dataset_name: str = "MNIST",
    metric: Any = None,
):
    """
    run_server:
        Run PPFL server that updates the global model parameter in a synchronous way.

    Args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function
        num_clients: the number of clients used in PPFL simulation
        test_dataset: optional testing data. If given, validation will run based on this data
        dataset_name: optional dataset name
        metric: evaluation metric function
    """
    device = "cpu"  # server aggregation happens on CPU
    communicator = MpiCommunicator(
        comm, Compressor(cfg) if cfg.enable_compression else None
    )

    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)
    cfg.logginginfo.comm_size = comm.Get_size()
    cfg.logginginfo.DataSet_name = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    ## Run validation if test data is given and the configuration is enabled.
    if cfg.validation and len(test_dataset) > 0:
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    ## Collect the number of data from each client and compute the weights for each client
    num_data = communicator.gather(0, dest=0)
    total_num_data = 0
    for num in num_data:
        total_num_data += num
    weights = [num / total_num_data for num in num_data]
    communicator.scatter(weights, 0)

    ## Synchronous federated learning server
    server = get_appfl_algorithm(
        algorithm_name=cfg.fed.servername,
        args=(weights, model, loss_fn, num_clients, device),
        kwargs=cfg.fed.args,
    )

    start_time = time.time()
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    for iter in range(cfg.num_epochs):
        per_iter_start = time.time()
        server.model.to("cpu")
        global_state = server.model.state_dict()

        communicator.broadcast_global_model(global_state, {"done": False})

        local_states = [None for _ in range(num_clients)]
        model_copy = copy.deepcopy(server.model)
        for _ in range(num_clients):
            client_idx, model = communicator.recv_local_model_from_client(model_copy)
            local_states[client_idx] = model

        cfg.logginginfo.LocalUpdate_time = time.time() - per_iter_start

        global_update_start = time.time()
        server.update(local_states)
        cfg.logginginfo.GlobalUpdate_time = time.time() - global_update_start

        validation_start = time.time()
        if cfg.validation:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)
            if cfg.use_tensorboard:
                writer.add_scalar("server_test_accuracy", test_accuracy, iter)
                writer.add_scalar("server_test_loss", test_loss, iter)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        cfg.logginginfo.Validation_time = time.time() - validation_start
        cfg.logginginfo.PerIter_time = time.time() - per_iter_start
        cfg.logginginfo.Elapsed_time = time.time() - start_time
        cfg.logginginfo.test_loss = test_loss
        cfg.logginginfo.test_accuracy = test_accuracy
        cfg.logginginfo.BestAccuracy = best_accuracy

        server.logging_iteration(cfg, logger, iter)

        ## Saving model
        if (iter + 1) % cfg.checkpoints_interval == 0 or iter + 1 == cfg.num_epochs:
            if cfg.save_model:
                save_model_iteration(iter + 1, server.model, cfg)

        if np.isnan(test_loss):
            break

    ## Notify the clients about the end of the learning
    communicator.cleanup()
    communicator.broadcast_global_model(args={"done": True})

    ## Summary
    server.logging_summary(cfg, logger)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    metric: Any = None,
):
    """
    run_client:
        Run PPFL simulation clients, each of which updates its own local parameters of model.

    args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function
        train_data: training data
        test_data: validation data
        metric: evaluation metric function
    """
    client_idx = comm.Get_rank() - 1
    communicator = MpiCommunicator(
        comm, Compressor(cfg) if cfg.enable_compression else None
    )

    ## log for clients
    output_filename = cfg.output_filename + "_client_%s" % (client_idx)
    outfile = client_log(cfg.output_dirname, output_filename)

    num_data = len(train_data[client_idx])
    communicator.gather(num_data, dest=0)
    weight = None
    weight = communicator.scatter(weight, source=0)

    batchsize = cfg.train_data_batch_size
    if not cfg.batch_training:
        batchsize = len(train_data[client_idx])

    ## Run validation if test data is given or the configuration is enabled
    if cfg.validation and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False
        test_dataloader = None

    client = get_appfl_algorithm(
        algorithm_name=cfg.fed.clientname,
        args=(
            client_idx,
            weight,
            model,
            loss_fn,
            DataLoader(
                train_data[client_idx],
                num_workers=cfg.num_workers,
                batch_size=batchsize,
                shuffle=True,
                pin_memory=True,
            ),
            cfg,
            outfile,
            test_dataloader,
            metric,
        ),
        kwargs=cfg.fed.args,
    )

    while True:
        model = communicator.recv_global_model_from_server(source=0)
        if isinstance(model, tuple):
            model, done = model[0], model[1]["done"]
        else:
            done = False
        if done:
            break
        client.model.load_state_dict(model)
        local_model = client.update()
        communicator.send_local_model_to_server(local_model, dest=0)
    outfile.close()

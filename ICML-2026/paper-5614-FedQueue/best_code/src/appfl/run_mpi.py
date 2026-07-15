"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import copy
import math
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from mpi4py import MPI
from omegaconf import DictConfig
from typing import Any, Union, List
from collections import OrderedDict
from torch.utils.data import DataLoader
from appfl.compressor import Compressor
from appfl.comm.mpi import MpiSyncCommunicator
from appfl.misc.data import Dataset
from appfl.misc.utils import (
    validation,
    save_model_iteration,
    save_partial_model_iteration,
    client_log,
    create_custom_logger,
    get_appfl_algorithm,
)


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
        Run PPFL simulation server that aggregates and updates the global parameters of model

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
    device = "cpu"
    communicator = MpiSyncCommunicator(
        comm, Compressor(cfg) if cfg.enable_compression else None
    )

    ## log for a server
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

    ## Run validation if test data is given or the configuration is enabled.
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
    for rank in range(1, comm.Get_size()):
        for num in num_data[rank].values():
            total_num_data += num
    client_weight = [{}]
    weights = []
    for rank in range(1, comm.Get_size()):
        temp = {}
        for cid, num in num_data[rank].items():
            weights.append(num / total_num_data)
            temp[cid] = num / total_num_data
        client_weight.append(temp)
    communicator.scatter(client_weight, 0)

    ## Synchronous federated learning server
    server = get_appfl_algorithm(
        algorithm_name=cfg.fed.servername,
        args=(
            weights,
            copy.deepcopy(model),
            loss_fn,
            num_clients,
            device,
        ),
        kwargs=cfg.fed.args,
    )

    start_time = time.time()
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    model_copy = copy.deepcopy(server.model)
    for t in range(cfg.num_epochs):
        per_iter_start = time.time()
        ## Local update
        server.model.to("cpu")
        global_model = server.model.state_dict()
        communicator.broadcast_global_model(global_model)
        local_models = communicator.recv_all_local_models_from_clients(
            num_clients, model_copy
        )
        cfg.logginginfo.LocalUpdate_time = time.time() - per_iter_start
        ## Global update
        global_update_start = time.time()
        server.update(local_models)
        cfg.logginginfo.GlobalUpdate_time = time.time() - global_update_start
        ## Validation
        validation_start = time.time()
        if cfg.validation:
            test_loss, test_accuracy = validation(server, test_dataloader, metric)
            if cfg.use_tensorboard:
                writer.add_scalar("server_test_accuracy", test_accuracy, t)
                writer.add_scalar("server_test_loss", test_loss, t)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        cfg.logginginfo.Validation_time = time.time() - validation_start
        cfg.logginginfo.PerIter_time = time.time() - per_iter_start
        cfg.logginginfo.Elapsed_time = time.time() - start_time
        cfg.logginginfo.test_loss = test_loss
        cfg.logginginfo.test_accuracy = test_accuracy
        cfg.logginginfo.BestAccuracy = best_accuracy
        server.logging_iteration(cfg, logger, t)
        ## Saving model
        if (t + 1) % cfg.checkpoints_interval == 0 or t + 1 == cfg.num_epochs:
            if cfg.save_model:
                if cfg.personalization:
                    save_partial_model_iteration(t + 1, server.model, cfg)
                else:
                    save_model_iteration(t + 1, server.model, cfg)
        if np.isnan(test_loss):
            break

    ## Notify the clients about the end of the learning
    communicator.broadcast_global_model(args={"done": True})

    ## Summary
    server.logging_summary(cfg, logger)


def run_client(
    cfg: DictConfig,
    comm: MPI.Comm,
    model: Union[nn.Module, List],
    loss_fn: nn.Module,
    num_clients: int,
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
    communicator = MpiSyncCommunicator(
        comm, Compressor(cfg) if cfg.enable_compression else None
    )
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    num_client_groups = np.array_split(range(num_clients), comm_size - 1)

    ## log for clients
    outfile = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        output_filename = cfg.output_filename + "_client_%s" % (cid)
        outfile[cid] = client_log(cfg.output_dirname, output_filename)

    num_data = {}
    for _, cid in enumerate(num_client_groups[comm_rank - 1]):
        num_data[cid] = len(train_data[cid])
    communicator.gather(num_data, dest=0)
    weights = None
    weights = communicator.scatter(weights, source=0)

    batchsize = {}
    for cid in num_client_groups[comm_rank - 1]:
        batchsize[cid] = cfg.train_data_batch_size
        if not cfg.batch_training:
            batchsize[cid] = len(train_data[cid])

    ## Run validation if test data is given or the configuration is enabled.
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

    if "cuda" in cfg.device:
        ## Check available GPUs if CUDA is used
        num_gpu = torch.cuda.device_count()
        client_per_gpu = math.ceil(num_clients / num_gpu)

    clients = []
    for cid in num_client_groups[comm_rank - 1]:
        if "cuda" in cfg.device:
            gpuindex = int(math.floor(cid / client_per_gpu))
            cfg.device = f"cuda:{gpuindex}"
        clients.append(
            get_appfl_algorithm(
                algorithm_name=cfg.fed.clientname,
                args=(
                    cid,
                    weights[cid],
                    copy.deepcopy(model) if not cfg.personalization else model[cid],
                    loss_fn,
                    DataLoader(
                        train_data[cid],
                        num_workers=cfg.num_workers,
                        batch_size=batchsize[cid],
                        shuffle=cfg.train_data_shuffle,
                        pin_memory=True,
                    ),
                    cfg,
                    outfile[cid],
                    test_dataloader,
                    metric,
                ),
                kwargs=cfg.fed.args,
            )
        )

    while True:
        ## Receive global model
        model = communicator.recv_global_model_from_server(source=0)
        if isinstance(model, tuple):
            model, done = model[0], model[1]["done"]
        else:
            done = False
        if done:
            break
        ## Delete personalized layers
        if cfg.personalization:
            for key in cfg.p_layers:
                del model[key]
        ## Start local update
        local_models = OrderedDict()
        for client in clients:
            cid = client.id
            client.model.load_state_dict(model, strict=not cfg.personalization)
            local_model = client.update()
            local_models[cid] = local_model
        ## Send local models
        communicator.send_local_models_to_server(local_models, 0)

    for client in clients:
        client.outfile.close()

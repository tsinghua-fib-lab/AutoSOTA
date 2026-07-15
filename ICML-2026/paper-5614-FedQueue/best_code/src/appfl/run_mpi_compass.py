"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import copy
import time
import logging
import torch.nn as nn
from typing import Any
from mpi4py import MPI
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from appfl.misc.data import Dataset
from appfl.misc.utils import (
    validation,
    compute_gradient,
    client_log,
    create_custom_logger,
    get_appfl_algorithm,
)
from appfl.compressor import Compressor
from appfl.comm.mpi import MpiCommunicator
from appfl.algorithm import SchedulerCompassMPI, SchedulerDummy


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
        Run PPFL simulation server that aggregates and updates the global parameters of model in an asynchronous way using a scheduler

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
    for num in num_data:
        total_num_data += num
    weights = [num / total_num_data for num in num_data]
    weights = [num / total_num_data for num in num_data]
    communicator.scatter(weights, 0)

    ## Asynchronous federated learning server (aggregator)
    server = get_appfl_algorithm(
        algorithm_name=cfg.fed.servername,
        args=(weights, model, loss_fn, num_clients, device),
        kwargs=cfg.fed.args,
    )
    server.model.to("cpu")
    global_model = server.model.state_dict()

    ## First broadcast the global model
    communicator.broadcast_global_model(global_model)

    ## Obtain the scheduler
    if cfg.fed.servername.startswith("ServerFedCompass"):
        scheduler = SchedulerCompassMPI(
            communicator,
            server,
            cfg.fed.args.num_local_steps,
            num_clients,
            cfg.num_epochs,
            cfg.fed.args.optim_args.lr,
            logger,
            cfg.fed.servername == "ServerFedCompassNova",
            cfg.fed.args.q_ratio,
            cfg.fed.args.lambda_val,
        )
    else:
        scheduler = SchedulerDummy(communicator, server, num_clients, cfg.num_epochs)

    # FedAsync: main global training loop
    start_time = time.time()
    global_step, test_loss, test_accuracy, best_accuracy = 0, 0.0, 0.0, 0.0
    while True:
        scheduler.update()
        global_step += 1
        if (
            scheduler.validation_flag and global_step % cfg.fed.args.val_range == 0
        ) or global_step == cfg.num_epochs:
            validation_start = time.time()
            if cfg.validation:
                test_loss, test_accuracy = validation(server, test_dataloader, metric)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                if cfg.use_tensorboard:
                    # Add them to tensorboard
                    writer.add_scalar(
                        "server_test_accuracy", test_accuracy, global_step
                    )
                    writer.add_scalar("server_test_loss", test_loss, global_step)
            cfg.logginginfo.test_loss = test_loss
            cfg.logginginfo.test_accuracy = test_accuracy
            cfg.logginginfo.BestAccuracy = best_accuracy
            cfg.logginginfo.Elapsed_time = time.time() - start_time
            cfg.logginginfo.Validation_time = time.time() - validation_start
            cfg.logginginfo.PerIter_time = 0  # TODO
            cfg.logginginfo.LocalUpdate_time = 0  # TODO
            cfg.logginginfo.GlobalUpdate_time = 0  # TODO
            if global_step != 1:
                logger.info(server.log_title())
            server.logging_iteration(cfg, logger, global_step - 1)
            if global_step == cfg.num_epochs:
                break

    for i in range(num_clients):
        communicator.send_global_model_to_client(None, {"done": True}, i)
    communicator.cleanup()

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
        Run PPFL simulation clients, each of which updates its own local parameters of model

    Args:
        cfg: the configuration for the FL experiment
        comm: MPI communicator
        model: neural network model to train
        loss_fn: loss function
        num_clients: the number of clients used in PPFL simulation
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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

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
            model, train_configs = model[0], model[1]
            done = train_configs["done"]
        else:
            done = False
            train_configs = {}
        if done:
            break
        client.num_local_steps = (
            client.num_local_steps
            if "step" not in train_configs
            else train_configs["steps"]
        )
        client.optim_args.lr = (
            client.optim_args.lr if "lr" not in train_configs else train_configs["lr"]
        )
        client.model.load_state_dict(model)
        client.update()
        ## Compute gradient if the algorithm is gradient-based
        if cfg.fed.args.gradient_based:
            local_model = compute_gradient(model, client.model)
        else:
            local_model = copy.deepcopy(client.primal_state)
        communicator.send_local_model_to_server(local_model, dest=0)
    outfile.close()

"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import copy
import time
import logging
import torch.nn as nn
from omegaconf import DictConfig
from typing import Union, List, Any
from torch.utils.data import DataLoader
from appfl.misc.utils import (
    save_model_iteration,
    save_partial_model_iteration,
    validation,
    client_log,
    create_custom_logger,
    get_appfl_algorithm,
)
from appfl.misc.data import Dataset


def run_serial(
    cfg: DictConfig,
    model: Union[nn.Module, List],
    loss_fn: nn.Module,
    train_data: Dataset,
    test_data: Dataset = Dataset(),
    dataset_name: str = "MNIST",
    metric: Any = None,
):
    """
    run_serial:
        Run serial simulation of PPFL.
    Args:
        cfg: the configuration for this run
        model (nn.Module or list): if personalization is disabled, neural network model to train. if personalization is enabled, it will be a LIST containing the server and client models (i.e. num_clients+1 models), which can be uninitialized or preloaded with saved weights depending on user's choice to load saved model
        loss_fn: loss function
        train_data: training data
        test_data: optional testing data. If given, validation will run based on this data
        dataset_name: optional dataset name
        metric: evaluation metric function
    """

    ## Server log
    logger = logging.getLogger(__name__)
    logger = create_custom_logger(logger, cfg)
    cfg.logginginfo.comm_size = 1
    cfg.logginginfo.DataSet_name = dataset_name

    ## Using tensorboard to visualize the test loss
    if cfg.use_tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            comment=cfg.fed.args.optim + "_clients_nums_" + str(cfg.num_clients)
        )

    ## Client logs
    outfile = {}
    for k in range(cfg.num_clients):
        output_filename = cfg.output_filename + "_client_%s" % (k)
        outfile[k] = client_log(cfg.output_dirname, output_filename)

    ## Weight calculation
    total_num_data = 0
    for k in range(cfg.num_clients):
        total_num_data += len(train_data[k])
    weights = {}
    for k in range(cfg.num_clients):
        weights[k] = len(train_data[k]) / total_num_data

    ## Run validation if test data is given or the configuration is enabled
    test_dataloader = None
    if cfg.validation and len(test_data) > 0:
        test_dataloader = DataLoader(
            test_data,
            num_workers=cfg.num_workers,
            batch_size=cfg.test_data_batch_size,
            shuffle=cfg.test_data_shuffle,
        )
    else:
        cfg.validation = False

    server = get_appfl_algorithm(
        algorithm_name=cfg.fed.servername,
        args=(weights, model, loss_fn, cfg.num_clients, cfg.device_server),
        kwargs=cfg.fed.args,
    )

    server.model.to(cfg.device_server)

    batchsize = {}
    for k in range(cfg.num_clients):
        if not cfg.batch_training:
            batchsize[k] = len(train_data[k])
        else:
            batchsize[k] = cfg.train_data_batch_size

    clients = [
        get_appfl_algorithm(
            algorithm_name=cfg.fed.clientname,
            args=(
                k,
                weights[k],
                # deepcopy the common model if there is no personalization, else use the clients' own model
                # the index is k+1, because the first model belongs to the server
                copy.deepcopy(model) if not cfg.personalization else model[k + 1],
                loss_fn,
                DataLoader(
                    train_data[k],
                    num_workers=cfg.num_workers,
                    batch_size=batchsize[k],
                    shuffle=cfg.train_data_shuffle,
                    pin_memory=True,
                ),
                cfg,
                outfile[k],
                test_dataloader,
                metric,
            ),
            kwargs=cfg.fed.args,
        )
        for k in range(cfg.num_clients)
    ]

    start_time = time.time()
    test_loss, test_accuracy, best_accuracy = 0.0, 0.0, 0.0
    for t in range(cfg.num_epochs):
        per_iter_start = time.time()
        local_states = []
        server.model.to("cpu")
        global_state = server.model.state_dict()
        if cfg.personalization:
            keys = [key for key, _ in model[0].named_parameters()]
            for key in keys:
                if key in cfg.p_layers:
                    _ = global_state.pop(key)

        ## Serialized client update
        local_update_start = time.time()
        for k, client in enumerate(clients):
            if cfg.personalization:
                client.model.load_state_dict(global_state, strict=False)
            else:
                client.model.load_state_dict(global_state)
            local_states.append(client.update())
        cfg.logginginfo.LocalUpdate_time = time.time() - local_update_start

        ## Global update
        global_update_start = time.time()
        server.update(local_states)
        cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        ## Global validation
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

    ## Summary
    server.logging_summary(cfg, logger)

    for k, client in enumerate(clients):
        client.outfile.close()

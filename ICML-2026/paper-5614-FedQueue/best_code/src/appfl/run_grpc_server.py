"""
[DEPRECATED] This run script is deprecated and will be removed in the future.
"""

import logging
import torch.nn as nn
from typing import Any
from .misc.data import Dataset
from omegaconf import DictConfig
from .comm.grpc import GRPCCommunicator, APPFLgRPCServer, grpc_serve
from appfl.misc.utils import get_appfl_authenticator


def run_server(
    cfg: DictConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    num_clients: int,
    test_data: Dataset = Dataset(),
    metric: Any = None,
) -> None:
    """Launch gRPC server to listen to the port to serve requests from clients.
    The service URI is set in the configuration.
    The server will not start training until the specified number of clients connect to the server.

    Args:
        cfg (DictConfig): the configuration for this run
        model (nn.Module): neural network model to train
        loss_fn (nn.Module): loss function
        num_clients (int): the number of clients used in PPFL simulation
        test_data (Dataset): optional testing data. If given, validation will run based on this data.
    """

    communicator = GRPCCommunicator(
        cfg.server.id,
        APPFLgRPCServer(cfg, model, loss_fn, test_data, num_clients, metric),
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting the server to listen to requests from clients . . .")

    grpc_serve(
        server_uri=cfg.uri,
        servicer=communicator,
        use_ssl=cfg.use_ssl,
        use_authenticator=cfg.use_authenticator,
        server_certificate_key=cfg.server.server_certificate_key,
        server_certificate=cfg.server.server_certificate,
        authenticator=get_appfl_authenticator(
            authenticator_name=cfg.authenticator,
            authenticator_args=cfg.server.authenticator_kwargs,
        )
        if cfg.use_authenticator
        else None,
        max_message_size=cfg.max_message_size,
    )

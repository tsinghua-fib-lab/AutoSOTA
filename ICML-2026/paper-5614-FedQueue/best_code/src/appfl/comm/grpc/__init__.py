from .serve import serve
from .channel import create_grpc_channel
from .utils import proto_to_databuffer, serialize_model, deserialize_model
from .grpc_client_communicator import GRPCClientCommunicator
from .grpc_server_communicator import GRPCServerCommunicator
from ._credentials import load_credential_from_file
from . import _credentials as _credentials_module
from ..grpc_legacy import (
    APPFLgRPCClient,
    APPFLgRPCServer,
    GRPCCommunicator,
    grpc_serve,
    Job,
)
from .setup_ssl import setup_ssl

__all__ = [
    "serve",
    "create_grpc_channel",
    "proto_to_databuffer",
    "serialize_model",
    "deserialize_model",
    "GRPCClientCommunicator",
    "GRPCServerCommunicator",
    "load_credential_from_file",
    "APPFLgRPCClient",
    "APPFLgRPCServer",
    "GRPCCommunicator",
    "grpc_serve",
    "Job",
    "setup_ssl",
]


def __getattr__(name):
    if name in _credentials_module._REMOVED_CERT_NAMES:
        return getattr(_credentials_module, name)
    raise AttributeError(name)

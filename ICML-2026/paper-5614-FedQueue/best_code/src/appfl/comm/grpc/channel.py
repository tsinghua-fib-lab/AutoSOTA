"""
Auxiliary function to create a secure/insecure gRPC channel.
"""

import grpc
from .auth import APPFLAuthMetadataProvider
from .utils import load_credential_from_file
from typing import Optional, Union, Dict, Any
from appfl.misc.utils import get_appfl_authenticator


def create_grpc_channel(
    server_uri: str,
    *,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    root_certificate: Optional[Union[str, bytes]] = None,
    authenticator: Optional[str] = None,
    authenticator_args: Dict[str, Any] = {},
    max_message_size: int = 2 * 1024 * 1024,
    **kwargs,
) -> grpc.Channel:
    """
    Create a secure/insecure gRPC channel with the given parameters.

    :param server_uri: The URI of the server to connect to.
    :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
    :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
    :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
    :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
    :param authenticator_args: The arguments to pass to the authenticator.
    :param max_message_size: The maximum message size in bytes.
    :return: The created gRPC channel.
    """
    assert not (use_authenticator and not use_ssl), (
        "Authenticator can only be used with SSL/TLS"
    )
    channel_options = [
        ("grpc.max_send_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 20000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
    ]
    if use_ssl:
        if root_certificate is not None:
            if isinstance(root_certificate, str):
                root_certificate = load_credential_from_file(root_certificate)
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificate
            )
        else:
            credentials = grpc.ssl_channel_credentials()
        if use_authenticator:
            assert authenticator is not None, (
                "Authenticator must be provided if use_authenticator is True"
            )
            authenticator = get_appfl_authenticator(authenticator, authenticator_args)
            call_credentials = grpc.metadata_call_credentials(
                APPFLAuthMetadataProvider(authenticator)
            )
            credentials = grpc.composite_channel_credentials(
                credentials, call_credentials
            )
        channel = grpc.secure_channel(server_uri, credentials, options=channel_options)
    else:
        channel = grpc.insecure_channel(server_uri, options=channel_options)
    return channel

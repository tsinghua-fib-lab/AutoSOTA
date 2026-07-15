"""
Auxiliary function to create a secure/insecure gRPC channel.
"""

import grpc
from typing import Optional
from appfl.comm.grpc.auth import APPFLAuthMetadataProvider
from appfl.login_manager import BaseAuthenticator


def create_grpc_channel(
    server_uri: str,
    *,
    use_ssl: bool = False,
    use_authenticator: bool = False,
    root_certificates: Optional[bytes] = None,
    authenticator: Optional[BaseAuthenticator] = None,
    max_message_size: int = 2 * 1024 * 1024,
) -> grpc.Channel:
    """
    Create a secure/insecure gRPC channel with the given parameters.

    :param server_uri: The URI of the server to connect to.
    :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
    :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
    :param root_certificates: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
    :param authenticator: The authenticator to use for authenticating the client in each RPC.
    :param max_message_size: The maximum message size in bytes.
    :return: The created gRPC channel.
    """
    assert not (use_authenticator and not use_ssl), (
        "Authenticator can only be used with SSL/TLS"
    )
    channel_options = [
        ("grpc.max_send_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
    ]
    if use_ssl:
        if root_certificates is not None:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates
            )
        else:
            credentials = grpc.ssl_channel_credentials()
        if use_authenticator:
            assert authenticator is not None, (
                "Authenticator must be provided if use_authenticator is True"
            )
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

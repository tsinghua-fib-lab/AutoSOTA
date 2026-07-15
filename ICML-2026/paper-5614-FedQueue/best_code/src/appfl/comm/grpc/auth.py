import grpc
from appfl.login_manager import BaseAuthenticator


class APPFLAuthMetadataProvider(grpc.AuthMetadataPlugin):
    """
    `APPFLAuthMetadataProvider` is used to attach authentication tokens to the metadata of each RPC.
    """

    def __init__(self, authenticator: BaseAuthenticator):
        self.authenticator = authenticator

    def __call__(self, context, callback):
        """
        Attach tokens from `authenticator.get_auth_token()` to the metadata of the RPC.
        """
        metadata = []
        auth_tokens = self.authenticator.get_auth_token()
        for key, value in auth_tokens.items():
            metadata.append((key, value))
        metadata = tuple(metadata)
        callback(metadata, None)


class APPFLAuthMetadataInterceptor(grpc.ServerInterceptor):
    """
    `APPFLAuthMetadataInterceptor` is used to intercept the metadata of each RPC and validate the authentication tokens.
    """

    def __init__(self, authenticator: BaseAuthenticator):
        self.authenticator = authenticator

        def abort(ignored_request, context):
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid Authentication Tokens."
            )

        self._abortion = grpc.unary_unary_rpc_method_handler(abort)

    def intercept_service(self, continuation, handler_call_details):
        """Validate the authentication tokens in the metadata of the RPC using `authenticator.validate_auth_token()`, and abort the RPC if the tokens are invalid."""
        metadata = dict(handler_call_details.invocation_metadata)
        valid = self.authenticator.validate_auth_token(metadata)
        if not valid:
            return self._abortion
        else:
            return continuation(handler_call_details)

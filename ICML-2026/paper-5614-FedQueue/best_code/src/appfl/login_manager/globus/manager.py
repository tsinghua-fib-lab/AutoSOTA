from __future__ import annotations

import sys
import platform
import threading
from typing import Iterator
from .tokenstore import get_token_storage_adapter
from globus_sdk.scopes import AuthScopes, GroupsScopes
from globus_sdk import (
    NativeAppAuthClient,
    RefreshTokenAuthorizer,
    AuthClient,
    GroupsClient,
)


class GlobusLoginManager:
    """
    The GlobusLoginManager is used to hold a tokenstorage object and combine it with
    - a login flow which authenticates the federated learning server and client users for the correct set of scopes
    - a helper method for ensuring that the user is logged in
    - a helper method to build Globus SDK client objects with correct RefreshTokenAuthorizer
    """

    SCOPES = {
        "appfl_client": {AuthScopes.resource_server: [AuthScopes.openid]},
        "appfl_server": {GroupsScopes.resource_server: [GroupsScopes.all]},
    }

    APPFL_CLIENT_ID = "3de1d1a4-cd37-4d5f-a775-01d9c430330b"

    def __init__(self, *, is_fl_server: bool):
        self._is_fl_server = is_fl_server
        self._token_storage = get_token_storage_adapter(is_fl_server=is_fl_server)
        self._access_lock = threading.Lock()

    def _is_jupyter(self) -> bool:
        return "jupyter_core" in sys.modules

    @property
    def login_requirements(self) -> Iterator[tuple[str, list[str]]]:
        if self._is_fl_server:
            yield from self.SCOPES["appfl_server"].items()
        else:
            yield from self.SCOPES["appfl_client"].items()

    def _get_auth_client(self) -> NativeAppAuthClient:
        return NativeAppAuthClient(
            client_id=self.APPFL_CLIENT_ID,
            app_name="APPFL",
        )

    def _start_auth_flow(self, *, scopes: list[str]) -> None:
        auth_client = self._get_auth_client()
        auth_client.oauth2_start_flow(
            refresh_tokens=True,
            requested_scopes=scopes,
            prefill_named_grant=platform.node(),
        )
        print(
            "Please authenticate with Globus here\n"
            "------------------------------------\n"
            f"{auth_client.oauth2_get_authorize_url(query_params={'prompt': 'login'})}\n"
            "------------------------------------\n"
        )

        auth_code = input(
            "Please enter the authorization code you get after login here: "
        ).strip()
        token_response = auth_client.oauth2_exchange_code_for_tokens(auth_code)
        return token_response

    def run_login_flow(self):
        """
        Run the globus login flow by having the user to manually enter the authorization code.
        This function is only used if the user is not logged in yet, and has to be run in an interactive environment.
        """
        if (not sys.stdin.isatty() or sys.stdin.closed) and not self._is_jupyter():
            raise RuntimeError(
                "Cannot run APPFL login flow in non-interactive environment"
            )
        scopes = [s for _, rs_scopes in self.login_requirements for s in rs_scopes]
        token_response = self._start_auth_flow(scopes=scopes)
        with self._access_lock:
            self._token_storage.store_token_response(token_response)

    def ensure_logged_in(self) -> bool:
        """
        Ensure that the user is logged in by checking if the token storage contains valid tokens.
        Return `True` if the user has logged in, and `False` if the user just logged in.
        """
        with self._access_lock:
            token_data = self._token_storage.get_token_data_by_resource_server()
        for rs, _ in self.login_requirements:
            if rs not in token_data:
                self.run_login_flow()
                return False
        return True

    def logout(self) -> None:
        """
        Return `True` if at least one set of tokens are found and revoked.
        """
        with self._access_lock:
            auth_client = self._get_auth_client()
            tokens_revoked = False
            for (
                rs,
                token_data,
            ) in self._token_storage.get_token_data_by_resource_server().items():
                auth_client.oauth2_revoke_token(token_data.refresh_token)
                auth_client.oauth2_revoke_token(token_data.access_token)
                self._token_storage.remove_token_data(rs)
                tokens_revoked = True
            return tokens_revoked

    def get_identity_client(self) -> AuthClient:
        return AuthClient(
            authorizer=self._get_authorizer(resource_server=AuthScopes.resource_server)
        )

    def get_group_client(self) -> GroupsClient:
        return GroupsClient(
            authorizer=self._get_authorizer(
                resource_server=GroupsScopes.resource_server
            )
        )

    def get_auth_token(self) -> dict:
        """This function is used for FL client to get the auth token for the FL server validation."""
        assert not self._is_fl_server, "Server does not need auth tokens"
        return {
            "access_token": self._token_storage.get_token_data(
                AuthScopes.resource_server
            ).access_token,
            "expires_at": self._token_storage.get_token_data(
                AuthScopes.resource_server
            ).expires_at_seconds,
            "refresh_token": self._token_storage.get_token_data(
                AuthScopes.resource_server
            ).refresh_token,
        }

    def get_identity_client_with_tokens(
        self, access_token=None, refresh_token=None, expires_at=None
    ) -> AuthClient | None:
        """
        Return a client object with the correct authorizer for the Globus identity (auth) server using provided tokens.
        Return `None` for invalid token data. This function is intended to be invoked by FL server for validating
        client information using the tokens provided by the client.
        """
        try:
            authorizer = RefreshTokenAuthorizer(
                refresh_token=refresh_token,
                auth_client=self._get_auth_client(),
                access_token=access_token,
                expires_at=expires_at,
                on_refresh=self._token_storage.store_token_response,
            )
            return AuthClient(authorizer=authorizer)
        except Exception:
            return None

    def _get_authorizer(self, *, resource_server: str) -> RefreshTokenAuthorizer:
        tokens = self._token_storage.get_token_data(resource_server)
        if tokens is None:
            raise LookupError(
                f"Login manager could not find tokens for resource server: {resource_server}"
            )
        with self._access_lock:
            return RefreshTokenAuthorizer(
                tokens.refresh_token,
                self._get_auth_client(),
                access_token=tokens.access_token,
                expires_at=tokens.expires_at_seconds,
                on_refresh=self._token_storage.store_token_response,
            )

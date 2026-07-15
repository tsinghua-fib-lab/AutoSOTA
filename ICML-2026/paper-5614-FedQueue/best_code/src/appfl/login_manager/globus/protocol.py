from __future__ import annotations
import globus_sdk

from typing import Protocol, runtime_checkable


@runtime_checkable
class GlobusLoginManagerProtocol(Protocol):
    def ensure_logged_in(self) -> bool: ...

    def logout(self) -> None: ...

    def get_identity_client(self) -> globus_sdk.AuthClient: ...

    def get_group_client(self) -> globus_sdk.GroupsClient: ...

    def get_auth_token(self) -> dict: ...

    def get_identity_client_with_tokens(
        self, access_token=None, refresh_token=None, expires_at=None
    ) -> globus_sdk.AuthClient | None: ...

from typing import Optional, Dict
from appfl.login_manager import BaseAuthenticator
from appfl.login_manager.globus import GlobusLoginManager


class GlobusAuthenticator(BaseAuthenticator):
    """
    Authenticator for federated learning server and client using Globus Auth.
    :param is_fl_server: Whether the authenticator is for the FL server or client.
    :param globus_group_id: The Globus group ID for the federation. This is only required if `is_fl_server` is `True`.
    """

    def __init__(
        self, *, is_fl_server: bool = False, globus_group_id: Optional[str] = None
    ) -> None:
        self.login_manager = GlobusLoginManager(is_fl_server=is_fl_server)
        self.login_manager.ensure_logged_in()
        self.group_id = globus_group_id
        self.is_fl_server = is_fl_server
        if self.is_fl_server:
            assert self.group_id is not None, (
                "FL server must be associated with a Globus group for authentication"
            )
            group_client = self.login_manager.get_group_client()
            try:
                group_memberships = group_client.get_group(
                    self.group_id, include="memberships"
                )["memberships"]
                self.identities = [m["identity_id"] for m in group_memberships]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get group membership for group {self.group_id}"
                    "Please make sure the group exists and you are the admin or manager of the group."
                ) from e

    def get_auth_token(self) -> Dict[str, str]:
        """
        Invoked by FL client to get the auth tokens as a `dict` for the FL server validation.
        """
        assert not self.is_fl_server, "Server does not need auth tokens"
        auth_tokens = self.login_manager.get_auth_token()
        auth_tokens["expires_at"] = str(auth_tokens["expires_at"])
        return auth_tokens

    def validate_auth_token(self, auth_tokens: Dict) -> bool:
        """
        Invoked by FL server to validate the auth tokens provided by the FL client.
        Return `True` if the tokens are valid, `False` otherwise.
        """
        assert self.is_fl_server, "Client does not need to validate auth tokens"
        identity_client = self.login_manager.get_identity_client_with_tokens(
            access_token=auth_tokens["access_token"],
            refresh_token=auth_tokens["refresh_token"],
            expires_at=int(auth_tokens["expires_at"]),
        )
        if identity_client is None:
            return False
        user_info = identity_client.userinfo()
        for identity in user_info["identity_set"]:
            if identity["sub"] in self.identities:
                return True
        return False

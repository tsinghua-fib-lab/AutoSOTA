import hmac
import warnings
from typing import Dict
from appfl.login_manager import BaseAuthenticator


class NaiveAuthenticator(BaseAuthenticator):
    """
    A naive shared-secret authenticator. The same `auth_token` string must
    be configured on the server and on every client; the server admits any
    peer that presents it.

    This is intended for demos and trusted-network testing only. Production
    deployments should use `GlobusAuthenticator` (or another identity-bound
    mechanism). To mint a fresh token::

        python -c "import secrets; print(secrets.token_urlsafe(32))"
    """

    _MIN_TOKEN_LEN = 16

    def __init__(self, *, auth_token: str):
        if not isinstance(auth_token, str):
            raise TypeError(
                f"auth_token must be a str, got {type(auth_token).__name__}"
            )
        if len(auth_token.strip()) < self._MIN_TOKEN_LEN:
            raise ValueError(
                f"auth_token must be a non-whitespace string of at least "
                f"{self._MIN_TOKEN_LEN} characters. Generate one with "
                f'`python -c "import secrets; print(secrets.token_urlsafe(32))"`.'
            )
        warnings.warn(
            "NaiveAuthenticator is a shared-secret scheme suitable only for "
            "demos and trusted-network testing. Use GlobusAuthenticator (or "
            "another identity-bound authenticator) in production.",
            stacklevel=2,
        )
        self.auth_token = auth_token

    def get_auth_token(self) -> Dict[str, str]:
        return {
            "auth_token": self.auth_token,
        }

    def validate_auth_token(self, token: dict) -> bool:
        provided = token.get("auth_token")
        if not isinstance(provided, str):
            return False
        return hmac.compare_digest(provided, self.auth_token)

import abc
from typing import Dict


class BaseAuthenticator:
    @abc.abstractmethod
    def get_auth_token(self) -> Dict[str, str]:
        """Obtain authentication token(s) in a python `dict` format with key-value pairs of `str` type."""
        pass

    @abc.abstractmethod
    def validate_auth_token(self, token: Dict) -> bool:
        """Validate the authentication token. Return `True` if the token is valid, `False` otherwise."""
        pass

from .utils import load_credential_from_file

__all__ = ["load_credential_from_file"]


_REMOVED_CERT_NAMES = {
    "SERVER_CERTIFICATE",
    "SERVER_CERTIFICATE_KEY",
    "ROOT_CERTIFICATE",
}


def __getattr__(name):
    if name in _REMOVED_CERT_NAMES:
        raise AttributeError(
            f"appfl.comm.grpc._credentials.{name} was removed. Run "
            "`appfl-setup-ssl` and pass the generated cert paths via "
            "load_credential_from_file()."
        )
    raise AttributeError(name)

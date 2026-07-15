from __future__ import annotations

import os
import pathlib
import stat

from appfl.misc.utils import _ensure_secure_dir

try:
    from globus_sdk.token_storage import SQLiteTokenStorage  # globus-sdk v4 and above
except ImportError:
    from globus_sdk.tokenstorage import SQLiteTokenStorage  # globus-sdk v3


def _home() -> pathlib.Path:
    return pathlib.Path.home()


def ensure_appfl_dir() -> pathlib.Path:
    user_dirname = os.getenv("APPFL_USER_DIR")
    if user_dirname:
        dirname = pathlib.Path(user_dirname)
        _ensure_secure_dir(dirname)
    else:
        appfl_root = _home() / ".appfl"
        _ensure_secure_dir(appfl_root)
        dirname = appfl_root / "globus_auth"
        _ensure_secure_dir(dirname)
    if dirname.is_file():
        raise FileExistsError(
            f"Error creating directory {dirname}, "
            "please rename or remove the conflicting file."
        )
    return dirname


def _get_storage_filename() -> str:
    dirname = ensure_appfl_dir()
    filename = os.path.join(dirname, "storage.db")
    if os.path.exists(filename):
        st = os.lstat(filename)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"{filename} is not a regular file (or is a symlink)")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise PermissionError(
                f"{filename} is owned by uid {st.st_uid}, expected {os.getuid()}"
            )
        mode_bits = stat.S_IMODE(st.st_mode)
        if mode_bits & (stat.S_IRWXG | stat.S_IRWXO):
            os.chmod(filename, 0o600)
    return filename


def _resolve_namespace(is_fl_server: bool) -> str:
    """
    Return the namespace for saving tokens:
    `appfl_server` if invoked by an FL server, and `appfl_client` if invoked by an FL client.

    :param `is_fl_server`: True if invoked by an FL server, False if invoked by an FL client.
    """
    if is_fl_server:
        return "appfl_server"
    else:
        return "appfl_client"


def get_token_storage_adapter(*, is_fl_server: bool) -> SQLiteTokenStorage:
    """
    Return the SQLite token storage adapter.

    :param `is_fl_server`: True if invoked by an FL server, False if invoked by an FL client.
    """
    filename = _get_storage_filename()
    namespace = _resolve_namespace(is_fl_server)
    return SQLiteTokenStorage(
        filename,
        namespace=namespace,
        connect_params={"check_same_thread": False},
    )

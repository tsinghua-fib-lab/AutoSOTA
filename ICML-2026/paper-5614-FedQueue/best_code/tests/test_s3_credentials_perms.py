"""Tests for the credentials-file permission check in
appfl.comm.utils.s3_storage._open_credentials_file_securely."""

import csv
import os

import pytest

from appfl.comm.utils.s3_storage import _open_credentials_file_securely


posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission semantics required"
)


def _write_creds(path, mode=0o600):
    path.write_text("us-east-1,AKIA_FAKE,SECRET_FAKE\n")
    path.chmod(mode)
    return path


@posix_only
def test_accepts_owner_only_file(tmp_path):
    creds = _write_creds(tmp_path / "creds.csv", mode=0o600)
    with _open_credentials_file_securely(str(creds)) as f:
        row = next(csv.reader(f))
    assert row == ["us-east-1", "AKIA_FAKE", "SECRET_FAKE"]


@posix_only
@pytest.mark.parametrize(
    "bad_mode",
    [
        0o644,  # group + world readable (AWS console default)
        0o640,  # group readable
        0o604,  # world readable
        0o620,  # group writable
        0o602,  # world writable
        0o755,
    ],
)
def test_rejects_group_or_world_accessible(tmp_path, bad_mode):
    creds = _write_creds(tmp_path / "creds.csv", mode=bad_mode)
    with pytest.raises(PermissionError, match="insecure permissions"):
        _open_credentials_file_securely(str(creds))


@posix_only
def test_rejects_symlink_to_credentials(tmp_path):
    """O_NOFOLLOW must reject a symlink at the leaf even if the symlink
    target is securely permissioned."""
    real = _write_creds(tmp_path / "real_creds.csv", mode=0o600)
    link = tmp_path / "link_creds.csv"
    os.symlink(str(real), str(link))
    with pytest.raises(OSError):
        _open_credentials_file_securely(str(link))


@posix_only
def test_rejects_file_owned_by_other_user(tmp_path, monkeypatch):
    creds = _write_creds(tmp_path / "creds.csv", mode=0o600)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 12345)
    with pytest.raises(PermissionError, match="not owned by the current user"):
        _open_credentials_file_securely(str(creds))


@posix_only
def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        _open_credentials_file_securely(str(tmp_path / "does_not_exist.csv"))


def test_windows_path_warns_and_opens(tmp_path, monkeypatch):
    """On Windows the perm check is skipped with a warning."""
    creds = tmp_path / "creds.csv"
    creds.write_text("us-east-1,AKIA_FAKE,SECRET_FAKE\n")
    monkeypatch.setattr(os, "name", "nt")
    with pytest.warns(UserWarning, match="Cannot enforce"):
        with _open_credentials_file_securely(str(creds)) as f:
            assert f.read().startswith("us-east-1")


@posix_only
def test_cloudstorage_init_propagates_permission_error(tmp_path):
    """End-to-end: CloudStorage.init refuses to start when the creds file
    is group-readable."""
    from appfl.comm.utils import s3_storage

    # Reset the singleton so init() actually runs.
    s3_storage.CloudStorage.instc = None

    creds = _write_creds(tmp_path / "creds.csv", mode=0o644)

    with pytest.raises(PermissionError):
        s3_storage.CloudStorage.init(
            s3_bucket="dummy",
            s3_creds_file=str(creds),
            s3_tmp_dir=str(tmp_path / "s3tmp"),
        )

    # Singleton must not have been populated by the failed init.
    assert s3_storage.CloudStorage.instc is None


def test_legacy_unsafe_open_not_in_credentials_block():
    """Defense in depth: the credentials-loading block must route through
    the secure helper, not plain open()."""
    import inspect
    from appfl.comm.utils import s3_storage

    src = inspect.getsource(s3_storage.CloudStorage.init)
    assert "_open_credentials_file_securely" in src
    assert "open(s3_creds_file)" not in src

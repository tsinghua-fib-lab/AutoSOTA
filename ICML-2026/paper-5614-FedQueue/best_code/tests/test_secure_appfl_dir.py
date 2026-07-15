"""Tests for appfl.misc.utils.secure_appfl_dir / _ensure_secure_dir, the
helper that creates APPFL working directories with restrictive permissions
(closes the /tmp/.appfl shared-host info-leak / RCE primitive)."""

import os
import pathlib
import stat

import pytest

from appfl.misc.utils import _ensure_secure_dir, secure_appfl_dir


posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission semantics required"
)


# ---------- secure_appfl_dir ----------


@posix_only
def test_secure_appfl_dir_uses_home_when_writable(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

    result = secure_appfl_dir("comm", "client_a", "exp_1")
    expected = fake_home / ".appfl" / "comm" / "client_a" / "exp_1"
    assert pathlib.Path(result) == expected
    assert expected.is_dir()

    # Every component below `.appfl` must be 0o700.
    for d in [expected, *expected.parents]:
        if d == fake_home:
            break
        mode = stat.S_IMODE(d.stat().st_mode)
        assert mode == 0o700, f"{d} has mode {oct(mode)}"


@posix_only
def test_secure_appfl_dir_falls_back_when_home_unwritable(tmp_path, monkeypatch):
    """When $HOME is read-only the helper falls back to
    /tmp/.appfl-<uid>/... rather than /tmp/.appfl/..."""
    ro_home = tmp_path / "ro_home"
    ro_home.mkdir(mode=0o500)  # r-x only — mkdir of children will fail
    try:
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: ro_home))

        result = secure_appfl_dir("globus_compute", "exp_2")
        uid = os.getuid()
        expected = pathlib.Path(f"/tmp/.appfl-{uid}/globus_compute/exp_2")
        assert pathlib.Path(result) == expected
        assert expected.is_dir()
        # Cleanup so subsequent runs start fresh.
        for parent in [expected, expected.parent, expected.parent.parent]:
            try:
                parent.rmdir()
            except OSError:
                pass
    finally:
        ro_home.chmod(0o700)


@posix_only
def test_secure_appfl_dir_fallback_namespaced_by_uid(monkeypatch, tmp_path):
    """The fallback directory name must include the current uid so a
    co-tenant cannot pre-create /tmp/.appfl and trap our writes."""
    ro_home = tmp_path / "ro_home"
    ro_home.mkdir(mode=0o500)
    try:
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: ro_home))
        path = secure_appfl_dir()
        assert f"/.appfl-{os.getuid()}" in path
        assert path != "/tmp/.appfl"
    finally:
        ro_home.chmod(0o700)


# ---------- _ensure_secure_dir ----------


@posix_only
def test_ensure_secure_dir_creates_with_0o700_under_strict_umask(tmp_path):
    """umask masks off bits at mkdir time; the helper must chmod after
    creating to enforce exactly 0o700."""
    target = tmp_path / "appfl_root"
    old_umask = os.umask(0o077)
    try:
        _ensure_secure_dir(target)
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o700


@posix_only
def test_ensure_secure_dir_repairs_loose_permissions(tmp_path):
    """If the directory already exists with permissive bits, the helper
    chmods it back to 0o700 (instead of refusing — the chmod happens
    before the verify)."""
    target = tmp_path / "loose"
    target.mkdir(mode=0o755)
    _ensure_secure_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@posix_only
def test_ensure_secure_dir_refuses_symlink(tmp_path):
    """A symlink at the path must be refused even if the target is a
    proper 0o700 directory."""
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    os.symlink(str(real), str(link))
    with pytest.raises(PermissionError, match="not a directory"):
        _ensure_secure_dir(link)


@posix_only
def test_ensure_secure_dir_refuses_wrong_owner(tmp_path, monkeypatch):
    target = tmp_path / "appfl"
    target.mkdir(mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 99999)
    with pytest.raises(PermissionError, match="owned by uid"):
        _ensure_secure_dir(target)


# ---------- legacy bug regression ----------


def test_legacy_unsafe_mkdir_not_in_misc_utils():
    """The duplicated `pathlib.Path(...).mkdir(parents=True, exist_ok=True)`
    blocks for `.appfl` should be gone from misc.utils — they're replaced by
    secure_appfl_dir."""
    import inspect
    from appfl.misc import utils

    src = inspect.getsource(utils.create_instance_from_file_source)
    assert "secure_appfl_dir" in src
    assert "/tmp/.appfl" not in src


def test_legacy_unsafe_mkdir_not_in_s3_utils():
    import inspect
    from appfl.comm.utils import s3_utils

    src = inspect.getsource(s3_utils)
    assert "/tmp/.appfl" not in src
    assert "mkdir(parents=True, exist_ok=True)" not in src


def test_legacy_unsafe_mkdir_not_in_s3_storage_init():
    import inspect
    from appfl.comm.utils import s3_storage

    src = inspect.getsource(s3_storage.CloudStorage.init)
    assert "secure_appfl_dir" in src
    assert "/tmp/.appfl/s3_tmp_dir" not in src

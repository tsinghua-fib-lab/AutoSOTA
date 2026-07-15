"""Tests for src/appfl/comm/grpc/setup_ssl.py covering the fixes for
security-review finding #6 (shell injection via os.system on a user-supplied
path)."""

import os
import stat
import subprocess

import pytest

import sys

import appfl.comm.grpc.setup_ssl  # noqa: F401  (ensures submodule is loaded)
from appfl.comm.grpc.setup_ssl import setup_ssl

# The grpc package's __init__.py re-exports the `setup_ssl` function under
# the same name as the submodule, shadowing it on the package object. Reach
# the actual module via sys.modules so monkeypatching attributes works.
setup_ssl_module = sys.modules["appfl.comm.grpc.setup_ssl"]


def _input_feeder(inputs):
    """Return a callable suitable for monkeypatching builtins.input that
    yields the given strings in order."""
    it = iter(inputs)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(
                f"setup_ssl asked for more input than the test provided. Last prompt: {prompt!r}"
            )

    return _input


@pytest.fixture
def stub_openssl(monkeypatch):
    """Replace subprocess.run so the test never actually invokes openssl."""
    calls = []

    def _run(argv, check=False, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(setup_ssl_module.subprocess, "run", _run)
    return calls


def test_rejects_shell_metachars_in_ssl_dir(
    monkeypatch, tmp_path, stub_openssl, capsys
):
    """A ssl_dir with shell metacharacters must be rejected and the prompt
    must re-ask, eventually accepting a clean path."""
    pwned_marker = tmp_path / "pwned"
    good_dir = tmp_path / "ssl"

    malicious = f"/tmp/x$(touch {pwned_marker})"

    monkeypatch.setattr(
        "builtins.input",
        _input_feeder(
            [
                malicious,
                str(good_dir),
                "",  # C
                "",  # ST
                "",  # ORG
                "",  # DNS
                "",  # IP
            ]
        ),
    )

    setup_ssl()

    assert not pwned_marker.exists(), (
        "Shell command substitution executed — the metachar guard is broken"
    )
    assert good_dir.is_dir()
    assert (good_dir / "generate_ssl.sh").is_file()

    captured = capsys.readouterr()
    assert "Invalid directory" in captured.out


@pytest.mark.parametrize(
    "bad",
    [
        "/tmp/a;rm -rf /",
        "/tmp/a|whoami",
        "/tmp/a&echo x",
        "/tmp/a`id`",
        "/tmp/a$(id)",
        "/tmp/a\nid",
        "/tmp/a*",
        "/tmp/a?",
        "/tmp/a b",
        "/tmp/a>b",
    ],
)
def test_metachar_classes_each_rejected(
    monkeypatch, tmp_path, stub_openssl, capsys, bad
):
    """Each shell-metacharacter class must be rejected by the whitelist."""
    good_dir = tmp_path / "ssl"

    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([bad, str(good_dir), "", "", "", "", ""]),
    )
    setup_ssl()
    assert good_dir.is_dir()
    assert "Invalid directory" in capsys.readouterr().out


def test_chmod_uses_os_chmod_not_shell(monkeypatch, tmp_path, stub_openssl):
    """The fix replaced os.system with os.chmod. Patch os.system to fail
    loudly so any regression is caught."""

    def _boom(cmd):
        raise AssertionError(
            f"setup_ssl invoked the shell via os.system({cmd!r}) — regression of issue #6"
        )

    monkeypatch.setattr(setup_ssl_module.os, "system", _boom)

    good_dir = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(good_dir), "", "", "", "", ""]),
    )
    setup_ssl()
    assert (good_dir / "generate_ssl.sh").is_file()


def test_generated_script_mode_is_0700(monkeypatch, tmp_path, stub_openssl):
    """The generated bash script must be owner-rwx only, regardless of umask."""
    good_dir = tmp_path / "ssl"
    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(good_dir), "", "", "", "", ""]),
    )

    old_umask = os.umask(0o022)
    try:
        setup_ssl()
    finally:
        os.umask(old_umask)

    script = good_dir / "generate_ssl.sh"
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode == 0o700, f"expected 0o700, got {oct(mode)}"


def test_surfaces_oserror(monkeypatch, tmp_path, stub_openssl, capsys):
    """An unwritable ssl_dir must surface the underlying OSError message
    (no more silent bare-except)."""
    unwritable = tmp_path / "readonly"
    unwritable.mkdir()
    unwritable.chmod(0o500)  # r-x for owner, no write

    target = unwritable / "ssl"  # mkdir will fail with PermissionError
    good = tmp_path / "good"

    monkeypatch.setattr(
        "builtins.input",
        _input_feeder([str(target), str(good), "", "", "", "", ""]),
    )

    try:
        setup_ssl()
    finally:
        unwritable.chmod(0o700)

    out = capsys.readouterr().out
    assert "Invalid directory" in out
    # The new code includes the underlying exception in the message
    assert "Permission denied" in out or "permission" in out.lower()
    assert good.is_dir()

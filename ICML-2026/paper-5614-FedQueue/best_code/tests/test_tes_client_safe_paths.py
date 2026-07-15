"""Tests for the symlink-race / predictable-tempfile fixes in
appfl.comm.tes.tes_client_communicator."""

import argparse
import json
import os
import pickle
import stat

import pytest

from appfl.comm.tes.tes_client_communicator import (
    TESClientCommunicator,
    _mkstemp_path,
    _safe_open_for_write,
    tes_client_entry_point,
)


# ---------- _mkstemp_path ----------


def test_mkstemp_path_creates_unique_owner_only_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    p1 = _mkstemp_path(".pkl")
    p2 = _mkstemp_path(".pkl")
    assert p1 != p2
    assert os.path.isfile(p1) and os.path.isfile(p2)
    mode = stat.S_IMODE(os.stat(p1).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    assert p1.endswith(".pkl")


def test_mkstemp_path_uses_appfl_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    p = _mkstemp_path(".json")
    assert os.path.basename(p).startswith("appfl-tes-")


# ---------- _safe_open_for_write ----------


def test_safe_open_writes_new_file(tmp_path):
    target = tmp_path / "out.bin"
    with _safe_open_for_write(str(target), binary=True) as f:
        f.write(b"hello")
    assert target.read_bytes() == b"hello"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_safe_open_truncates_regular_file(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old contents that should be erased")
    with _safe_open_for_write(str(target), binary=False) as f:
        f.write("new")
    assert target.read_text() == "new"


def test_safe_open_refuses_symlink(tmp_path):
    """A pre-existing symlink at the target path must not be followed —
    this is the core symlink-race fix."""
    target = tmp_path / "evil.pkl"
    sensitive = tmp_path / "sensitive_file"
    sensitive.write_text("operator's data")
    os.symlink(str(sensitive), str(target))

    with pytest.raises(OSError):
        _safe_open_for_write(str(target), binary=True)

    # The symlink target was not touched.
    assert sensitive.read_text() == "operator's data"


# ---------- save_model_to_path / save_logs_to_path ----------


@pytest.fixture
def client():
    return TESClientCommunicator(client_agent_config=None)


def test_save_model_refuses_symlink(tmp_path, client):
    target = tmp_path / "model.pkl"
    sensitive = tmp_path / "authorized_keys"
    sensitive.write_text("ssh-ed25519 AAAA...")
    os.symlink(str(sensitive), str(target))

    with pytest.raises(RuntimeError, match="Failed to save model"):
        client.save_model_to_path({"weights": [1, 2, 3]}, str(target))

    assert sensitive.read_text() == "ssh-ed25519 AAAA..."


def test_save_logs_refuses_symlink(tmp_path, client):
    target = tmp_path / "logs.json"
    sensitive = tmp_path / "shadow"
    sensitive.write_text("root:!:...")
    os.symlink(str(sensitive), str(target))

    with pytest.raises(RuntimeError, match="Failed to save logs"):
        client.save_logs_to_path({"loss": 0.1}, str(target))

    assert sensitive.read_text() == "root:!:..."


def test_save_model_round_trip(tmp_path, client):
    target = tmp_path / "model.pkl"
    payload = {"weights": [1.0, 2.0]}
    client.save_model_to_path(payload, str(target))
    with open(target, "rb") as f:
        assert pickle.load(f) == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_save_logs_round_trip(tmp_path, client):
    target = tmp_path / "logs.json"
    payload = {"loss": 0.42, "epoch": 3}
    client.save_logs_to_path(payload, str(target))
    assert json.loads(target.read_text()) == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


# ---------- defaults ----------


def test_execute_task_signature_has_no_predictable_defaults():
    """Regression guard: the predictable /tmp/output_model.pkl and
    /tmp/training_logs.json defaults must not come back."""
    import inspect

    sig = inspect.signature(TESClientCommunicator.execute_task)
    assert sig.parameters["output_path"].default is None
    assert sig.parameters["logs_path"].default is None


def test_entry_point_argparse_defaults_are_none(monkeypatch):
    """The CLI defaults for --output-path / --logs-path must not be the
    well-known /tmp/... paths."""
    captured = {}

    def _fake_parse(self, args=None, namespace=None):
        # Snapshot the registered defaults instead of actually running.
        for action in self._actions:
            if action.dest in ("output_path", "logs_path"):
                captured[action.dest] = action.default
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse)
    with pytest.raises(SystemExit):
        tes_client_entry_point()

    assert captured == {"output_path": None, "logs_path": None}


def test_predictable_paths_not_in_source():
    """Defense in depth: ensure the legacy fixed paths are gone from
    the implementation file."""
    import inspect
    from appfl.comm.tes import tes_client_communicator

    src = inspect.getsource(tes_client_communicator)
    assert "/tmp/output_model.pkl" not in src
    assert "/tmp/training_logs.json" not in src

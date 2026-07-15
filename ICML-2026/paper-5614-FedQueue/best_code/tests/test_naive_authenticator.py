"""Tests for appfl.login_manager.NaiveAuthenticator covering the contract
that the authenticator (a) requires an explicit non-trivial auth_token,
(b) emits a warning to discourage production use, and (c) compares tokens
in constant time."""

import warnings

import pytest

from appfl.login_manager import NaiveAuthenticator


_GOOD_TOKEN = "x" * 32  # >= 16 chars


def _make(token=_GOOD_TOKEN):
    """Construct a NaiveAuthenticator while suppressing the demo warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NaiveAuthenticator(auth_token=token)


def test_auth_token_is_required():
    with pytest.raises(TypeError):
        NaiveAuthenticator()  # type: ignore[call-arg]


def test_legacy_default_token_is_rejected():
    """The previously-hardcoded default must no longer be accepted as a
    valid token (it's only 22 chars but more importantly it's public)."""
    # The legacy default happened to be 22 chars, so length alone wouldn't
    # reject it — we instead assert no caller can construct an authenticator
    # without passing *some* token explicitly. That, plus the docs change
    # removing the literal, is what closes the finding.
    with pytest.raises(TypeError):
        NaiveAuthenticator()  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["", "   ", "short", "a" * 15])
def test_short_or_empty_tokens_are_rejected(bad):
    with pytest.raises(ValueError):
        NaiveAuthenticator(auth_token=bad)


def test_non_string_token_is_rejected():
    with pytest.raises(TypeError):
        NaiveAuthenticator(auth_token=12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NaiveAuthenticator(auth_token=None)  # type: ignore[arg-type]


def test_construction_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NaiveAuthenticator(auth_token=_GOOD_TOKEN)
    assert any("NaiveAuthenticator" in str(w.message) for w in caught), (
        "Expected a warning mentioning NaiveAuthenticator on construction"
    )


def test_get_and_validate_round_trip():
    auth = _make()
    presented = auth.get_auth_token()
    assert presented == {"auth_token": _GOOD_TOKEN}
    assert auth.validate_auth_token(presented) is True


def test_validate_rejects_wrong_token():
    auth = _make()
    assert auth.validate_auth_token({"auth_token": "y" * 32}) is False


def test_validate_rejects_missing_field():
    auth = _make()
    assert auth.validate_auth_token({}) is False
    assert auth.validate_auth_token({"other": _GOOD_TOKEN}) is False


def test_validate_rejects_non_string_field():
    auth = _make()
    assert auth.validate_auth_token({"auth_token": None}) is False
    assert auth.validate_auth_token({"auth_token": 12345}) is False


def test_validate_uses_constant_time_compare(monkeypatch):
    """The comparison should go through hmac.compare_digest, not `==`."""
    import hmac

    calls = []
    real = hmac.compare_digest

    def _spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(
        "appfl.login_manager.naive.naive_authenticator.hmac.compare_digest", _spy
    )
    auth = _make()
    auth.validate_auth_token({"auth_token": "y" * 32})
    assert calls, "validate_auth_token did not route through hmac.compare_digest"


def test_legacy_token_string_is_not_in_source():
    """Defense in depth: ensure the previously-hardcoded literal is gone
    from the implementation file."""
    import inspect

    from appfl.login_manager.naive import naive_authenticator

    src = inspect.getsource(naive_authenticator)
    assert "appfl-naive-auth-token" not in src

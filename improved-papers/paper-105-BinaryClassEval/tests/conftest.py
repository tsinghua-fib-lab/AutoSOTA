"""Configuration for pytest."""
import pytest


@pytest.fixture(autouse=True)
def mpl_no_show(monkeypatch):
    """Prevent matplotlib from showing plots during tests."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)

"""Shared pytest configuration.

Custom markers let CI skip tests that need a GPU or the heavy experiment
dependencies (ehtim / pMF / checkpoints) that aren't present on the cloud runner:

    pytest -m "not gpu and not blackhole and not superres"
"""

import pytest

MARKERS = {
    "gpu": "requires a CUDA device",
    "blackhole": "requires the black-hole (InverseBench / ehtim) stack",
    "superres": "requires the super-resolution (pMF) stack",
    "slow": "slow test (large sample counts / many steps)",
}


def pytest_configure(config):
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed torch before every test for reproducibility."""
    import torch

    torch.manual_seed(0)
    yield

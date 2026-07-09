import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cpc_cfg_init():
    """Config with constrain_against='init' for 2-model case."""
    return OmegaConf.create({"conformal_policy_control": {"constrain_against": "init"}})


@pytest.fixture
def cpc_cfg_sequential():
    """Config with constrain_against='sequential' for multi-model case."""
    return OmegaConf.create(
        {"conformal_policy_control": {"constrain_against": "sequential"}}
    )


@pytest.fixture
def rng():
    """Reproducible numpy random generator."""
    return np.random.default_rng(42)

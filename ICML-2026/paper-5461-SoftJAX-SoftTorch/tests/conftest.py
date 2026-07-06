from __future__ import annotations

import pytest

from . import common


@pytest.fixture(scope="session")
def make_input():
    return common.make_array


@pytest.fixture(scope="session")
def make_pair():
    return common.pair_arrays

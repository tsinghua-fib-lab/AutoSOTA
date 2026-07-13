"""Round-trip checks for the pickle-free results format."""

from __future__ import annotations

import math

import numpy as np

from dropout_mft.results import (
    load_npz_result,
    parse_tuple_key,
    save_npz_result,
)


def test_save_load_roundtrip_nested(tmp_path):
    data = {
        "results": {
            "constant": {
                "test_acc": np.arange(12, dtype=float).reshape(3, 4),
                "best": [1.0, 2.0, 3.0],
            }
        },
        "theory": {"constant": {"xi_eff": 4.0}},
        "config": {"H_BAR_VALUES": [0.05, 0.1], "pair": (1, "constant")},
    }

    path = tmp_path / "run.npz"
    save_npz_result(path, data)
    loaded = load_npz_result(path)

    np.testing.assert_array_equal(
        loaded["results"]["constant"]["test_acc"],
        data["results"]["constant"]["test_acc"],
    )
    assert loaded["theory"]["constant"]["xi_eff"] == 4.0
    assert loaded["config"]["H_BAR_VALUES"] == [0.05, 0.1]
    assert loaded["config"]["pair"] == (1, "constant")


def test_save_load_preserves_non_finite_floats(tmp_path):
    data = {"nan": math.nan, "inf": math.inf, "ninf": -math.inf, "x": 1.5}

    path = tmp_path / "floats.npz"
    save_npz_result(path, data)
    loaded = load_npz_result(path)

    assert math.isnan(loaded["nan"])
    assert loaded["inf"] == math.inf
    assert loaded["ninf"] == -math.inf
    assert loaded["x"] == 1.5


def test_load_without_pickle(tmp_path):
    # The whole point of the format: it loads with allow_pickle=False.
    path = tmp_path / "safe.npz"
    save_npz_result(path, {"arr": np.ones(3)})
    with np.load(path, allow_pickle=False) as bundle:
        assert "__metadata__" in bundle.files


def test_parse_tuple_key():
    assert parse_tuple_key("(0.1, 'constant')") == (0.1, "constant")

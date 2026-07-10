from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.functional import list_functional_methods, merge_functional, merge_raw_matrices
from merge_and_rebase.merge.registry import get_method

ALL_FUNCTIONAL_METHODS = [
    "actmerge",
    "actmat",
    "task_arithmetic",
    "weighted_average",
    "wudi",
    "tsv_merge",
    "isoc_merge",
    "isocts_merge",
    "dc_merge",
    "dare_merge",
    "ties_merge",
    "pcb",
    "pcb_merge",
    "wudi_merge",
    "cart_merge",
]


def _toy_matrices_2d() -> list[torch.Tensor]:
    return [
        torch.tensor(
            [
                [1.0, -0.5, 0.2],
                [-0.3, 0.8, -0.7],
                [0.4, -0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.3, 0.6, -0.2],
                [0.9, -0.4, 0.5],
                [-0.8, 0.2, 0.7],
            ],
            dtype=torch.float32,
        ),
    ]


def _toy_matrices_1d() -> list[torch.Tensor]:
    return [
        torch.tensor([0.2, -0.4, 0.1, 0.5], dtype=torch.float32),
        torch.tensor([-0.3, 0.5, -0.2, 0.1], dtype=torch.float32),
    ]


def test_list_functional_methods_contains_all() -> None:
    methods = set(list_functional_methods())
    assert methods.issuperset(set(ALL_FUNCTIONAL_METHODS))


@pytest.mark.parametrize("method_name", ALL_FUNCTIONAL_METHODS)
def test_merge_functional_accepts_raw_matrices(method_name: str) -> None:
    mats = _toy_matrices_2d()

    merged = merge_functional(
        method_name,
        matrices=mats,
        svd_dtype="fp32",
        accum_dtype="fp32",
        topk=1.0,
        merging_type="mean",
        drop_rate=0.0,
        common_space_fraction=0.0,
        pruning_rank=1,
        scaling_coeffs=0.5,
        clamp_min_ratio=0.0,
        clamp_max_ratio=0.0,
        att_ratio=1.0,
        lam=1.0,
    )

    assert merged.shape == mats[0].shape
    assert torch.isfinite(merged).all()


def test_merge_raw_matrices_alias() -> None:
    mats = _toy_matrices_2d()
    merged = merge_raw_matrices("task_arithmetic", matrices=mats)
    assert merged.shape == mats[0].shape
    assert torch.isfinite(merged).all()


def test_merge_functional_unknown_method_raises() -> None:
    mats = _toy_matrices_2d()
    with pytest.raises(KeyError):
        merge_functional("not_a_method", matrices=mats)


def test_vector_methods_accept_1d() -> None:
    mats = _toy_matrices_1d()
    for name in ["task_arithmetic", "weighted_average", "wudi", "wudi_merge", "dare_merge", "ties_merge", "pcb", "pcb_merge"]:
        merged = merge_functional(
            name,
            matrices=mats,
            topk=1.0,
            drop_rate=0.0,
            clamp_min_ratio=0.0,
            clamp_max_ratio=0.0,
            att_ratio=1.0,
        )
        assert merged.shape == mats[0].shape

def test_matrix_only_methods_reject_1d() -> None:
    mats = _toy_matrices_1d()
    for name in ["cart_merge"]:
        with pytest.raises(ValueError, match="requires 2D matrices"):
            merge_functional(name, matrices=mats)


def test_svd_methods_zero_1d_by_default() -> None:
    mats = _toy_matrices_1d()
    for name in ["tsv_merge", "isoc_merge", "isocts_merge", "dc_merge"]:
        merged = merge_functional(name, matrices=mats)
        assert torch.equal(merged, torch.zeros_like(mats[0]))


def test_svd_methods_can_average_1d() -> None:
    mats = _toy_matrices_1d()
    expected = torch.stack(mats).mean(dim=0)
    for name in ["tsv_merge", "isoc_merge", "isocts_merge", "dc_merge"]:
        merged = merge_functional(name, matrices=mats, method_params={"vector_1d_merge": "average"})
        assert torch.allclose(merged, expected)


def test_cart_still_rejects_1d() -> None:
    mats = _toy_matrices_1d()
    with pytest.raises(ValueError, match="requires 2D matrices"):
        merge_functional("cart_merge", matrices=mats)


@pytest.mark.parametrize("topk", [0.4, 1.0])
@pytest.mark.parametrize("merging_type", ["mean", "sum", "max"])
def test_ties_low_memory_matches_dense(topk: float, merging_type: str) -> None:
    base = {
        "linear.weight": torch.tensor([[1.0, -1.0, 0.5], [0.2, -0.3, 0.4]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
    }
    tuned = [
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.3, -0.7, 0.0], [0.5, 0.1, -0.2]]),
            "linear.bias": base["linear.bias"] + torch.tensor([0.2, -0.4, 0.1]),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[-0.2, -0.8, 0.6], [0.1, -0.2, -0.5]]),
            "linear.bias": base["linear.bias"] + torch.tensor([-0.3, -0.1, 0.5]),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.7, 0.2, -0.4], [-0.6, 0.3, -0.1]]),
            "linear.bias": base["linear.bias"] + torch.tensor([0.4, -0.6, -0.2]),
        },
    ]
    method = get_method("ties_merge")

    dense = method.prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"topk": topk, "merging_type": merging_type},
    )
    low_memory = method.prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"topk": topk, "merging_type": merging_type, "low_memory": True},
    )

    assert dense[1].keys() == low_memory[1].keys()
    for key in dense[1]:
        assert torch.allclose(dense[1][key], low_memory[1][key])


@pytest.mark.parametrize("drop_rate", [0.0, 0.4])
@pytest.mark.parametrize("rescale", [False, True])
def test_dare_low_memory_matches_dense(drop_rate: float, rescale: bool) -> None:
    base = {
        "linear.weight": torch.tensor([[1.0, -1.0, 0.5], [0.2, -0.3, 0.4]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
    }
    tuned = [
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.3, -0.7, 0.0], [0.5, 0.1, -0.2]]),
            "linear.bias": base["linear.bias"] + torch.tensor([0.2, -0.4, 0.1]),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[-0.2, -0.8, 0.6], [0.1, -0.2, -0.5]]),
            "linear.bias": base["linear.bias"] + torch.tensor([-0.3, -0.1, 0.5]),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.7, 0.2, -0.4], [-0.6, 0.3, -0.1]]),
            "linear.bias": base["linear.bias"] + torch.tensor([0.4, -0.6, -0.2]),
        },
    ]
    method = get_method("dare_merge")
    params = {"drop_rate": drop_rate, "rescale": rescale, "seed": 123}

    dense = method.prepare(base=base, tuned=tuned, strict=True, weights=[1.0, 0.5, -0.25], method_params=params)
    low_memory = method.prepare(
        base=base,
        tuned=tuned,
        strict=True,
        weights=[1.0, 0.5, -0.25],
        method_params={**params, "low_memory": True},
    )

    assert dense[1].keys() == low_memory[1].keys()
    for key in dense[1]:
        assert torch.allclose(dense[1][key], low_memory[1][key])


def test_dare_merge_forwards_method_params_to_prepare() -> None:
    base = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    tuned = [{"weight": torch.tensor([3.0, 5.0], dtype=torch.float32)}]

    method = get_method("dare_merge")
    merged = method.merge(
        base=base,
        tuned=tuned,
        alpha=0.5,
        strict=True,
        method_params={"drop_rate": 0.0, "low_memory": True},
    )

    assert torch.allclose(merged["weight"], torch.tensor([2.0, 3.5]))


def test_state_dict_svd_methods_can_average_1d_deltas() -> None:
    base = {
        "linear.weight": torch.eye(2),
        "linear.bias": torch.tensor([1.0, -1.0]),
    }
    tuned = [
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "linear.bias": torch.tensor([2.0, 1.0]),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.4, 0.3], [0.2, 0.1]]),
            "linear.bias": torch.tensor([4.0, -3.0]),
        },
    ]
    expected_bias_delta = torch.tensor([2.0, 0.0])

    for name in ["tsv_merge", "isoc_merge", "isocts_merge"]:
        method = get_method(name)
        zero_prepared = method.prepare(base=base, tuned=tuned, strict=True)
        assert torch.equal(zero_prepared[1].get("linear.bias", torch.zeros(2)), torch.zeros(2))

        avg_prepared = method.prepare(
            base=base,
            tuned=tuned,
            strict=True,
            method_params={"vector_1d_merge": "average", "common_space_fraction": 0.0},
        )
        assert torch.allclose(avg_prepared[1]["linear.bias"], expected_bias_delta)


def test_tsv_average_1d_without_2d_keys() -> None:
    base = {"bias": torch.tensor([1.0, -1.0])}
    tuned = [
        {"bias": torch.tensor([2.0, 1.0])},
        {"bias": torch.tensor([4.0, -3.0])},
    ]

    method = get_method("tsv_merge")
    prepared = method.prepare(base=base, tuned=tuned, strict=True, method_params={"vector_1d_merge": "average"})

    assert torch.allclose(prepared[1]["bias"], torch.tensor([2.0, 0.0]))

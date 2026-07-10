from __future__ import annotations

import json
import math

import pytest
import torch

from merge_and_rebase.merge.runtime import build_merged_state_for_alpha
from merge_and_rebase.merge.subspaces.geo_core import GeodesicCoreSpace
from merge_and_rebase.merge.subspaces.registry import get_subspace


def _peft_state_for_layer(prefix: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": a,
        f"{prefix}.lora_B.weight": b,
    }


def _rank1_basis(angle_radians: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle_radians)],
            [math.sin(angle_radians)],
        ],
        dtype=torch.float32,
    )


def _subspace_angle_to_e1(basis: torch.Tensor) -> float:
    unit = basis[:, 0] / basis[:, 0].norm()
    cosine = float(unit[0].abs().clamp(max=1.0).item())
    return math.acos(cosine)


def test_geo_core_registered() -> None:
    method = get_subspace("geo_core")
    assert isinstance(method, GeodesicCoreSpace)


def test_geo_core_single_task_roundtrip_reconstructs_update() -> None:
    torch.manual_seed(0)
    layer = "visual.transformer.resblocks.0.attn.q_proj"
    a = torch.randn(2, 5)
    b = torch.randn(7, 2)
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a, b),
    }

    geo = GeodesicCoreSpace()
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 8},
    )
    projected = geo.project(
        prepared,
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 8},
    )
    lifted = geo.lift(
        prepared,
        merged_core={layer: projected["t1"][layer]},
        lora_template=lora_by_task["t1"],
        peft_cfg={"lora_alpha": 8},
    )

    assert torch.allclose(lifted[f"{layer}.weight"], b @ a, atol=1e-5, rtol=1e-5)


def test_geo_core_shared_subspace_roundtrip_reconstructs_each_task() -> None:
    layer = "visual.transformer.resblocks.0.attn.v_proj"
    u = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    a1 = torch.tensor([[3.0, 1.5, 0.75, 2.25]], dtype=torch.float32)
    b1 = 2.0 * u
    a2 = torch.tensor([[1.0, 0.5, 0.25, 0.75]], dtype=torch.float32)
    b2 = 5.0 * u
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    geo = GeodesicCoreSpace()
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
    )
    projected = geo.project(
        prepared,
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
    )

    for task, expected_a, expected_b in (("t1", a1, b1), ("t2", a2, b2)):
        lifted = geo.lift(
            prepared,
            merged_core={layer: projected[task][layer]},
            lora_template=lora_by_task["t1"],
            peft_cfg={"lora_alpha": 4},
        )
        assert torch.allclose(lifted[f"{layer}.weight"], expected_b @ expected_a, atol=1e-5, rtol=1e-5)


def test_geo_core_core_referenced_tangent_single_task_roundtrip_reconstructs_update() -> None:
    torch.manual_seed(2)
    layer = "visual.transformer.resblocks.0.attn.out_proj"
    a = torch.randn(2, 4)
    b = torch.randn(6, 2)
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a, b),
    }

    geo = GeodesicCoreSpace()
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 8},
        method_params={"geo_core_variant": "core_referenced_tangent"},
    )
    projected = geo.project(
        prepared,
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 8},
    )
    lifted = geo.lift(
        prepared,
        merged_core={layer: projected["t1"][layer]},
        lora_template=lora_by_task["t1"],
        peft_cfg={"lora_alpha": 8},
    )

    assert torch.allclose(lifted[f"{layer}.weight"], b @ a, atol=1e-5, rtol=1e-5)


def test_geo_core_equal_weighting_uses_incremental_one_over_t_schedule() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_mean_weighting": "equal"},
    )

    angle = _subspace_angle_to_e1(prepared.bases[layer]["U"])
    assert angle == pytest.approx((2.0 * math.pi) / 9.0, abs=1e-5)


def test_geo_core_merge_weighted_mean_moves_toward_heavier_task() -> None:
    layer = "visual.transformer.resblocks.0.attn.o_proj"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }
    geo = GeodesicCoreSpace()

    equal_prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_mean_weighting": "equal"},
    )
    weighted_prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_mean_weighting": "merge_weights"},
        weights=[1.0, 9.0, 1.0],
    )

    equal_angle = _subspace_angle_to_e1(equal_prepared.bases[layer]["U"])
    weighted_angle = _subspace_angle_to_e1(weighted_prepared.bases[layer]["U"])
    assert weighted_angle > equal_angle
    assert weighted_angle > (2.0 * math.pi) / 9.0
    assert weighted_angle < target_angle


def test_geo_core_core_referenced_tangent_weights_move_basis_toward_heavier_task() -> None:
    layer = "visual.transformer.resblocks.0.attn.fc1"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }
    geo = GeodesicCoreSpace()

    equal_prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_referenced_tangent"},
    )
    weighted_prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_referenced_tangent"},
        weights=[1.0, 9.0, 1.0],
    )

    equal_angle = _subspace_angle_to_e1(equal_prepared.bases[layer]["U"])
    weighted_angle = _subspace_angle_to_e1(weighted_prepared.bases[layer]["U"])
    assert weighted_angle > equal_angle
    assert weighted_angle < target_angle


def test_geo_core_merge_weighted_mean_rejects_negative_weights() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="non-negative weights"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={"geo_mean_weighting": "merge_weights"},
            weights=[1.0, -1.0],
        )


def test_geo_core_core_referenced_tangent_rejects_negative_weights() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="non-negative weights"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={"geo_core_variant": "core_referenced_tangent"},
            weights=[1.0, -1.0],
        )


def test_geo_core_core_similarity_weights_computes_geometry_aware_weight_override() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_similarity_weights"},
        weights=[1.0, 1.0, 1.0],
    )

    assert prepared.merge_weight_override is not None
    w1, w2, w3 = prepared.merge_weight_override
    assert w1 == pytest.approx(1.0 / 18.0, abs=1e-6)
    assert w2 == pytest.approx(17.0 / 36.0, abs=1e-6)
    assert w3 == pytest.approx(17.0 / 36.0, abs=1e-6)


def test_geo_core_core_similarity_weights_lambda_zero_keeps_core_weights() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={
            "geo_core_variant": "core_similarity_weights",
            "geo_weight_lambda": 0.0,
        },
        weights=[2.0, 1.0, 1.0],
    )

    assert prepared.merge_weight_override == pytest.approx((0.5, 0.25, 0.25), abs=1e-6)


def test_geo_core_core_similarity_weights_lambda_half_blends_core_and_geometry() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    zero = 0.0
    target_angle = math.pi / 3.0
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(zero).T, _rank1_basis(zero)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
        "t3": _peft_state_for_layer(layer, _rank1_basis(target_angle).T, _rank1_basis(target_angle)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={
            "geo_core_variant": "core_similarity_weights",
            "geo_weight_lambda": 0.5,
        },
        weights=[1.0, 1.0, 1.0],
    )

    assert prepared.merge_weight_override is not None
    w1, w2, w3 = prepared.merge_weight_override
    assert w1 == pytest.approx((1.0 / 3.0 + 1.0 / 18.0) / 2.0, abs=1e-6)
    assert w2 == pytest.approx((1.0 / 3.0 + 17.0 / 36.0) / 2.0, abs=1e-6)
    assert w3 == pytest.approx((1.0 / 3.0 + 17.0 / 36.0) / 2.0, abs=1e-6)


def test_geo_core_core_similarity_weights_falls_back_when_geometry_has_no_signal() -> None:
    layer = "visual.transformer.resblocks.0.attn.v_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(0.0).T, _rank1_basis(0.0)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(math.pi / 2.0).T, _rank1_basis(math.pi / 2.0)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_similarity_weights"},
        weights=[2.0, 1.0],
    )

    assert prepared.merge_weight_override == pytest.approx((2.0 / 3.0, 1.0 / 3.0), abs=1e-6)


def test_geo_core_core_similarity_weights_rejects_lambda_out_of_range() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="geo_weight_lambda"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={
                "geo_core_variant": "core_similarity_weights",
                "geo_weight_lambda": 1.5,
            },
            weights=[1.0, 1.0],
        )


def test_geo_core_core_similarity_mask_single_task_roundtrip_reconstructs_update() -> None:
    torch.manual_seed(3)
    layer = "visual.transformer.resblocks.0.attn.proj"
    a = torch.randn(2, 4)
    b = torch.randn(5, 2)
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a, b),
    }

    geo = GeodesicCoreSpace()
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_similarity_mask"},
    )
    projected = geo.project(prepared, lora_by_task=lora_by_task, peft_cfg={"lora_alpha": 4})
    lifted = geo.lift(
        prepared,
        merged_core={layer: projected["t1"][layer]},
        lora_template=lora_by_task["t1"],
        peft_cfg={"lora_alpha": 4},
    )

    assert torch.allclose(lifted[f"{layer}.weight"], b @ a, atol=1e-5, rtol=1e-5)


def test_geo_core_core_similarity_mask_lambda_zero_is_noop() -> None:
    layer = "visual.transformer.resblocks.0.attn.mask0"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(0.0).T, _rank1_basis(0.0)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(math.pi / 3.0).T, _rank1_basis(math.pi / 3.0)),
    }

    geo = GeodesicCoreSpace()
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={
            "geo_core_variant": "core_similarity_mask",
            "geo_mask_lambda": 0.0,
        },
    )

    assert prepared.task_masks is not None
    for task in lora_by_task:
        mask = prepared.task_masks[task][layer]
        assert torch.allclose(mask, torch.ones_like(mask), atol=1e-6, rtol=1e-6)


def test_geo_core_core_similarity_mask_suppresses_incompatible_rows_and_cols() -> None:
    layer = "visual.transformer.resblocks.0.attn.mask1"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(0.0).T, _rank1_basis(0.0)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(math.pi / 2.0).T, _rank1_basis(math.pi / 2.0)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_similarity_mask"},
    )

    assert prepared.task_masks is not None
    mask_t1 = prepared.task_masks["t1"][layer]
    mask_t2 = prepared.task_masks["t2"][layer]
    assert float(mask_t1.min().item()) < 1.0
    assert float(mask_t2.min().item()) < 1.0


def test_geo_core_core_similarity_mask_rejects_lambda_out_of_range() -> None:
    layer = "visual.transformer.resblocks.0.attn.mask2"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="geo_mask_lambda"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={
                "geo_core_variant": "core_similarity_mask",
                "geo_mask_lambda": -0.1,
            },
        )


def test_geo_core_core_similarity_mask_rejects_unknown_support_mode() -> None:
    layer = "visual.transformer.resblocks.0.attn.mask_support_bad"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="geo_mask_support"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={
                "geo_core_variant": "core_similarity_mask",
                "geo_mask_support": "mystery",
            },
        )


def test_geo_core_core_similarity_mask_factor_support_is_magnitude_aware() -> None:
    layer = "visual.transformer.resblocks.0.attn.mask_factor"
    base_u = _rank1_basis(0.0)
    base_v_t = _rank1_basis(0.0).T
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, base_v_t.clone(), 2.0 * base_u),
        "t2": _peft_state_for_layer(layer, base_v_t.clone(), 1.0 * base_u),
    }

    prepared_subspace = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={
            "geo_core_variant": "core_similarity_mask",
            "geo_mask_support": "subspace",
        },
    )
    prepared_factor = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={
            "geo_core_variant": "core_similarity_mask",
            "geo_mask_support": "factor",
        },
    )

    assert prepared_subspace.task_masks is not None
    assert prepared_factor.task_masks is not None
    subspace_mask_t1 = prepared_subspace.task_masks["t1"][layer]
    factor_mask_t1 = prepared_factor.task_masks["t1"][layer]

    assert torch.allclose(subspace_mask_t1, torch.ones_like(subspace_mask_t1), atol=1e-6, rtol=1e-6)
    assert float(factor_mask_t1.min().item()) < 1.0
    assert prepared_factor.mask_support_mode == "factor"


def test_geo_core_core_similarity_mask_saves_similarity_artifact(tmp_path) -> None:
    layer = "visual.transformer.resblocks.0.attn.mask_save"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, _rank1_basis(0.0).T, _rank1_basis(0.0)),
        "t2": _peft_state_for_layer(layer, _rank1_basis(math.pi / 3.0).T, _rank1_basis(math.pi / 3.0)),
    }

    prepared = GeodesicCoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params={"geo_core_variant": "core_similarity_mask"},
        artifact_dir=tmp_path,
    )

    assert prepared.similarity_artifact_path is not None
    artifact_path = tmp_path / "geo_core_similarity.json"
    assert prepared.similarity_artifact_path == str(artifact_path)
    with artifact_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["tasks"] == ["t1", "t2"]
    assert payload["num_layers"] == 1
    assert payload["mask_support_mode"] == "subspace"
    assert payload["mean_joint_similarity"][0][0] == pytest.approx(1.0, abs=1e-6)
    assert payload["mean_joint_similarity"][1][1] == pytest.approx(1.0, abs=1e-6)
    assert payload["layers"][layer]["joint_similarity"][0][1] < 1.0


def test_geo_core_core_posterior_refines_toward_task_supported_cores() -> None:
    layer = "visual.transformer.resblocks.0.attn.posterior"
    lora_by_task = {
        "t1": _peft_state_for_layer(
            layer,
            torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        ),
        "t2": _peft_state_for_layer(
            layer,
            torch.tensor([[0.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        ),
    }

    geo = GeodesicCoreSpace()
    method_params = {
        "geo_core_variant": "core_posterior",
        "geo_posterior_tau": 1.0,
        "geo_posterior_tol": 1e-10,
        "geo_posterior_max_iter": 50,
    }
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params=method_params,
        weights=[1.0, 1.0],
    )
    projected = geo.project(prepared, lora_by_task=lora_by_task, peft_cfg={"lora_alpha": 4})

    refined = geo.refine_merged_core(
        prepared,
        merged_core={layer: torch.zeros((2, 2), dtype=torch.float32)},
        tuned_cores=[projected["t1"], projected["t2"]],
        weights=[1.0, 1.0],
        method_params=method_params,
        tasks=["t1", "t2"],
    )

    expected = torch.tensor([[2.0 / 3.0, 0.0], [0.0, 4.0 / 3.0]], dtype=torch.float32)
    assert torch.allclose(refined[layer], expected, atol=1e-5, rtol=1e-5)


def test_geo_core_core_posterior_rejects_negative_weights() -> None:
    layer = "visual.transformer.resblocks.0.attn.posterior_bad"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
        "t2": _peft_state_for_layer(layer, torch.randn(1, 2), torch.randn(2, 1)),
    }

    with pytest.raises(ValueError, match="non-negative weights"):
        GeodesicCoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={"geo_core_variant": "core_posterior"},
            weights=[1.0, -1.0],
        )


def test_build_merged_state_for_alpha_applies_geo_core_posterior_refinement() -> None:
    class _FixedCoreMethod:
        def apply(self, prepared, *, alpha: float, **kwargs):
            _ = alpha
            _ = kwargs
            return prepared

    layer = "visual.transformer.resblocks.0.attn.posterior_runtime"
    lora_by_task = {
        "t1": _peft_state_for_layer(
            layer,
            torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        ),
        "t2": _peft_state_for_layer(
            layer,
            torch.tensor([[0.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        ),
    }
    tasks = ["t1", "t2"]
    geo = GeodesicCoreSpace()
    method_params = {
        "geo_core_variant": "core_posterior",
        "geo_posterior_tau": 1.0,
        "geo_posterior_tol": 1e-10,
        "geo_posterior_max_iter": 50,
    }
    prepared = geo.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params=method_params,
        weights=[1.0, 1.0],
    )
    projected = geo.project(prepared, lora_by_task=lora_by_task, peft_cfg={"lora_alpha": 4})

    merged_sd = build_merged_state_for_alpha(
        method=_FixedCoreMethod(),
        prepared={layer: torch.zeros((2, 2), dtype=torch.float32)},
        base_sd_for_merge={layer: torch.zeros((2, 2), dtype=torch.float32)},
        tuned_sds_list=[projected["t1"], projected["t2"]],
        weights=[1.0, 1.0],
        method_params=method_params,
        alpha=1.0,
        peft_subspace="geo_core",
        subspace=geo,
        subspace_prepared=prepared,
        peft_cfg={"lora_alpha": 4},
        peft_state_by_task=lora_by_task,
        tasks=tasks,
        merge_base_sd={f"{layer}.weight": torch.zeros((2, 2), dtype=torch.float32)},
    )

    expected = torch.tensor([[2.0 / 3.0, 0.0], [0.0, 4.0 / 3.0]], dtype=torch.float32)
    assert torch.allclose(merged_sd[f"{layer}.weight"], expected, atol=1e-5, rtol=1e-5)

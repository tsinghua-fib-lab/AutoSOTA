from __future__ import annotations

import pytest
import torch

from merge_and_rebase.eval.vision_connectivity import _load_checkpoint_payload
from merge_and_rebase.models.forward_modes import resolve_shared_forward_mode_params
from merge_and_rebase.eval.vision_merge import (
    _acc_cache_key,
    _assert_qkv_patched_before_linearizing,
    _extract_checkpoint_attn_patch_info,
    _resolve_zero_shot_only,
)


def test_extract_checkpoint_attn_patch_info_requires_cfg_when_patched() -> None:
    with pytest.raises(ValueError, match="patched_attn=True"):
        _extract_checkpoint_attn_patch_info(
            obj={"patched_attn": True},
            ckpt_path="dummy.pt",
        )


def test_extract_checkpoint_attn_patch_info_rejects_unlabeled_patched_state_dict() -> None:
    with pytest.raises(ValueError, match="lacks patched_attn/attn_patch_cfg metadata"):
        _extract_checkpoint_attn_patch_info(
            obj={
                "state_dict": {
                    "visual.transformer.resblocks.0.attn.q_proj.weight": 0,
                    "visual.transformer.resblocks.0.attn.k_proj.weight": 0,
                    "visual.transformer.resblocks.0.attn.v_proj.weight": 0,
                }
            },
            ckpt_path="dummy.pt",
        )


def test_extract_checkpoint_attn_patch_info_rejects_patched_false_with_split_qkv() -> None:
    with pytest.raises(ValueError, match="patched_attn=False"):
        _extract_checkpoint_attn_patch_info(
            obj={
                "patched_attn": False,
                "state_dict": {
                    "visual.transformer.resblocks.0.attn.q_proj.weight": 0,
                    "visual.transformer.resblocks.0.attn.k_proj.weight": 0,
                    "visual.transformer.resblocks.0.attn.v_proj.weight": 0,
                },
            },
            ckpt_path="dummy.pt",
        )


def test_extract_checkpoint_attn_patch_info_uses_cfg_when_patched_false_but_split_qkv() -> None:
    meta = _extract_checkpoint_attn_patch_info(
        obj={
            "patched_attn": False,
            "attn_patch_cfg": {"attn_impl": "linear", "kernel": "exp", "eps": 1e-4},
            "state_dict": {
                "visual.transformer.resblocks.0.attn.q_proj.weight": 0,
                "visual.transformer.resblocks.0.attn.k_proj.weight": 0,
                "visual.transformer.resblocks.0.attn.v_proj.weight": 0,
            },
        },
        ckpt_path="dummy.pt",
    )
    assert meta.patched_attn is True
    assert meta.attn_patch_cfg is not None
    assert meta.attn_patch_cfg["attn_impl"] == "linear"
    assert meta.linearized_attn is True


def test_assert_qkv_patched_before_linearizing_requires_patch_first() -> None:
    with pytest.raises(RuntimeError, match="was not patched first"):
        _assert_qkv_patched_before_linearizing(
            needs_linear_attention=True,
            base_patched_for_attn=False,
            model_state_dict={},
        )


def test_assert_qkv_patched_before_linearizing_rejects_fused_qkv() -> None:
    with pytest.raises(RuntimeError, match="not fully q/k/v patched"):
        _assert_qkv_patched_before_linearizing(
            needs_linear_attention=True,
            base_patched_for_attn=True,
            model_state_dict={"visual.transformer.resblocks.0.attn.in_proj_weight": 0},
        )


def test_assert_qkv_patched_before_linearizing_accepts_split_qkv() -> None:
    _assert_qkv_patched_before_linearizing(
        needs_linear_attention=True,
        base_patched_for_attn=True,
        model_state_dict={
            "visual.transformer.resblocks.0.attn.q_proj.weight": 0,
            "visual.transformer.resblocks.0.attn.k_proj.weight": 0,
            "visual.transformer.resblocks.0.attn.v_proj.weight": 0,
        },
    )


def test_acc_cache_key_differs_by_baseline_mode() -> None:
    tuned_key = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="task_a.pt",
        baseline_mode="tuned",
        forward_mode="standard",
        forward_mode_params={},
        classnames_mode="raw",
    )
    zero_shot_key = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="open_clip_pretrained",
        baseline_mode="zero_shot",
        forward_mode="standard",
        forward_mode_params={},
        classnames_mode="raw",
    )
    assert tuned_key != zero_shot_key


def test_acc_cache_key_differs_by_text_features_mode() -> None:
    zs_key = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="task_a.pt",
        baseline_mode="tuned",
        forward_mode="standard",
        forward_mode_params={},
        classnames_mode="raw",
        text_features_mode="zero_shot",
    )
    tuned_key = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="task_a.pt",
        baseline_mode="tuned",
        forward_mode="standard",
        forward_mode_params={},
        classnames_mode="raw",
        text_features_mode="tuned_ckpt",
    )
    assert zs_key != tuned_key


def test_acc_cache_key_differs_by_forward_mode_params() -> None:
    key_true = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="task_a.pt",
        baseline_mode="tuned",
        forward_mode="linearized_ntk",
        forward_mode_params={"linearized_feature_normalization": True},
        classnames_mode="raw",
    )
    key_false = _acc_cache_key(
        "ViT-B-32",
        "openai",
        "cifar10",
        chk_path="task_a.pt",
        baseline_mode="tuned",
        forward_mode="linearized_ntk",
        forward_mode_params={"linearized_feature_normalization": False},
        classnames_mode="raw",
    )
    assert key_true != key_false


def test_resolve_shared_forward_mode_params_rejects_conflicts() -> None:
    with pytest.raises(ValueError, match="Inconsistent forward_mode_params"):
        resolve_shared_forward_mode_params(
            "linearized_ntk",
            [
                {"linearized_feature_normalization": True},
                {"linearized_feature_normalization": False},
            ],
        )


def test_checkpoint_payload_loads_forward_mode_params(tmp_path) -> None:
    path = tmp_path / "linearized.pt"
    torch.save(
        {
            "strategy": "full",
            "forward_mode": "linearized_ntk",
            "forward_mode_params": {"linearized_feature_normalization": False},
            "patched_attn": False,
            "tuned_text_features": torch.eye(2),
        },
        path,
    )

    payload = _load_checkpoint_payload(str(path))
    assert payload.forward_mode == "linearized_ntk"
    assert payload.forward_mode_params == {"linearized_feature_normalization": False}


def test_resolve_zero_shot_only_defaults_true_without_tuned_ckpts() -> None:
    assert _resolve_zero_shot_only({"tuned_ckpts": None}) is True


def test_resolve_zero_shot_only_respects_explicit_and_checkpoint_cases() -> None:
    assert _resolve_zero_shot_only({"zero_shot_only": True, "tuned_ckpts": {"Cars": "cars.pt"}}) is True
    assert _resolve_zero_shot_only({"zero_shot_only": False, "tuned_ckpts": {"Cars": "cars.pt"}}) is False

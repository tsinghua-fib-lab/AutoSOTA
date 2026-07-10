from __future__ import annotations

from types import MethodType

import pytest
import torch
import torch.nn as nn

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_extract_tuned_text_features_from_checkpoint_returns_cpu_tensor() -> None:
    obj = {"tuned_text_features": torch.randn(3, 4)}
    feats = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(obj=obj, ckpt_path="dummy.pt")
    assert isinstance(feats, torch.Tensor)
    assert tuple(feats.shape) == (3, 4)
    assert feats.device.type == "cpu"


def test_extract_tuned_text_features_from_checkpoint_ignores_context_only_payload() -> None:
    obj = {"tuned_prompt_context": torch.randn(2, 4)}
    feats = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(obj=obj, ckpt_path="dummy.pt")
    assert feats is None


def test_resolve_eval_text_features_auto_prefers_tuned_features() -> None:
    clf = OpenClipClassifier(model=_DummyModel(), tokenizer=None, preprocess=None)
    calls: list[int] = []

    def _fake_build(self, classnames, cfg, **kwargs):
        del cfg, kwargs
        calls.append(len(classnames))

    clf.build_zeroshot_text_features = MethodType(_fake_build, clf)
    tuned = torch.randn(2, 8)
    feats, mode = clf.resolve_eval_text_features(
        text_features_source="auto",
        classnames=["cat", "dog"],
        build_cfg=OpenClipBuildConfig(),
        tuned_text_features=tuned,
    )
    assert mode == "tuned_ckpt"
    assert feats is not None
    assert torch.allclose(feats, tuned.cpu())
    assert calls == []


def test_resolve_eval_text_features_tuned_ckpt_requires_present_features() -> None:
    clf = OpenClipClassifier(model=_DummyModel(), tokenizer=None, preprocess=None)
    with pytest.raises(ValueError, match="has no tuned_text_features"):
        clf.resolve_eval_text_features(
            text_features_source="tuned_ckpt",
            classnames=["cat", "dog"],
            build_cfg=OpenClipBuildConfig(),
            tuned_text_features=None,
            task_name="Pets",
            ckpt_path="pets.pt",
        )


def test_top1_with_text_features_restores_classifier_state() -> None:
    clf = OpenClipClassifier(model=_DummyModel(), tokenizer=None, preprocess=None, normalize=False, logit_scale=1.0)
    original = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf._zs_text_features = original.clone()
    clf._zs_text_fingerprint = "orig"

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    loader = [(x, y)]
    acc = clf.top1_with_text_features(
        loader,
        device="cpu",
        text_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        expected_num_classes=2,
    )

    assert abs(acc - 1.0) < 1e-8
    assert torch.allclose(clf._zs_text_features, original)
    assert clf._zs_text_fingerprint == "orig"

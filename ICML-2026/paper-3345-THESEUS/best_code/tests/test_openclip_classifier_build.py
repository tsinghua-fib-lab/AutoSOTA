from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier, zero_shot_logits_from_features


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))


def test_build_uses_hf_hub_model_name_without_pretrained_tag(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["device"] = device
        calls["quick_gelu"] = quick_gelu
        return _DummyModel(), None, "preprocess"

    def _fake_get_tokenizer(model_name):
        calls["tokenizer_model_name"] = model_name
        return "tokenizer"

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=_fake_get_tokenizer,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    cfg = OpenClipBuildConfig(model_name="hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K", pretrained="openai")
    clf = OpenClipClassifier.build(cfg)

    assert isinstance(clf, OpenClipClassifier)
    assert clf.preprocess == "preprocess"
    assert clf.train_preprocess == "preprocess"
    assert calls == {
        "model_name": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        "pretrained": None,
        "device": "cuda",
        "quick_gelu": False,
        "tokenizer_model_name": "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    }


def test_build_stores_train_and_eval_preprocess(monkeypatch) -> None:
    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        return _DummyModel(), "train_preprocess", "eval_preprocess"

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=lambda model_name: "tokenizer",
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    clf = OpenClipClassifier.build(OpenClipBuildConfig())

    assert clf.train_preprocess == "train_preprocess"
    assert clf.preprocess == "eval_preprocess"


def test_build_accepts_hf_hub_reference_in_pretrained_field(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        calls["device"] = device
        calls["quick_gelu"] = quick_gelu
        return _DummyModel(), None, "preprocess"

    def _fake_get_tokenizer(model_name):
        calls["tokenizer_model_name"] = model_name
        return "tokenizer"

    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=_fake_get_tokenizer,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    OpenClipClassifier.build(cfg)

    assert calls == {
        "model_name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "pretrained": None,
        "device": "cuda",
        "quick_gelu": False,
        "tokenizer_model_name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    }


def test_build_openai_clip_loader_uses_legacy_clip_and_openclip_train_transform(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _LegacyClipModel(_DummyModel):
        pass

    def _fake_clip_load(model_name, device=None, jit=None):
        calls["clip_model_name"] = model_name
        calls["clip_device"] = device
        calls["clip_jit"] = jit
        return _LegacyClipModel(), "legacy_eval_preprocess"

    def _fake_create_model_and_transforms(model_name, *, pretrained=None, device=None, quick_gelu=None):
        calls["openclip_model_name"] = model_name
        calls["openclip_pretrained"] = pretrained
        calls["openclip_device"] = device
        calls["openclip_quick_gelu"] = quick_gelu
        return _DummyModel(), "openclip_train_preprocess", "openclip_eval_preprocess"

    fake_clip = SimpleNamespace(load=_fake_clip_load, tokenize=lambda texts: texts)
    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=_fake_create_model_and_transforms,
        get_tokenizer=lambda model_name: "unused_tokenizer",
    )
    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    clf = OpenClipClassifier.build(
        OpenClipBuildConfig(loader="openai_clip", model_name="ViT-B-32", pretrained="openai", device="cpu")
    )

    assert isinstance(clf, OpenClipClassifier)
    assert clf.preprocess == "legacy_eval_preprocess"
    assert clf.train_preprocess == "openclip_train_preprocess"
    assert calls == {
        "clip_model_name": "ViT-B/32",
        "clip_device": "cpu",
        "clip_jit": False,
        "openclip_model_name": "ViT-B-32",
        "openclip_pretrained": "openai",
        "openclip_device": "cpu",
        "openclip_quick_gelu": True,
    }


def test_zero_shot_logits_aligns_text_features_dtype_with_image_features() -> None:
    classifier = SimpleNamespace(
        _zs_text_features=torch.randn(3, 4, dtype=torch.float32),
        normalize=False,
        logit_scale=1.0,
    )
    image_features = torch.randn(2, 4, dtype=torch.float16)

    logits = zero_shot_logits_from_features(classifier, image_features, normalize_image_features=False)

    assert logits.dtype == torch.float16
    assert logits.shape == (2, 3)

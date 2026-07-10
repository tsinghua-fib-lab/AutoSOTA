from types import MethodType

import torch
import torch.nn as nn

from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))


def test_zeroshot_features_rebuild_when_request_changes():
    clf = OpenClipClassifier(model=_DummyModel(), tokenizer=None, preprocess=None)

    calls: list[int] = []

    def _fake_compute(self, classnames, cfg):
        calls.append(len(classnames))
        return torch.full((len(classnames), 2), float(len(calls)))

    clf._compute_zeroshot_text_features = MethodType(_fake_compute, clf)

    cfg = OpenClipBuildConfig(prompt_templates=[lambda c: f"a photo of {c}"])

    clf.build_zeroshot_text_features(["cat", "dog", "car"], cfg, cache_dir=None, force_rebuild=False)
    assert tuple(clf._zs_text_features.shape) == (3, 2)
    assert calls == [3]

    # Different classnames => different fingerprint => must rebuild.
    clf.build_zeroshot_text_features(["apple", "banana"], cfg, cache_dir=None, force_rebuild=False)
    assert tuple(clf._zs_text_features.shape) == (2, 2)
    assert calls == [3, 2]

    # Same request => reuse in-memory features.
    clf.build_zeroshot_text_features(["apple", "banana"], cfg, cache_dir=None, force_rebuild=False)
    assert calls == [3, 2]

    # Explicit force rebuild still recomputes.
    clf.build_zeroshot_text_features(["apple", "banana"], cfg, cache_dir=None, force_rebuild=True)
    assert calls == [3, 2, 2]

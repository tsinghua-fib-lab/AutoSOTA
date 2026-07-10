from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage
from PIL import Image

from merge_and_rebase.data.vision_loaders import KMNIST_CLASSNAMES, build_vision_loaders, emnist_fix_transform, load_hf_splits


def _identity(x):
    return x


def test_load_hf_splits_loads_only_requested_splits(monkeypatch) -> None:
    calls: list[tuple[str | None, str | None]] = []
    datasets_by_split = {
        "train": Dataset.from_dict({"label": [0, 1]}),
        "test": Dataset.from_dict({"label": [1, 0]}),
    }

    def _fake_load_dataset(path, *args, split=None, **kwargs):
        config = args[0] if args else None
        calls.append((config, split))
        if split is None:
            raise AssertionError("Fallback loading should not run when requested splits load directly.")
        return datasets_by_split[split]

    monkeypatch.setattr("merge_and_rebase.data.vision_loaders.hf_load_dataset", _fake_load_dataset)

    ds = load_hf_splits("tanganke/sun397", requested_splits=("train", "test"))

    assert list(ds.keys()) == ["train", "test"]
    assert calls == [(None, "train"), (None, "test")]


def test_load_hf_splits_rejects_unsupported_requested_split() -> None:
    with pytest.raises(ValueError, match="Unsupported requested splits"):
        load_hf_splits("tanganke/sun397", requested_splits=("train", "foo"))


def test_emnist_fix_transform_is_picklable() -> None:
    img = Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8), mode="L")
    transform = emnist_fix_transform(_identity)

    pickle.dumps(transform)

    out = torch.from_numpy(np.array(transform(img)))
    expected = torch.tensor([[0, 2], [1, 3]], dtype=torch.uint8)

    assert torch.equal(out, expected)


def test_build_vision_loaders_uses_miil_tree_classnames_for_imagenet21kp(monkeypatch) -> None:
    train_labels = [0, 1] * 10
    validation_labels = [1, 0] * 10
    hf_ds = DatasetDict(
        {
            "train": Dataset.from_dict({"label": train_labels}),
            "validation": Dataset.from_dict({"label": validation_labels}),
        }
    )
    monkeypatch.setattr(
        "merge_and_rebase.data.vision_loaders._imagenet21kp_classnames_from_tree",
        lambda: ["alpha", "beta"],
    )

    loaders = build_vision_loaders(
        hf_ds,
        hf_path="timm/imagenet-w21-p",
        preprocess=_identity,
        ft_epochs=1,
        split_map={"train": "train", "test": "validation"},
        batch_size=2,
        num_workers=0,
    )

    assert list(loaders.classnames) == ["alpha", "beta"]
    assert list(loaders.train.dataset.classes) == ["alpha", "beta"]


def test_build_vision_loaders_imagenet21kp_falls_back_to_numeric_when_non_strict(monkeypatch) -> None:
    train_labels = [0, 1] * 10
    validation_labels = [1, 0] * 10
    hf_ds = DatasetDict(
        {
            "train": Dataset.from_dict({"label": train_labels}),
            "validation": Dataset.from_dict({"label": validation_labels}),
        }
    )

    def _raise_missing_tree():
        raise FileNotFoundError("missing tree")

    monkeypatch.setattr(
        "merge_and_rebase.data.vision_loaders._imagenet21kp_classnames_from_tree",
        _raise_missing_tree,
    )

    loaders = build_vision_loaders(
        hf_ds,
        hf_path="timm/imagenet-w21-p",
        preprocess=_identity,
        ft_epochs=1,
        split_map={"train": "train", "test": "validation"},
        batch_size=2,
        num_workers=0,
        strict_classnames=False,
    )

    assert list(loaders.classnames) == ["0", "1"]


def test_build_vision_loaders_can_use_train_transform_for_train_only() -> None:
    hf_ds = DatasetDict(
        {
            "train": Dataset.from_dict({"image": [0, 1, 2, 3], "label": [0, 1, 0, 1]}),
            "test": Dataset.from_dict({"image": [4, 5, 6, 7], "label": [0, 1, 0, 1]}),
        }
    )

    loaders = build_vision_loaders(
        hf_ds,
        hf_path="dummy",
        preprocess=lambda _: torch.tensor([2.0]),
        train_preprocess=lambda _: torch.tensor([1.0]),
        ft_epochs=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.5,
        classnames_override=["a", "b"],
    )

    train_x, _ = next(iter(loaders.train))
    val_x, _ = next(iter(loaders.val))
    test_x, _ = next(iter(loaders.test))

    assert torch.equal(train_x, torch.ones_like(train_x))
    assert torch.equal(val_x, torch.full_like(val_x, 2.0))
    assert torch.equal(test_x, torch.full_like(test_x, 2.0))

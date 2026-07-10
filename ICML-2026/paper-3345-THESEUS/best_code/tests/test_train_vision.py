from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from merge_and_rebase.finetune import train_vision


def test_build_parser_accepts_train_preprocess_override() -> None:
    args = train_vision.build_parser().parse_args(
        ["--vision-config", "config.yaml", "--train-preprocess", "train"]
    )

    assert args.train_preprocess == "train"


def test_build_parser_accepts_vision_loader_overrides() -> None:
    args = train_vision.build_parser().parse_args(
        [
            "--vision-config",
            "config.yaml",
            "--vision-loader-profile",
            "hf",
            "--vision-data-root",
            "/tmp/vision-data",
        ]
    )

    assert args.vision_loader_profile == "hf"
    assert args.vision_data_root == "/tmp/vision-data"


def test_build_parser_accepts_force_recompute_override() -> None:
    args = train_vision.build_parser().parse_args(["--vision-config", "config.yaml", "--force-recompute"])

    assert args.force_recompute is True


def test_resolve_train_preprocess_uses_cli_override() -> None:
    assert train_vision._resolve_train_preprocess(None) == "eval"
    assert train_vision._resolve_train_preprocess("train") == "train"
    assert train_vision._resolve_train_preprocess("eval", cli_value="train", task="Cars") == "train"

    with pytest.raises(ValueError, match=r"\[Cars\] data\.train_preprocess"):
        train_vision._resolve_train_preprocess("augment", task="Cars")


def test_resolve_dense_lr_prefers_optimizer_override_then_train_override_then_lr() -> None:
    assert train_vision._resolve_dense_lr({"train": {"lr": 1e-4}}, default_lr=1e-4) == pytest.approx(1e-4)
    assert train_vision._resolve_dense_lr({"train": {"lr": 1e-4, "dense_lr": 3e-5}}, default_lr=1e-4) == pytest.approx(3e-5)
    assert train_vision._resolve_dense_lr(
        {"train": {"lr": 1e-4, "dense_lr": 3e-5, "optimizer": {"dense_lr": 7e-5}}},
        default_lr=1e-4,
    ) == pytest.approx(7e-5)


def test_set_seed_seeds_python_numpy_and_torch() -> None:
    train_vision._set_seed(123)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    train_vision._set_seed(123)
    second_python = random.random()
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == second_python
    np.testing.assert_allclose(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)
